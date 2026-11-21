"""Passo 7 da fase 1: treino da BiGRU com uma trajetória completa por amostra.

O que este script garante:
* loss de BCE **mascarada**: o padding de um batch nunca entra na conta;
* sem pesos de classe nem reamostragem (decisão da proposta);
* o early stopping e a escolha do estado final olham só para o **F1 macro da validação**;
  o teste não é tocado aqui, nem a divisão por navio (só por pedido explícito);
* reprodutibilidade: semente fixa, ``torch.set_num_threads`` e registo época a época com tempos.

Comandos:
    python src/treinar.py --exp EXP-001_features_com_xy --semente 42
    python src/treinar.py --sanidade       # overfit de 5 trajetórias curtas
    python src/treinar.py --determinismo   # dois treinos iguais dão o mesmo resultado?
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from avaliar import (
    PASTA_PREVISOES,
    avaliar_previsoes,
    construir_previsoes,
    escrever_previsoes,
    guardar_avaliacao,
    imprimir_avaliacao,
    metricas_ponto,
)
from dataset import (
    DIVISOES,
    DatasetTrajetorias,
    carregar_dataset,
    iterar_batches,
)
from modelos import BiGRU, loss_mascarada, probabilidades

RAIZ = Path(__file__).resolve().parents[1]
CONFIG_POR_DEFEITO = RAIZ / "configs" / "bigru.json"
PASTA_RESULTADOS = RAIZ / "data" / "resultados"
PASTA_MODELOS = RAIZ / "data" / "modelos"

# Avaliar trajetória a trajetória evita o empacotamento e, sem padding, é ~6x mais rápido em CPU
# do que juntar 4 trajetórias num batch com packing (D17).
BATCH_AVALIACAO = 1

CAMPOS_OBRIGATORIOS = (
    "nome",
    "descricao",
    "divisao",
    "experiencias",
    "hidden",
    "camadas",
    "dropout",
    "learning_rate",
    "tamanho_batch",
    "epocas",
    "paciencia",
    "grad_clip",
    "limiar",
    "semente",
)


def ler_config(caminho: Path = CONFIG_POR_DEFEITO) -> dict:
    """Lê e valida a configuração da BiGRU (hiperparâmetros, épocas e sementes)."""
    assert caminho.exists(), f"configuração inexistente: {caminho}"
    config = json.loads(caminho.read_text(encoding="utf-8"))
    faltam = [campo for campo in CAMPOS_OBRIGATORIOS if campo not in config]
    assert not faltam, f"campos em falta em {caminho.name}: {faltam}"
    assert config["divisao"] in DIVISOES, f"divisão desconhecida: {config['divisao']}"
    assert config["divisao"] == "principal", (
        "a divisão por navio é de sensibilidade e só se corre a pedido explícito"
    )
    assert int(config["hidden"]) >= 1 and int(config["camadas"]) >= 1, "hidden/camadas >= 1"
    assert 0.0 <= float(config["dropout"]) < 1.0, "dropout tem de estar em [0, 1)"
    assert float(config["learning_rate"]) > 0, "learning rate tem de ser positivo"
    assert int(config["tamanho_batch"]) >= 1, "tamanho de batch >= 1"
    assert int(config["epocas"]) >= 1, "épocas >= 1"
    assert int(config["paciencia"]) >= 0, "paciência >= 0"
    assert float(config["grad_clip"]) >= 0, "grad clip >= 0 (0 desliga)"
    assert 0.0 < float(config["limiar"]) < 1.0, "limiar em (0, 1)"
    assert int(config["semente"]) >= 0, "semente >= 0"
    for experiencia in config["experiencias"]:
        pasta = RAIZ / "data" / "features" / experiencia
        assert pasta.is_dir(), (
            f"não há features normalizadas para {experiencia}; correr src/features.py"
        )
    return config


def fixar_determinismo(semente: int, threads: int = 1) -> None:
    """Fixa as fontes de aleatoriedade e limita o torch a ``threads`` (1 thread é mais rápido)."""
    assert semente >= 0, "semente >= 0"
    assert threads >= 1, "threads >= 1"
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.set_num_threads(threads)
    # Alguns kernels da GRU em CPU não têm variante determinística; avisar em vez de rebentar.
    torch.use_deterministic_algorithms(True, warn_only=True)


def treinar_epoca(
    modelo: BiGRU,
    treino: DatasetTrajetorias,
    otimizador: torch.optim.Optimizer,
    tamanho_batch: int,
    semente_epoca: int,
    grad_clip: float,
) -> dict:
    """Uma passagem completa pelo treino, em modo treino, com gradient clipping opcional."""
    modelo.train()
    soma_loss = 0.0
    soma_pontos = 0
    normas: list[float] = []
    inicio = time.perf_counter()
    for batch in iterar_batches(treino, tamanho_batch, semente_epoca):
        otimizador.zero_grad(set_to_none=True)
        logits = modelo(batch["x"], batch["comprimentos"])
        loss = loss_mascarada(logits, batch["y"], batch["mascara"])
        loss.backward()
        if grad_clip > 0:
            norma = torch.nn.utils.clip_grad_norm_(modelo.parameters(), grad_clip)
            normas.append(norma.detach().item())
        otimizador.step()
        n_pontos = int(batch["mascara"].sum())
        soma_loss += loss.detach().item() * n_pontos
        soma_pontos += n_pontos
    assert soma_pontos == treino.n_pontos, (
        f"a loss viu {soma_pontos} pontos, o treino tem {treino.n_pontos}"
    )
    return {
        "loss": soma_loss / soma_pontos,
        "tempo_s": time.perf_counter() - inicio,
        "norma_gradiente_max": max(normas) if normas else None,
    }


def f1_macro_validacao(modelo: BiGRU, validacao: DatasetTrajetorias, limiar: float) -> float:
    """F1 macro por ponto na validação, com decisão por limiar (0,5 por omissão)."""
    probs = probabilidades(modelo, validacao, BATCH_AVALIACAO)
    y_real = np.concatenate([trajetoria.y for trajetoria in validacao.trajetorias])
    y_prev = np.concatenate([(prob >= limiar).astype(np.int8) for prob in probs])
    metricas = metricas_ponto(y_real, y_prev)
    f1s = [metricas["pesca"]["f1"], metricas["navegacao"]["f1"]]
    # Uma classe sem pontos reais e sem previsões tem F1 indefinido (None); conta como 0.
    return float(np.mean([0.0 if valor is None else valor for valor in f1s]))


def contar_parametros(modelo: torch.nn.Module) -> int:
    """Número de parâmetros treináveis do modelo."""
    return int(sum(p.numel() for p in modelo.parameters() if p.requires_grad))


def treinar(
    experiencia: str,
    config: dict,
    semente: int,
    threads: int | None = None,
    epocas: int | None = None,
    verboso: bool = True,
) -> dict:
    """Treina uma BiGRU numa experiência, com early stopping pelo F1 macro da validação.

    Devolve o registo do treino (histórico época a época, melhor época, tempos e caminhos dos
    artefactos escritos em ``data/``). O estado final é o da melhor época na validação, não o da
    última: evita escolher um modelo já em sobreajuste.
    """
    threads = int(threads if threads is not None else config.get("threads", 1))
    fixar_determinismo(semente, threads)
    treino = carregar_dataset(experiencia, "treino", config["divisao"])
    validacao = carregar_dataset(experiencia, "validacao", config["divisao"])
    tamanho_batch = int(config["tamanho_batch"])
    limiar = float(config["limiar"])
    paciencia = int(config["paciencia"])
    grad_clip = float(config["grad_clip"])
    n_epocas = int(epocas if epocas is not None else config["epocas"])

    modelo = BiGRU(
        treino.n_features,
        hidden=int(config["hidden"]),
        camadas=int(config["camadas"]),
        dropout=float(config["dropout"]),
    )
    otimizador = torch.optim.Adam(modelo.parameters(), lr=float(config["learning_rate"]))

    historico: list[dict] = []
    melhor_f1 = float("-inf")
    melhor_epoca = 0
    melhor_estado = None
    sem_melhoria = 0
    inicio = time.perf_counter()
    for epoca in range(1, n_epocas + 1):
        registo_epoca = treinar_epoca(
            modelo, treino, otimizador, tamanho_batch, semente + epoca, grad_clip
        )
        f1 = f1_macro_validacao(modelo, validacao, limiar)
        registo_epoca.update({"epoca": epoca, "f1_macro_validacao": f1})
        historico.append(registo_epoca)
        melhorou = f1 > melhor_f1 + 1e-9
        if melhorou:
            melhor_f1, melhor_epoca, sem_melhoria = f1, epoca, 0
            melhor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sem_melhoria += 1
        if verboso:
            marca = " *" if melhorou else ""
            print(
                f"  época {epoca:2d}/{n_epocas}: loss {registo_epoca['loss']:.4f} | "
                f"F1 macro val {f1:.4f} | {registo_epoca['tempo_s']:5.1f}s{marca}",
                flush=True,
            )
        if sem_melhoria >= paciencia:
            if verboso:
                print(f"  paragem antecipada: {paciencia} épocas sem melhorar", flush=True)
            break
    tempo_total = time.perf_counter() - inicio
    assert melhor_estado is not None, "não houve nenhuma época avaliada"
    modelo.load_state_dict(melhor_estado)
    modelo.eval()

    probs = probabilidades(modelo, validacao, BATCH_AVALIACAO)
    previsoes = construir_previsoes(validacao, probs, limiar)
    parquet = escrever_previsoes(
        previsoes, PASTA_PREVISOES / experiencia / f"BiGRU_semente{semente}_validacao.parquet"
    )
    avaliacao = avaliar_previsoes(previsoes)
    json_avaliacao = guardar_avaliacao(
        avaliacao, PASTA_RESULTADOS / experiencia / f"BiGRU_semente{semente}_validacao.json"
    )
    checkpoint = PASTA_MODELOS / experiencia / f"bigru_semente{semente}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "estado": melhor_estado,
            "experiencia": experiencia,
            "semente": semente,
            "melhor_epoca": melhor_epoca,
            "melhor_f1_macro_validacao": melhor_f1,
            "hiperparametros": {
                campo: config[campo]
                for campo in (
                    "hidden",
                    "camadas",
                    "dropout",
                    "learning_rate",
                    "tamanho_batch",
                    "grad_clip",
                    "limiar",
                )
            },
            "n_features": treino.n_features,
            "features": treino.features,
        },
        checkpoint,
    )
    registo = {
        "modelo": "BiGRU",
        "experiencia": experiencia,
        "divisao": config["divisao"],
        "semente": semente,
        "threads": threads,
        "parametros": contar_parametros(modelo),
        "epocas_pedidas": n_epocas,
        "epocas_corridas": len(historico),
        "melhor_epoca": melhor_epoca,
        "melhor_f1_macro_validacao": melhor_f1,
        "tempo_total_s": tempo_total,
        "tempo_medio_epoca_s": float(np.mean([r["tempo_s"] for r in historico])),
        "pontos_treino": treino.n_pontos,
        "pontos_validacao": validacao.n_pontos,
        "trajetorias_treino": len(treino),
        "trajetorias_validacao": len(validacao),
        "historico": historico,
        "ficheiros": {
            "checkpoint": str(checkpoint.relative_to(RAIZ)),
            "previsoes": str(parquet.relative_to(RAIZ)),
            "avaliacao": str(json_avaliacao.relative_to(RAIZ)),
        },
    }
    caminho_registo = PASTA_RESULTADOS / experiencia / f"bigru_semente{semente}_treino.json"
    caminho_registo.parent.mkdir(parents=True, exist_ok=True)
    caminho_registo.write_text(
        json.dumps(registo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    registo["caminho_registo"] = str(caminho_registo.relative_to(RAIZ))
    if verboso:
        imprimir_avaliacao(avaliacao, f"BiGRU {experiencia} semente {semente} (validação)")
        print(
            f"  {len(historico)} épocas em {tempo_total / 60:.1f} min "
            f"({registo['tempo_medio_epoca_s']:.1f}s/época); melhor época {melhor_epoca} "
            f"(F1 macro {melhor_f1:.4f}); registo em {registo['caminho_registo']}",
            flush=True,
        )


def amostra_treino(
    treino: DatasetTrajetorias, n_trajetorias: int
) -> DatasetTrajetorias:
    """Amostra fixa das trajetórias mais curtas **com as duas classes**, para testes rápidos."""
    candidatas = [
        (trajetoria.comprimento, indice)
        for indice, trajetoria in enumerate(treino.trajetorias)
        if 0.05 <= float(trajetoria.y.mean()) <= 0.95
    ]
    assert len(candidatas) >= n_trajetorias, (
        f"só {len(candidatas)} trajetórias com as duas classes; pedidas {n_trajetorias}"
    )
    candidatas.sort()
    escolhidas = [indice for _, indice in candidatas[:n_trajetorias]]
    return DatasetTrajetorias(
        [treino.trajetorias[indice] for indice in escolhidas], treino.features, nome="amostra"
    )


def sanidade(
    config: dict,
    experiencia: str,
    semente: int = 0,
    threads: int = 1,
    n_trajetorias: int = 5,
    epocas: int = 60,
) -> None:
    """Overfit de poucas trajetórias: se a loss não descer, há um erro no pipeline.

    Não mede qualidade (é sobreajuste deliberado); só confirma que o modelo, a loss mascarada e o
    otimizador conseguem memorizar. Sem early stopping e com batch de 1.
    """
    fixar_determinismo(semente, threads)
    completo = carregar_dataset(experiencia, "treino", config["divisao"])
    treino = amostra_treino(completo, n_trajetorias)
    modelo = BiGRU(
        treino.n_features,
        hidden=int(config["hidden"]),
        camadas=int(config["camadas"]),
        dropout=float(config["dropout"]),
    )
    otimizador = torch.optim.Adam(modelo.parameters(), lr=float(config["learning_rate"]))
    losses: list[float] = []
    inicio = time.perf_counter()
    for epoca in range(1, epocas + 1):
        registo = treinar_epoca(
            modelo, treino, otimizador, 1, semente + epoca, float(config["grad_clip"])
        )
        losses.append(registo["loss"])
        if epoca == 1 or epoca % 10 == 0:
            print(f"  sanidade época {epoca:3d}: loss {registo['loss']:.4f}", flush=True)
    f1 = f1_macro_validacao(modelo, treino, float(config["limiar"]))
    tempo = time.perf_counter() - inicio
    print(
        f"Sanidade: {len(treino)} trajetórias ({treino.n_pontos} pontos), loss "
        f"{losses[0]:.4f} -> {losses[-1]:.4f}, F1 macro na própria amostra {f1:.4f} "
        f"em {tempo:.1f}s"
    )
    assert losses[-1] < 0.5 * losses[0], (
        f"a loss não desceu o suficiente: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    assert f1 >= 0.8, f"o modelo não memorizou a amostra: F1 macro {f1:.4f}"


def determinismo(
    config: dict,
    experiencia: str,
    semente: int = 0,
    threads: int = 1,
    n_trajetorias: int = 3,
    epocas: int = 2,
    tolerancia: float = 1e-7,
) -> dict:
    """Dois treinos com a mesma semente têm de dar a mesma loss e as mesmas probabilidades."""
    assinaturas = []
    for _ in range(2):
        fixar_determinismo(semente, threads)
        completo = carregar_dataset(experiencia, "treino", config["divisao"])
        amostra = amostra_treino(completo, n_trajetorias)
        modelo = BiGRU(
            amostra.n_features,
            hidden=int(config["hidden"]),
            camadas=int(config["camadas"]),
            dropout=float(config["dropout"]),
        )
        otimizador = torch.optim.Adam(modelo.parameters(), lr=float(config["learning_rate"]))
        losses = [
            treinar_epoca(
                modelo, amostra, otimizador, 2, semente + epoca, float(config["grad_clip"])
            )["loss"]
            for epoca in range(1, epocas + 1)
        ]
        assinaturas.append((losses, np.concatenate(probabilidades(modelo, amostra, 2))))
    (losses_1, probs_1), (losses_2, probs_2) = assinaturas
    diferenca_loss = float(np.max(np.abs(np.array(losses_1) - np.array(losses_2))))
    diferenca_prob = float(np.max(np.abs(probs_1 - probs_2)))
    print(
        f"Determinismo: losses {['%.6f' % v for v in losses_1]} vs "
        f"{['%.6f' % v for v in losses_2]}; diferença máxima nas probabilidades "
        f"{diferenca_prob:.3e}"
    )
    assert diferenca_loss <= tolerancia, f"loss não reprodutível: diferença {diferenca_loss:.3e}"
    assert diferenca_prob <= tolerancia, f"probabilidades não reprodutíveis: {diferenca_prob:.3e}"
    return {"diferenca_loss": diferenca_loss, "diferenca_prob": diferenca_prob}


def principal() -> None:
    """Ponto de entrada: testes rápidos (``--sanidade``, ``--determinismo``) ou treino a sério."""
    parser = argparse.ArgumentParser(
        description="Passo 6: treino da BiGRU por trajetória completa, com loss mascarada."
    )
    parser.add_argument("--config", default=str(CONFIG_POR_DEFEITO))
    parser.add_argument("--exp", default=None, help="experiência de features (EXP-00x)")
    parser.add_argument("--semente", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None, help="1 thread é o mais rápido (D16)")
    parser.add_argument("--epocas", type=int, default=None, help="sobrepõe o config (testes)")
    parser.add_argument("--sanidade", action="store_true", help="overfit de 5 trajetórias curtas")
    parser.add_argument(
        "--determinismo", action="store_true", help="dois treinos iguais dão o mesmo resultado"
    )
    argumentos = parser.parse_args()

    config = ler_config(Path(argumentos.config))
    threads = int(
        argumentos.threads if argumentos.threads is not None else config.get("threads", 1)
    )
    experiencia = argumentos.exp or config["experiencias"][0]
    assert experiencia in config["experiencias"], (
        f"{experiencia} não está em {config['experiencias']}"
    )
    semente = int(argumentos.semente if argumentos.semente is not None else config["semente"])

    if argumentos.sanidade:
        sanidade(
            config,
            experiencia,
            semente=0,
            threads=threads,
            epocas=int(argumentos.epocas or 60),
        )
        return
    if argumentos.determinismo:
        determinismo(
            config, experiencia, semente=0, threads=threads, epocas=int(argumentos.epocas or 2)
        )
        return
    print(
        f"BiGRU em {experiencia}, semente {semente}, {threads} thread(s): "
        f"hidden {config['hidden']}, {config['camadas']} camada(s), lr {config['learning_rate']}, "
        f"batch {config['tamanho_batch']}, até {argumentos.epocas or config['epocas']} épocas",
        flush=True,
    )
    treinar(experiencia, config, semente, threads=threads, epocas=argumentos.epocas)


if __name__ == "__main__":
    principal()
