"""Modelos sequenciais da fase 1: a BiGRU que se treina e a de brinquedo que mede o custo em CPU.

``BiGRU`` é o modelo principal do passo 6: projeção linear, GRU bidirecional **com packing** e um
logit por ponto. O packing garante que o padding de um batch não entra em nenhuma das direções,
logo a previsão de uma trajetória é a mesma quer vá sozinha quer acompanhada. Quando não há
padding (``B = 1`` ou todos os comprimentos iguais) o ``forward`` salta o empacotamento, porque em
CPU ele custa ~6× mais do que correr a GRU diretamente (D17).

``BigruBrinquedo`` e o ``main`` continuam a ser a medição do passo 5: uma arquitetura equivalente
sem packing, para medir o custo do pior caso (com padding) na trajetória mais longa do treino
(45 251 pontos) e decidir se a BiGRU era viável em CPU. Números em
``data/benchmarks/bigru_cpu.json`` (1 thread) e no plano.

Executar a partir da raiz do repositório:

    python src/modelos.py --threads 1 --saida bigru_cpu.json   # medição de referência (1 thread)
    python src/modelos.py --repeticoes 1 --hidden 32           # medição rápida
    python src/modelos.py --testes --threads 1                 # packing e loss mascarada
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset import CONFIG_POR_DEFEITO, carregar_dataset, collate_trajetorias, ler_config

RAIZ = Path(__file__).resolve().parents[1]
PASTA_BENCHMARKS = RAIZ / "data" / "benchmarks"


def loss_mascarada(logits: torch.Tensor, y: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
    """BCE média só sobre os pontos reais; o padding (``y = -1``) nunca entra na conta."""
    alvo = y[mascara]
    assert alvo.numel() > 0, "batch sem pontos reais"
    assert bool((alvo >= 0).all()), "há padding a entrar na loss"
    return nn.functional.binary_cross_entropy_with_logits(
        logits[mascara], alvo.to(logits.dtype)
    )


class BiGRU(nn.Module):
    """Projeção linear + GRU bidirecional **com packing** + saída por ponto.

    Usa ``pack_padded_sequence``/``pad_packed_sequence``: o padding dos batches não entra em
    nenhuma das duas direções, por isso a previsão de uma trajetória não depende das outras
    trajetórias do batch. É o modelo do ``treinar.py``; a ``BigruBrinquedo`` abaixo fica como
    referência de custo do pior caso (com padding), medida no passo 5.
    """

    def __init__(
        self, n_features: int, hidden: int = 64, camadas: int = 1, dropout: float = 0.0
    ):
        super().__init__()
        assert hidden >= 1 and camadas >= 1, "hidden e camadas têm de ser >= 1"
        assert 0.0 <= dropout < 1.0, "dropout tem de estar em [0, 1)"
        self.projecao = nn.Linear(n_features, hidden)
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=camadas,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if camadas > 1 else 0.0,
        )
        self.saida = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor, comprimentos: torch.Tensor | None = None) -> torch.Tensor:
        """Recebe ``[B, L, F]`` (e os comprimentos reais) e devolve um logit por ponto, ``[B, L]``.

        Quando o batch não tem padding (``B = 1`` ou todas as sequências com o comprimento máximo),
        salta o empacotamento: em CPU o ``pack_padded_sequence`` custa cerca de 6× mais do que correr
        a GRU diretamente, e sem padding os dois caminhos dão exatamente o mesmo resultado.
        """
        ativacoes = torch.relu(self.projecao(x))
        if comprimentos is None:
            saida, _ = self.gru(ativacoes)
        else:
            assert comprimentos.shape == (x.shape[0],), "comprimentos com forma errada"
            assert int(comprimentos.max()) <= x.shape[1], "comprimento maior do que o batch"
            assert bool((comprimentos > 0).all()), "trajetória com comprimento zero"
            if bool((comprimentos == x.shape[1]).all()):
                saida, _ = self.gru(ativacoes)
            else:
                empacotado = nn.utils.rnn.pack_padded_sequence(
                    ativacoes,
                    comprimentos.detach().to("cpu"),
                    batch_first=True,
                    enforce_sorted=False,
                )
                saida, _ = self.gru(empacotado)
                saida, _ = nn.utils.rnn.pad_packed_sequence(
                    saida, batch_first=True, total_length=x.shape[1]
                )
        return self.saida(saida).squeeze(-1)


@torch.no_grad()
def prever_logits(
    modelo: nn.Module, x: torch.Tensor, comprimentos: torch.Tensor | None = None
) -> np.ndarray:
    """Logits por ponto em modo de avaliação, sem tocar no estado de treino do modelo."""
    estava_a_treinar = modelo.training
    modelo.eval()
    try:
        logits = modelo(x, comprimentos)
    finally:
        modelo.train(estava_a_treinar)
    return logits.detach().cpu().numpy()


def probabilidades(
    modelo: nn.Module, dataset, tamanho_batch: int = 4
) -> list[np.ndarray]:
    """Probabilidade de pesca por trajetória, com batches agrupados por comprimento."""
    from dataset import iterar_batches

    probabilidades_por_trajetoria: dict[tuple[int, int], np.ndarray] = {}
    for batch in iterar_batches(dataset, tamanho_batch, semente=0):
        logits = prever_logits(modelo, batch["x"], batch["comprimentos"])
        for linha, identificador in enumerate(batch["ids"]):
            comprimento = int(batch["comprimentos"][linha])
            prob = 1.0 / (1.0 + np.exp(-logits[linha, :comprimento].astype(np.float64)))
            probabilidades_por_trajetoria[identificador] = prob
    saida = [probabilidades_por_trajetoria[i] for i in dataset.ids]
    assert len(saida) == len(dataset), "trajetórias perdidas na previsão"
    return saida


def verificar_padding(
    modelo: nn.Module, dataset, tolerancia: float = 1e-5
) -> dict:
    """A previsão de uma trajetória não pode mudar por estar num batch com padding.

    Compara a trajetória mais curta e a mais longa do conjunto: sozinhas e no mesmo batch (com
    padding na curta). O teste cobre as duas direções da GRU, porque a diferença na direção inversa
    é precisamente a que o padding introduz.
    """
    assert len(dataset) >= 2, "preciso de pelo menos duas trajetórias"
    ordem = np.argsort(dataset.comprimentos, kind="stable")
    indice_curta = int(ordem[0])
    indice_longa = int(ordem[-1])
    assert dataset.comprimentos[indice_curta] < dataset.comprimentos[indice_longa], (
        "as duas trajetórias têm o mesmo comprimento; o teste não mede padding"
    )
    sozinha_curta = prever_logits(
        modelo, dataset[indice_curta]["x"][None], torch.tensor([dataset[indice_curta]["comprimento"]])
    )[0]
    sozinha_longa = prever_logits(
        modelo, dataset[indice_longa]["x"][None], torch.tensor([dataset[indice_longa]["comprimento"]])
    )[0]
    batch = collate_trajetorias([dataset[indice_curta], dataset[indice_longa]])
    assert int(batch["x"].shape[1]) == dataset.comprimentos[indice_longa], "sem padding no batch"
    juntas = prever_logits(modelo, batch["x"], batch["comprimentos"])
    comprimento_curta = dataset.comprimentos[indice_curta]
    comprimento_longa = dataset.comprimentos[indice_longa]
    diferenca_curta = float(
        np.max(np.abs(juntas[0, :comprimento_curta] - sozinha_curta))
    )
    diferenca_longa = float(
        np.max(np.abs(juntas[1, :comprimento_longa] - sozinha_longa))
    )
    assert diferenca_curta <= tolerancia, (
        f"trajetória curta muda dentro do batch: diferença máxima {diferenca_curta:.3e}"
    )
    assert diferenca_longa <= tolerancia, (
        f"trajetória longa muda dentro do batch: diferença máxima {diferenca_longa:.3e}"
    )
    return {
        "diferenca_max_curta": diferenca_curta,
        "diferenca_max_longa": diferenca_longa,
        "comprimento_curta": int(comprimento_curta),
        "comprimento_longa": int(comprimento_longa),
        "tolerancia": tolerancia,
    }


class BigruBrinquedo(nn.Module):
    """Projeção linear + GRU bidirecional + saída por ponto (um logit por ponto)."""

    def __init__(self, n_features: int, hidden: int = 64, camadas: int = 1):
        super().__init__()
        assert hidden >= 1 and camadas >= 1, "hidden e camadas têm de ser >= 1"
        self.projecao = nn.Linear(n_features, hidden)
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=camadas,
            batch_first=True,
            bidirectional=True,
        )
        self.saida = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recebe ``[B, L, F]`` e devolve um logit por ponto, ``[B, L]``."""
        ativacoes = torch.relu(self.projecao(x))
        saida, _ = self.gru(ativacoes)
        return self.saida(saida).squeeze(-1)


def memoria_mb() -> dict[str, float]:
    """Memória residente atual e pico do processo, em MB (lido de /proc/self/status)."""
    valores: dict[str, float] = {}
    with open("/proc/self/status", encoding="utf-8") as ficheiro:
        for linha in ficheiro:
            if linha.startswith(("VmRSS:", "VmHWM:")):
                nome, valor, _ = linha.split()
                valores[nome.rstrip(":")] = int(valor) / 1024.0
    return {"rss_mb": valores["VmRSS"], "pico_mb": valores["VmHWM"]}


def medir_passo(
    modelo: nn.Module, batch: dict, repeticoes: int = 3
) -> dict[str, list[float]]:
    """Cronometra ``repeticoes`` vezes um forward+backward (com o passo do SGD) sobre o batch."""
    assert repeticoes >= 1, "repetições tem de ser >= 1"
    otimizador = torch.optim.SGD(modelo.parameters(), lr=1e-3)
    tempos: list[float] = []
    perdas: list[float] = []
    for _ in range(repeticoes):
        otimizador.zero_grad(set_to_none=True)
        inicio = time.perf_counter()
        logits = modelo(batch["x"])
        loss = loss_mascarada(logits, batch["y"], batch["mascara"])
        loss.backward()
        otimizador.step()
        tempos.append(time.perf_counter() - inicio)
        perdas.append(float(loss.detach()))
        del logits, loss
    return {"tempos_s": tempos, "perdas": perdas}


def cenarios(dataset_treino) -> dict[str, dict]:
    """Três batches para medir: um mediano, a trajetória mais longa e as duas mais longas.

    A ordem é crescente em número de pontos, para que o pico de memória de cada cenário se possa
    atribuir ao próprio cenário (VmHWM nunca desce).
    """
    comprimentos = dataset_treino.comprimentos
    ordem = np.argsort(comprimentos, kind="stable")
    meio = len(ordem) // 2
    mediano = collate_trajetorias([dataset_treino[ordem[meio - 1]], dataset_treino[ordem[meio]]])
    mais_longa = collate_trajetorias([dataset_treino[ordem[-1]]])
    duas_longas = collate_trajetorias([dataset_treino[ordem[-2]], dataset_treino[ordem[-1]]])
    return {
        "batch mediano (2 trajetórias)": mediano,
        "trajetória mais longa": mais_longa,
        "duas mais longas (com padding)": duas_longas,
    }


def pontos_do_batch(batch: dict) -> int:
    """Número de pontos reais (sem contar com o padding) de um batch."""
    return int(batch["comprimentos"].sum())


def principal() -> None:
    """Mede o custo de uma BiGRU de brinquedo nos três cenários e escreve o JSON."""
    parser = argparse.ArgumentParser(
        description="Passo 5: custo de uma BiGRU de brinquedo em CPU (forward+backward)."
    )
    parser.add_argument("--config", default=str(CONFIG_POR_DEFEITO))
    parser.add_argument("--exp", default=None, help="experiência de features do dataset")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--camadas", type=int, default=1)
    parser.add_argument("--repeticoes", type=int, default=3)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="threads de CPU para o torch (por omissão, as que o torch escolher)",
    )
    parser.add_argument(
        "--saida",
        default="bigru_cpu.json",
        help="nome do ficheiro em data/benchmarks/ onde ficam os números",
    )
    parser.add_argument(
        "--testes",
        action="store_true",
        help="só verificar que o padding não muda as previsões (packing) e sair",
    )
    argumentos = parser.parse_args()

    if argumentos.threads is not None:
        assert argumentos.threads >= 1, "threads tem de ser >= 1"
        torch.set_num_threads(argumentos.threads)

    if argumentos.testes:
        config = ler_config(Path(argumentos.config))
        experiencia = argumentos.exp or config["experiencia"]
        treino = carregar_dataset(experiencia, "treino", "principal")
        torch.manual_seed(int(config["semente"]))
        modelo = BiGRU(treino.n_features, argumentos.hidden, argumentos.camadas)
        resultado = verificar_padding(modelo, treino)
        print(
            f"Packing ok (1 thread: {torch.get_num_threads()}): trajetória de "
            f"{resultado['comprimento_curta']} pontos dentro de um batch de "
            f"{resultado['comprimento_longa']} pontos difere no máximo "
            f"{resultado['diferenca_max_curta']:.3e} (limite {resultado['tolerancia']:.0e}); "
            f"a longa difere {resultado['diferenca_max_longa']:.3e}"
        )
        return

    config = ler_config(Path(argumentos.config))
    experiencia = argumentos.exp or config["experiencia"]
    treino = carregar_dataset(experiencia, "treino", "principal")

    torch.manual_seed(int(config["semente"]))
    modelo = BigruBrinquedo(treino.n_features, argumentos.hidden, argumentos.camadas)
    n_parametros = sum(p.numel() for p in modelo.parameters())
    print(
        f"BiGRU de brinquedo: {treino.n_features} features, hidden {argumentos.hidden}, "
        f"{argumentos.camadas} camada(s), {n_parametros} parâmetros"
    )
    print(
        f"torch {torch.__version__} | {torch.get_num_threads()} threads | "
        f"{argumentos.repeticoes} repetições por cenário"
    )
    print("Treino: 706 522 pontos em 86 trajetórias (divisão principal)")

    resultados: dict[str, dict] = {}
    for nome, batch in cenarios(treino).items():
        antes = memoria_mb()
        medicao = medir_passo(modelo, batch, argumentos.repeticoes)
        depois = memoria_mb()
        pontos = pontos_do_batch(batch)
        tempos = medicao["tempos_s"]
        resultados[nome] = {
            "trajetorias": len(batch["ids"]),
            "pontos": pontos,
            "comprimento_maximo": int(batch["comprimentos"].max()),
            "padding_pontos": int(batch["x"].shape[0] * batch["x"].shape[1]) - pontos,
            "tempo_min_s": min(tempos),
            "tempo_mediana_s": float(np.median(tempos)),
            "tempo_max_s": max(tempos),
            "tempo_por_1000_pontos_s": float(np.median(tempos)) / pontos * 1000,
            "loss": medicao["perdas"][-1],
            "memoria_extra_mb": max(0.0, depois["pico_mb"] - antes["pico_mb"]),
            "pico_total_mb": depois["pico_mb"],
        }
        r = resultados[nome]
        print(
            f"  {nome:32s} {pontos:6d} pontos | "
            f"tempo mediana {r['tempo_mediana_s']:6.2f} s "
            f"(min {r['tempo_min_s']:5.2f}, max {r['tempo_max_s']:5.2f}) | "
            f"memória +{r['memoria_extra_mb']:7.1f} MB "
            f"(pico {r['pico_total_mb']:7.1f} MB)"
        )

    registo = {
        "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiencia": experiencia,
        "divisao": "principal",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "dispositivo": "cpu",
        "modelo": {
            "tipo": "BigruBrinquedo",
            "hidden": argumentos.hidden,
            "camadas": argumentos.camadas,
            "bidirecional": True,
            "parametros": n_parametros,
        },
        "repeticoes": argumentos.repeticoes,
        "medida": "forward+backward+passo de SGD, com a loss mascarada",
        "cenarios": resultados,
    }
    PASTA_BENCHMARKS.mkdir(parents=True, exist_ok=True)
    destino = PASTA_BENCHMARKS / argumentos.saida
    destino.write_text(json.dumps(registo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Números guardados em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    principal()
