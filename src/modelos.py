"""Passo 5 da fase 1: modelo de brinquedo (BiGRU) e medição do custo de um passo em CPU.

Antes de escolher a arquitetura (BiGRU ou CNN dilatada), mede-se o que custa um forward+backward
de uma BiGRU pequena sobre a trajetória mais longa do treino (45 251 pontos), em CPU, com e sem
padding. O resultado decide se a BiGRU é viável ou se a CNN dilatada passa a principal; os números
ficam em ``data/benchmarks/bigru_cpu.json`` e no plano.

O modelo é de brinquedo: uma projeção linear, uma GRU bidirecional e uma camada de saída por ponto.
Serve para medir custo e para fixar a forma da loss mascarada, não para ser treinado a sério (isso é
o passo 6, ``treinar.py``).

Executar a partir da raiz do repositório:

    python src/modelos.py --threads 1 --saida bigru_cpu.json   # medição de referência (1 thread)
    python src/modelos.py --repeticoes 1 --hidden 32           # medição rápida
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
    argumentos = parser.parse_args()

    if argumentos.threads is not None:
        assert argumentos.threads >= 1, "threads tem de ser >= 1"
        torch.set_num_threads(argumentos.threads)

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
