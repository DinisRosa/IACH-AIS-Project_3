"""Passo 3 da fase 1: dividir as trajetórias por id completo em treino, validação e teste.

A unidade da divisão é o id completo (``id_navio``, ``n_traj``): todos os pontos de uma
trajetória ficam no mesmo conjunto. A divisão é 70/15/15 com semente 42 e produz 86, 18 e 19
trajetórias. O sorteio é ``numpy.random.default_rng(42).permutation`` sobre a lista de ids
ordenada por ``(id_navio, n_traj)``, com os cortes ``[:86]`` (treino), ``[86:104]`` (teste) e
``[104:]`` (validação). Esta é a única variante testada que reproduz os conjuntos da proposta
entregue (706 522 / 146 225 / 173 532 pontos e 75,6 % / 73,5 % / 77,1 % de pesca); as
variantes com ``random.shuffle``, ``numpy.random.shuffle`` ou ``rng.choice`` davam outros
conjuntos.

O resultado é escrito em ``data/dividir/conjuntos.json`` (a lista de ids de cada conjunto), que
os scripts seguintes leem em vez de repetirem o sorteio.

Executar a partir da raiz do repositório:

    python src/dividir.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_PARQUET = RAIZ / "data" / "ais_123.parquet"
CAMINHO_SAIDA = RAIZ / "data" / "dividir" / "conjuntos.json"

SEMENTE = 42
N_TRAJETORIAS_ESPERADAS = 123
N_NAVIOS_ESPERADOS = 84
N_PONTOS_ESPERADOS = 1_026_279

# Cortes da divisão, pela ordem do código: treino, depois teste, depois validação.
N_TREINO = 86
N_TESTE = 18
N_VALIDACAO = 19
NOMES_CONJUNTOS = ("treino", "validacao", "teste")

# Números da proposta entregue (docs/latex/proposta_pt.tex), usados como verificação de
# regressão: se a divisão deixar de os reproduzir, algo mudou nos dados ou no sorteio.
PROPOSTA = {
    "treino": (86, 706_522, 75.6),
    "validacao": (19, 173_532, 77.1),
    "teste": (18, 146_225, 73.5),
}


def ler_ids(caminho: Path = CAMINHO_PARQUET) -> list[tuple[int, int]]:
    """Lê do Parquet a lista ordenada de ids completos ``(id_navio, n_traj)``."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado em {caminho}. Correr primeiro: python src/preparar_parquet.py"
        )
    ids = pd.read_parquet(caminho, columns=["id_navio", "n_traj"]).drop_duplicates()
    ids = ids.sort_values(["id_navio", "n_traj"]).reset_index(drop=True)
    return [(int(navio), int(traj)) for navio, traj in ids.itertuples(index=False)]


def formatar_id(id_completo: tuple[int, int]) -> str:
    """Devolve o id no formato original ``<navio>-<n>``."""
    return f"{id_completo[0]}-{id_completo[1]}"


def dividir_ids(
    ids: list[tuple[int, int]], semente: int = SEMENTE
) -> dict[str, list[tuple[int, int]]]:
    """Sorteia os ids e reparte-os em treino, teste e validação, com os cortes fixos."""
    assert len(ids) == N_TRAJETORIAS_ESPERADAS, (
        f"trajetórias: {len(ids)} != {N_TRAJETORIAS_ESPERADAS}"
    )
    ordem = [ids[i] for i in np.random.default_rng(semente).permutation(len(ids))]
    return {
        "treino": ordem[:N_TREINO],
        "teste": ordem[N_TREINO : N_TREINO + N_TESTE],
        "validacao": ordem[N_TREINO + N_TESTE :],
    }


def validar_divisao(
    conjuntos: dict[str, list[tuple[int, int]]], df: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Confirma que a divisão é uma partição dos ids e devolve o resumo por conjunto."""
    todos: set[tuple[int, int]] = set()
    for nome, conjunto in conjuntos.items():
        assert len(conjunto) == len(set(conjunto)), f"ids repetidos em {nome}"
        repetidos = todos & set(conjunto)
        assert not repetidos, f"ids em mais de um conjunto: {sorted(repetidos)}"
        todos |= set(conjunto)
    assert len(todos) == N_TRAJETORIAS_ESPERADAS, (
        f"a divisão cobre {len(todos)} trajetórias, esperadas {N_TRAJETORIAS_ESPERADAS}"
    )
    assert len(conjuntos["treino"]) == N_TREINO, f"treino: {len(conjuntos['treino'])} != {N_TREINO}"
    assert len(conjuntos["teste"]) == N_TESTE, f"teste: {len(conjuntos['teste'])} != {N_TESTE}"
    assert len(conjuntos["validacao"]) == N_VALIDACAO, (
        f"validação: {len(conjuntos['validacao'])} != {N_VALIDACAO}"
    )

    # Resumo por conjunto, calculado sobre o Parquet e não sobre os ids.
    resumo: dict[str, dict[str, float]] = {}
    total_pontos = 0
    for nome in NOMES_CONJUNTOS:
        chaves = pd.MultiIndex.from_frame(df[["id_navio", "n_traj"]])
        # `isin` sobre o MultiIndex marca os pontos do conjunto sem depender da ordem das linhas.
        escolhidos = chaves.isin(pd.MultiIndex.from_tuples(conjuntos[nome]))
        sub = df[escolhidos]
        n_inteiras = sub.groupby(["id_navio", "n_traj"], observed=True).ngroups
        assert n_inteiras == len(conjuntos[nome]), (
            f"o conjunto {nome} devia ter {len(conjuntos[nome])} trajetórias inteiras, "
            f"tem {n_inteiras}"
        )
        pontos = len(sub)
        pesca = int(sub["label"].sum())
        total_pontos += pontos
        resumo[nome] = {
            "trajetorias": len(conjuntos[nome]),
            "navios": int(sub["id_navio"].nunique()),
            "pontos": pontos,
            "pesca": pesca,
            "percentagem_pesca": round(100 * pesca / pontos, 1),
        }
    assert total_pontos == N_PONTOS_ESPERADOS, (
        f"a divisão soma {total_pontos} pontos, esperados {N_PONTOS_ESPERADOS}"
    )
    return resumo


def comparar_com_proposta(resumo: dict[str, dict[str, float]]) -> None:
    """Compara o resumo com os números da proposta e imprime a diferença."""
    for nome in NOMES_CONJUNTOS:
        traj, pontos, pesca = PROPOSTA[nome]
        obtido = resumo[nome]
        print(
            f"  {nome:10s} proposta: {traj} traj / {pontos} pontos / {pesca:.1f} % pesca | "
            f"obtido: {obtido['trajetorias']} / {obtido['pontos']} / "
            f"{obtido['percentagem_pesca']:.1f} %"
        )
        assert (
            obtido["trajetorias"] == traj
            and obtido["pontos"] == pontos
            and obtido["percentagem_pesca"] == pesca
        ), (
            f"a divisão deixou de reproduzir a da proposta em {nome}; se os dados ou o "
            "sorteio mudaram de propósito, atualizar PROPOSTA e o texto da proposta"
        )


def escrever_conjuntos(
    conjuntos: dict[str, list[tuple[int, int]]], caminho: Path = CAMINHO_SAIDA
) -> None:
    """Guarda os ids de cada conjunto em JSON, para os scripts seguintes os reutilizarem."""
    conteudo = {
        "semente": SEMENTE,
        "proporcoes": {"treino": 0.7, "validacao": 0.15, "teste": 0.15},
        "n_trajetorias": {nome: len(conjuntos[nome]) for nome in NOMES_CONJUNTOS},
        "ids": {
            nome: [formatar_id(id_completo) for id_completo in conjuntos[nome]]
            for nome in NOMES_CONJUNTOS
        },
    }
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, indent=2) + "\n", encoding="utf-8")


def carregar_conjuntos(caminho: Path = CAMINHO_SAIDA) -> dict[str, set[tuple[int, int]]]:
    """Lê o JSON da divisão e devolve os ids de cada conjunto como conjunto de tuplos."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Divisão não encontrada em {caminho}. Correr primeiro: python src/dividir.py"
        )
    conteudo = json.loads(caminho.read_text(encoding="utf-8"))
    conjuntos: dict[str, set[tuple[int, int]]] = {}
    for nome in NOMES_CONJUNTOS:
        ids: set[tuple[int, int]] = set()
        for texto in conteudo["ids"][nome]:
            navio, traj = texto.split("-", maxsplit=1)
            ids.add((int(navio), int(traj)))
        conjuntos[nome] = ids
    return conjuntos


def dividir(
    caminho_parquet: Path = CAMINHO_PARQUET, caminho_saida: Path = CAMINHO_SAIDA
) -> dict[str, set[tuple[int, int]]]:
    """Executa o passo 3 completo: sorteio, validação, escrita do JSON e resumo."""
    df = pd.read_parquet(caminho_parquet, columns=["id_navio", "n_traj", "label"])
    ids = ler_ids(caminho_parquet)
    conjuntos = dividir_ids(ids, SEMENTE)
    resumo = validar_divisao(conjuntos, df)
    escrever_conjuntos(conjuntos, caminho_saida)

    print(f"Divisão escrita em {caminho_saida.relative_to(RAIZ)} (semente {SEMENTE})")
    for nome in NOMES_CONJUNTOS:
        r = resumo[nome]
        print(
            f"  {nome:10s} {r['trajetorias']:3d} trajetórias | {r['navios']:2d} navios | "
            f"{r['pontos']:7d} pontos | {r['percentagem_pesca']:.1f} % de pesca"
        )
    comparar_com_proposta(resumo)
    return carregar_conjuntos(caminho_saida)


if __name__ == "__main__":
    dividir()
