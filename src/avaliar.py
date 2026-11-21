"""Passo 8 da fase 1: métricas por ponto e por segmento, e parquet de previsões.

O que faz:

- **por ponto:** precisão, sensibilidade, F1 e IoU de cada classe (pesca = 1, navegação = 0),
  F1 macro e PR-AUC (média da precisão, ``average_precision_score``);
- **por segmento:** fragmentação (nº de segmentos previstos vs reais), erro de fronteira em pontos
  (emparelhamento ordenado das fronteiras reais e previstas) e IoU por segmento;
- **por trajetória:** as mesmas métricas em cada trajetória, com média ± desvio entre trajetórias,
  além do agregado sobre todos os pontos;
- escreve o **parquet de previsões** por ponto (``id_navio, n_traj, indice, y_real, y_prev, prob``),
  usando os índices que o ``collate`` do ``dataset.py`` transporta, para a fase 2 comparar
  exatamente sobre os mesmos pontos.

Todas as métricas são calculadas à mão (exceto a PR-AUC), para não depender de convenções opacas e
para os testes compararem com contas feitas à mão. Quando um denominador é zero a métrica vale 0,0
e as contagens ``n_real``/``n_previsto`` ficam no resultado, para o caso degenerado ser visível.

Executar a partir da raiz do repositório:

    python src/avaliar.py --previsoes data/previsoes/EXP-001_features_com_xy/B0_validacao.parquet
    python src/avaliar.py --previsoes ... --modelo B0_sempre_pesca --tolerancia 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

RAIZ = Path(__file__).resolve().parents[1]
PASTA_PREVISOES = RAIZ / "data" / "previsoes"
PASTA_RESULTADOS = RAIZ / "data" / "resultados"

# Colunas do parquet de previsões, pela ordem em que são escritas.
COLUNAS_PREVISOES = ("id_navio", "n_traj", "indice", "y_real", "y_prev", "prob")

# Limiar de decisão por omissão: prob >= 0.5 é pesca.
LIMIAR_POR_DEFEITO = 0.5

# Tolerância (em pontos) do erro de fronteira: uma fronteira prevista a <= 3 pontos conta como
# certa na fração "dentro da tolerância".
TOLERANCIA_FRONTEIRA = 3

CLASSES = {"pesca": 1, "navegacao": 0}


def segmentos(y: np.ndarray) -> list[tuple[int, int, int]]:
    """Corridas consecutivas de rótulos iguais: ``(início, fim exclusivo, classe)``."""
    y = np.asarray(y)
    assert y.ndim == 1 and y.size > 0, "segmentos precisa de um vetor de rótulos não vazio"
    assert set(np.unique(y)).issubset({0, 1}), "rótulos fora de {0, 1}"
    troca = np.r_[True, y[1:] != y[:-1]]
    inicios = np.flatnonzero(troca)
    fins = np.r_[inicios[1:], len(y)]
    return [(int(a), int(b), int(y[a])) for a, b in zip(inicios, fins)]


def fronteiras(y: np.ndarray) -> np.ndarray:
    """Posições em que a classe muda, i.e. os índices ``i`` com ``y[i] != y[i-1]``."""
    y = np.asarray(y)
    assert y.ndim == 1 and y.size > 0, "fronteiras precisa de um vetor de rótulos não vazio"
    return np.flatnonzero(y[1:] != y[:-1]) + 1


def emparelhar_fronteiras(real: np.ndarray, prevista: np.ndarray) -> list[tuple[int, int, int]]:
    """Emparelha fronteiras reais e previstas, uma a uma e por ordem.

    Em cada passo emparelha-se o par atual ou avança-se na lista que aproxima a fronteira da outra;
    como as duas listas estão ordenadas, cada fronteira é usada no máximo uma vez e o resultado é o
    emparelhamento ordenado de menor deslocamento total (guloso). Devolve ``(real, prevista,
    distância_em_pontos)``; as fronteiras que ficam sem par não aparecem.
    """
    real = np.asarray(real, dtype=np.int64)
    prevista = np.asarray(prevista, dtype=np.int64)
    pares: list[tuple[int, int, int]] = []
    i = j = 0
    while i < len(real) and j < len(prevista):
        distancia = abs(int(real[i]) - int(prevista[j]))
        if i + 1 < len(real) and abs(int(real[i + 1]) - int(prevista[j])) <= distancia:
            i += 1
            continue
        if j + 1 < len(prevista) and abs(int(real[i]) - int(prevista[j + 1])) < distancia:
            j += 1
            continue
        pares.append((int(real[i]), int(prevista[j]), distancia))
        i += 1
        j += 1
    return pares


def iou_por_segmento(y_real: np.ndarray, y_prev: np.ndarray) -> dict[int, list[float]]:
    """IoU de cada segmento real com o melhor segmento previsto **da mesma classe**.

    Um segmento mal previsto vale 0 (não há segmento da mesma classe a sobrepor-se), e o facto de o
    previsto estar fragmentado aparece no IoU de cada fragmento, não escondido numa união. Segmentos
    previstos de outra classe não entram: a troca de classe é medida na precisão/sensibilidade.
    """
    segs_real = segmentos(y_real)
    segs_prev = segmentos(y_prev)
    por_classe: dict[int, list[float]] = {1: [], 0: []}
    primeiro = 0
    for inicio, fim, classe in segs_real:
        while primeiro < len(segs_prev) and segs_prev[primeiro][1] <= inicio:
            primeiro += 1
        melhor = 0.0
        j = primeiro
        while j < len(segs_prev) and segs_prev[j][0] < fim:
            inicio_prev, fim_prev, classe_prev = segs_prev[j]
            if classe_prev == classe:
                inter = min(fim, fim_prev) - max(inicio, inicio_prev)
                if inter > 0:
                    uniao = (fim - inicio) + (fim_prev - inicio_prev) - inter
                    melhor = max(melhor, inter / uniao)
            j += 1
        por_classe[classe].append(melhor)
    return por_classe


def _divisao(numerador: float, denominador: float) -> float:
    """Rácio com denominador zero a valer 0,0 (o caso fica visível nas contagens)."""
    return float(numerador) / float(denominador) if denominador else 0.0


def contar_confusoes(y_real: np.ndarray, y_prev: np.ndarray) -> dict[str, int]:
    """Matriz de confusão binária (TP, FP, FN, TN) da classe pesca."""
    y_real = np.asarray(y_real, dtype=np.int64)
    y_prev = np.asarray(y_prev, dtype=np.int64)
    assert y_real.shape == y_prev.shape, "y_real e y_prev com formas diferentes"
    assert y_real.size > 0, "sem pontos para avaliar"
    assert set(np.unique(y_real)).issubset({0, 1}), "y_real fora de {0, 1}"
    assert set(np.unique(y_prev)).issubset({0, 1}), "y_prev fora de {0, 1}"
    positivos = y_prev == 1
    reais = y_real == 1
    confusoes = {
        "tp": int(np.sum(reais & positivos)),
        "fp": int(np.sum(~reais & positivos)),
        "fn": int(np.sum(reais & ~positivos)),
        "tn": int(np.sum(~reais & ~positivos)),
    }
    assert sum(confusoes.values()) == y_real.size, "pontos perdidos na matriz de confusão"
    return confusoes


def metricas_classe(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Precisão, sensibilidade, F1 e IoU de uma classe, a partir da matriz de confusão."""
    return {
        "precisao": _divisao(tp, tp + fp),
        "sensibilidade": _divisao(tp, tp + fn),
        "f1": _divisao(2 * tp, 2 * tp + fp + fn),
        "iou": _divisao(tp, tp + fp + fn),
    }


def metricas_ponto(
    y_real: np.ndarray, y_prev: np.ndarray, prob: np.ndarray | None = None
) -> dict:
    """Métricas por ponto: as duas classes, F1 macro e (se houver ``prob``) PR-AUC."""
    confusoes = contar_confusoes(y_real, y_prev)
    tp, fp, fn, tn = confusoes["tp"], confusoes["fp"], confusoes["fn"], confusoes["tn"]
    pesca = metricas_classe(tp, fp, fn, tn)
    navegacao = metricas_classe(tn, fn, fp, tp)
    pesca.update({"n_real": tp + fn, "n_previsto": tp + fp})
    navegacao.update({"n_real": tn + fp, "n_previsto": tn + fn})
    resultado = {
        "n_pontos": int(np.asarray(y_real).size),
        "pesca": pesca,
        "navegacao": navegacao,
        "f1_macro": 0.5 * (pesca["f1"] + navegacao["f1"]),
        "pr_auc": None,
        "fracao_pesca_real": _divisao(tp + fn, np.asarray(y_real).size),
        "fracao_pesca_prevista": _divisao(tp + fp, np.asarray(y_real).size),
    }
    if prob is not None:
        prob = np.asarray(prob, dtype=np.float64)
        assert prob.shape == np.asarray(y_real).shape, "prob com forma diferente de y_real"
        assert np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all(), (
            "probabilidades fora de [0, 1]"
        )
        if len(np.unique(y_real)) == 2:
            resultado["pr_auc"] = float(average_precision_score(y_real, prob))
    return resultado


def metricas_segmento(
    y_real: np.ndarray, y_prev: np.ndarray, tolerancia: int = TOLERANCIA_FRONTEIRA
) -> dict:
    """Fragmentação, erro de fronteira (em pontos) e IoU por segmento de uma trajetória."""
    assert tolerancia >= 0, "tolerância tem de ser >= 0"
    segs_real = segmentos(y_real)
    segs_prev = segmentos(y_prev)
    fr_real = fronteiras(y_real)
    fr_prev = fronteiras(y_prev)
    pares = emparelhar_fronteiras(fr_real, fr_prev)
    distancias = np.array([distancia for _, _, distancia in pares], dtype=np.float64)
    ious = iou_por_segmento(y_real, y_prev)
    todos = ious[1] + ious[0]
    sem_fronteiras = len(fr_real) == 0 and len(fr_prev) == 0
    return {
        "n_segmentos_real": len(segs_real),
        "n_segmentos_previsto": len(segs_prev),
        "segmentos_extra": len(segs_prev) - len(segs_real),
        "razao_fragmentacao": _divisao(len(segs_prev), len(segs_real)),
        "n_fronteiras_real": int(len(fr_real)),
        "n_fronteiras_prevista": int(len(fr_prev)),
        "n_fronteiras_emparelhadas": len(pares),
        "fracao_fronteiras_emparelhadas": (
            1.0 if sem_fronteiras else _divisao(len(pares), len(fr_real))
        ),
        "fronteira_media_pontos": float(distancias.mean()) if len(distancias) else None,
        "fronteira_mediana_pontos": float(np.median(distancias)) if len(distancias) else None,
        "fronteira_p90_pontos": float(np.percentile(distancias, 90)) if len(distancias) else None,
        "fracao_fronteiras_na_tolerancia": (
            1.0
            if sem_fronteiras
            else _divisao(int(np.sum(distancias <= tolerancia)), len(fr_real))
        ),
        "iou_segmento_medio": float(np.mean(todos)) if todos else None,
        "iou_segmento_pesca_medio": float(np.mean(ious[1])) if ious[1] else None,
        "iou_segmento_navegacao_medio": float(np.mean(ious[0])) if ious[0] else None,
        "n_segmentos_pesca": len(ious[1]),
        "n_segmentos_navegacao": len(ious[0]),
    }


def _achatar(valores: dict, prefixo: str = "") -> dict[str, float]:
    """Converte um dicionário de métricas (com sub-dicionários) em chaves planas numéricas."""
    saida: dict[str, float] = {}
    for chave, valor in valores.items():
        if isinstance(valor, dict):
            saida.update(_achatar(valor, f"{prefixo}{chave}."))
        elif isinstance(valor, (int, float)) and not isinstance(valor, bool):
            saida[f"{prefixo}{chave}"] = float(valor)
    return saida


def _media_desvio(por_trajetoria: list[dict], chave: str) -> tuple[float | None, float | None]:
    """Média e desvio padrão amostral de uma métrica entre trajetórias (ignora ``None``)."""
    valores = np.array(
        [float(t[chave]) for t in por_trajetoria if t.get(chave) is not None], dtype=np.float64
    )
    if valores.size == 0:
        return None, None
    if valores.size == 1:
        return float(valores[0]), 0.0
    return float(valores.mean()), float(valores.std(ddof=1))


def agregar_segmentos(por_trajetoria: list[dict]) -> dict:
    """Soma contagens e pondera as médias entre trajetórias (cada trajetória pesa o que tem)."""
    assert por_trajetoria, "sem trajetórias para agregar segmentos"
    total_segs_real = sum(r["n_segmentos_real"] for r in por_trajetoria)
    total_segs_prev = sum(r["n_segmentos_previsto"] for r in por_trajetoria)
    total_fr_real = sum(r["n_fronteiras_real"] for r in por_trajetoria)
    total_fr_prev = sum(r["n_fronteiras_prevista"] for r in por_trajetoria)
    pares = sum(r["n_fronteiras_emparelhadas"] for r in por_trajetoria)
    soma_distancias = sum(
        r["fronteira_media_pontos"] * r["n_fronteiras_emparelhadas"]
        for r in por_trajetoria
        if r["fronteira_media_pontos"] is not None
    )
    fronteiras_na_tolerancia = sum(
        r["fracao_fronteiras_na_tolerancia"] * r["n_fronteiras_real"] for r in por_trajetoria
    )
    segs_com_iou = sum(
        r["n_segmentos_real"] for r in por_trajetoria if r["iou_segmento_medio"] is not None
    )
    soma_iou = sum(
        r["iou_segmento_medio"] * r["n_segmentos_real"]
        for r in por_trajetoria
        if r["iou_segmento_medio"] is not None
    )
    soma_iou_pesca = sum(
        r["iou_segmento_pesca_medio"] * r["n_segmentos_pesca"]
        for r in por_trajetoria
        if r["iou_segmento_pesca_medio"] is not None
    )
    soma_iou_navegacao = sum(
        r["iou_segmento_navegacao_medio"] * r["n_segmentos_navegacao"]
        for r in por_trajetoria
        if r["iou_segmento_navegacao_medio"] is not None
    )
    total_pesca = sum(r["n_segmentos_pesca"] for r in por_trajetoria)
    total_navegacao = sum(r["n_segmentos_navegacao"] for r in por_trajetoria)
    return {
        "n_segmentos_real": total_segs_real,
        "n_segmentos_previsto": total_segs_prev,
        "segmentos_extra": total_segs_prev - total_segs_real,
        "razao_fragmentacao": _divisao(total_segs_prev, total_segs_real),
        "n_fronteiras_real": total_fr_real,
        "n_fronteiras_prevista": total_fr_prev,
        "n_fronteiras_emparelhadas": pares,
        "fracao_fronteiras_emparelhadas": _divisao(pares, total_fr_real),
        "fronteira_media_pontos": _divisao(soma_distancias, pares) if pares else None,
        "fracao_fronteiras_na_tolerancia": _divisao(fronteiras_na_tolerancia, total_fr_real),
        "iou_segmento_medio": _divisao(soma_iou, segs_com_iou) if segs_com_iou else None,
        "iou_segmento_pesca_medio": (
            _divisao(soma_iou_pesca, total_pesca) if total_pesca else None
        ),
        "iou_segmento_navegacao_medio": (
            _divisao(soma_iou_navegacao, total_navegacao) if total_navegacao else None
        ),
        "n_segmentos_pesca": total_pesca,
        "n_segmentos_navegacao": total_navegacao,
    }


def avaliar_previsoes(previsoes: pd.DataFrame, tolerancia: int = TOLERANCIA_FRONTEIRA) -> dict:
    """Avalia um parquet de previsões: métricas por trajetória, média ± desvio e agregado."""
    faltam = [c for c in COLUNAS_PREVISOES if c not in previsoes.columns]
    assert not faltam, f"colunas em falta nas previsões: {faltam}"
    previsoes = previsoes.sort_values(["id_navio", "n_traj", "indice"], kind="stable")
    por_trajetoria: list[dict] = []
    for (id_navio, n_traj), grupo in previsoes.groupby(["id_navio", "n_traj"], sort=True):
        y_real = grupo["y_real"].to_numpy(np.int64)
        y_prev = grupo["y_prev"].to_numpy(np.int64)
        prob = grupo["prob"].to_numpy(np.float64)
        por_trajetoria.append(
            {
                "id_navio": int(id_navio),
                "n_traj": int(n_traj),
                "n_pontos": int(len(grupo)),
                "ponto": metricas_ponto(y_real, y_prev, prob),
                "segmentos": metricas_segmento(y_real, y_prev, tolerancia),
            }
        )
    assert por_trajetoria, "parquet de previsões vazio"

    ponto = metricas_ponto(
        previsoes["y_real"].to_numpy(np.int64),
        previsoes["y_prev"].to_numpy(np.int64),
        previsoes["prob"].to_numpy(np.float64),
    )
    segmentos = agregar_segmentos([t["segmentos"] for t in por_trajetoria])

    planas = [
        _achatar({**t["ponto"], **{f"seg.{k}": v for k, v in t["segmentos"].items()}})
        for t in por_trajetoria
    ]
    chaves = sorted(set().union(*(p.keys() for p in planas)))
    media: dict[str, float | None] = {}
    desvio: dict[str, float | None] = {}
    for chave in chaves:
        m, d = _media_desvio(planas, chave)
        media[chave] = m
        desvio[chave] = d
    return {
        "agregado": {"ponto": ponto, "segmentos": segmentos},
        "por_trajetoria": {
            "n_trajetorias": len(por_trajetoria),
            "media": media,
            "desvio": desvio,
            "min": {c: min(p[c] for p in planas if c in p) for c in chaves},
            "max": {c: max(p[c] for p in planas if c in p) for c in chaves},
        },
        "trajetorias": por_trajetoria,
    }


def construir_previsoes(
    dataset,
    probabilidades: list[np.ndarray],
    limiar: float = LIMIAR_POR_DEFEITO,
    decisoes: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    """Converte probabilidades por trajetória no parquet de previsões por ponto.

    Usa os ``indices`` que o dataset transporta (posição de cada ponto no Parquet do conjunto): os
    pontos são exatamente os do conjunto avaliado, sem depender da ordem dos batches. Por omissão a
    decisão é ``prob >= limiar``; ``decisoes`` permite impor outra (por exemplo o Viterbi do HMM).
    """
    assert 0.0 < limiar < 1.0, "limiar tem de estar em (0, 1)"
    trajetorias = list(dataset.trajetorias)
    assert len(probabilidades) == len(trajetorias), (
        f"{len(probabilidades)} previsões para {len(trajetorias)} trajetórias"
    )
    if decisoes is not None:
        assert len(decisoes) == len(trajetorias), (
            f"{len(decisoes)} decisões para {len(trajetorias)} trajetórias"
        )
    blocos = []
    for posicao, (trajetoria, prob) in enumerate(zip(trajetorias, probabilidades)):
        prob = np.asarray(prob, dtype=np.float64)
        assert prob.shape == (trajetoria.comprimento,), (
            f"prob com forma {prob.shape} em {trajetoria.id_completo}, "
            f"esperado ({trajetoria.comprimento},)"
        )
        assert np.isfinite(prob).all(), f"prob não finita em {trajetoria.id_completo}"
        assert (prob >= 0).all() and (prob <= 1).all(), (
            f"prob fora de [0, 1] em {trajetoria.id_completo}"
        )
        if decisoes is None:
            y_prev = (prob >= limiar).astype(np.int8)
        else:
            decisao = np.asarray(decisoes[posicao]).astype(np.int8)
            assert decisao.shape == prob.shape, "decisão com forma diferente da probabilidade"
            assert set(np.unique(decisao)).issubset({0, 1}), "decisão tem de ser binária"
            y_prev = decisao
        comprimento = trajetoria.comprimento
        blocos.append(
            pd.DataFrame(
                {
                    "id_navio": np.full(comprimento, trajetoria.id_navio, np.int64),
                    "n_traj": np.full(comprimento, trajetoria.n_traj, np.int64),
                    "indice": trajetoria.indices.astype(np.int64),
                    "y_real": trajetoria.y.astype(np.int8),
                    "y_prev": y_prev,
                    "prob": prob.astype(np.float32),
                }
            )
        )
    previsoes = pd.concat(blocos, ignore_index=True)
    previsoes = previsoes.sort_values(["id_navio", "n_traj", "indice"], kind="stable")
    previsoes = previsoes.reset_index(drop=True)
    assert len(previsoes) == dataset.n_pontos, (
        f"{len(previsoes)} previsões para {dataset.n_pontos} pontos"
    )
    indices = np.sort(previsoes["indice"].to_numpy(np.int64))
    assert np.array_equal(indices, np.arange(dataset.n_pontos)), "índices com buracos ou repetidos"
    return previsoes[list(COLUNAS_PREVISOES)]


def escrever_previsoes(previsoes: pd.DataFrame, destino: Path) -> Path:
    """Escreve o parquet de previsões (cria a pasta se for preciso) e devolve o caminho."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    previsoes.to_parquet(destino, index=False)
    return destino


def verificar_metricas() -> None:
    """Casos pequenos calculados à mão: métricas por ponto, PR-AUC e casos degenerados."""
    y_real = np.array([1, 1, 0, 0])
    y_prev = np.array([1, 0, 0, 1])
    m = metricas_ponto(y_real, y_prev)
    assert np.isclose(m["pesca"]["precisao"], 0.5), m["pesca"]
    assert np.isclose(m["pesca"]["sensibilidade"], 0.5), m["pesca"]
    assert np.isclose(m["pesca"]["f1"], 0.5), m["pesca"]
    assert np.isclose(m["pesca"]["iou"], 1 / 3), m["pesca"]
    assert np.isclose(m["navegacao"]["precisao"], 0.5), m["navegacao"]
    assert np.isclose(m["navegacao"]["f1"], 0.5), m["navegacao"]
    assert np.isclose(m["navegacao"]["iou"], 1 / 3), m["navegacao"]
    assert np.isclose(m["f1_macro"], 0.5), m["f1_macro"]
    assert (m["pesca"]["n_real"], m["pesca"]["n_previsto"]) == (2, 2)
    assert m["pr_auc"] is None, "sem prob não há PR-AUC"

    perfeito = metricas_ponto(y_real, y_real, prob=y_real.astype(float))
    for chave in ("precisao", "sensibilidade", "f1", "iou"):
        assert np.isclose(perfeito["pesca"][chave], 1.0), (chave, perfeito["pesca"])
        assert np.isclose(perfeito["navegacao"][chave], 1.0), (chave, perfeito["navegacao"])
    assert np.isclose(perfeito["f1_macro"], 1.0) and np.isclose(perfeito["pr_auc"], 1.0)

    invertido = metricas_ponto(np.array([0, 1]), np.array([1, 0]), prob=np.array([0.9, 0.1]))
    assert np.isclose(invertido["pr_auc"], 0.5), invertido["pr_auc"]

    degenerado = metricas_ponto(y_real, np.ones(4, dtype=np.int64))
    assert degenerado["navegacao"]["n_previsto"] == 0, degenerado["navegacao"]
    for chave in ("precisao", "sensibilidade", "f1", "iou"):
        assert degenerado["navegacao"][chave] == 0.0, (chave, degenerado["navegacao"])
    assert np.isclose(degenerado["f1_macro"], 0.5 * degenerado["pesca"]["f1"])


def verificar_segmentos() -> None:
    """Segmentação, fronteiras e IoU por segmento num exemplo sintético calculado à mão."""
    y_real = np.array([1, 1, 0, 0, 0, 1])
    y_prev = np.array([1, 1, 1, 0, 0, 0])
    assert segmentos(y_real) == [(0, 2, 1), (2, 5, 0), (5, 6, 1)]
    assert segmentos(y_prev) == [(0, 3, 1), (3, 6, 0)]
    assert fronteiras(y_real).tolist() == [2, 5]
    assert fronteiras(y_prev).tolist() == [3]
    assert emparelhar_fronteiras([2, 5], [3]) == [(2, 3, 1)]
    assert emparelhar_fronteiras([5], [2, 3]) == [(5, 3, 2)]
    assert emparelhar_fronteiras([], [3]) == []

    ious = iou_por_segmento(y_real, y_prev)
    assert np.allclose(ious[1], [2 / 3, 0.0]), ious[1]
    assert np.allclose(ious[0], [0.5]), ious[0]

    s = metricas_segmento(y_real, y_prev, tolerancia=3)
    assert s["n_segmentos_real"] == 3 and s["n_segmentos_previsto"] == 2, s
    assert s["segmentos_extra"] == -1, s["segmentos_extra"]
    assert np.isclose(s["razao_fragmentacao"], 2 / 3), s["razao_fragmentacao"]
    assert (s["n_fronteiras_real"], s["n_fronteiras_prevista"]) == (2, 1), s
    assert s["n_fronteiras_emparelhadas"] == 1, s["n_fronteiras_emparelhadas"]
    assert np.isclose(s["fracao_fronteiras_emparelhadas"], 0.5), s
    assert np.isclose(s["fronteira_media_pontos"], 1.0), s["fronteira_media_pontos"]
    assert np.isclose(s["fracao_fronteiras_na_tolerancia"], 0.5), s
    assert np.isclose(s["iou_segmento_medio"], (2 / 3 + 0.0 + 0.5) / 3), s["iou_segmento_medio"]
    assert np.isclose(s["iou_segmento_pesca_medio"], (2 / 3 + 0.0) / 2), s
    assert np.isclose(s["iou_segmento_navegacao_medio"], 0.5), s

    constante = metricas_segmento(np.ones(5), np.ones(5))
    assert constante["n_segmentos_real"] == 1 and constante["n_fronteiras_real"] == 0, constante
    assert constante["fracao_fronteiras_emparelhadas"] == 1.0, constante
    assert np.isclose(constante["iou_segmento_medio"], 1.0), constante
    assert constante["fronteira_media_pontos"] is None, constante

    fragmentado = metricas_segmento(np.ones(6), np.array([1, 0, 1, 0, 1, 0]))
    assert fragmentado["n_segmentos_previsto"] == 6 and fragmentado["segmentos_extra"] == 5
    # O único segmento real (6 pontos) só encontra 1 ponto do melhor segmento previsto da mesma
    # classe: IoU = 1 / (6 + 1 - 1) = 1/6, apesar de metade dos pontos estar bem classificada.
    assert np.isclose(fragmentado["iou_segmento_medio"], 1 / 6), fragmentado["iou_segmento_medio"]


def guardar_avaliacao(avaliacao: dict, destino: Path) -> Path:
    """Escreve o JSON com o resultado da avaliação (agregado, média ± desvio e por trajetória)."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(avaliacao, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destino



def verificar_previsoes() -> None:
    """O parquet de previsões cobre exatamente os índices do dataset e respeita o limiar."""
    from types import SimpleNamespace

    primeira = SimpleNamespace(
        id_navio=1,
        n_traj=1,
        id_completo=(1, 1),
        comprimento=3,
        indices=np.array([0, 1, 2], np.int64),
        y=np.array([1, 0, 1], np.int64),
    )
    segunda = SimpleNamespace(
        id_navio=1,
        n_traj=2,
        id_completo=(1, 2),
        comprimento=2,
        indices=np.array([3, 4], np.int64),
        y=np.array([0, 1], np.int64),
    )
    dataset = SimpleNamespace(trajetorias=[primeira, segunda], n_pontos=5)
    previsoes = construir_previsoes(
        dataset, [np.array([0.9, 0.2, 0.6]), np.array([0.4, 0.7])]
    )
    assert list(previsoes.columns) == list(COLUNAS_PREVISOES), list(previsoes.columns)
    assert previsoes["indice"].tolist() == [0, 1, 2, 3, 4], previsoes["indice"].tolist()
    assert previsoes["y_real"].tolist() == [1, 0, 1, 0, 1], previsoes["y_real"].tolist()
    assert previsoes["y_prev"].tolist() == [1, 0, 1, 0, 1], previsoes["y_prev"].tolist()
    assert np.allclose(previsoes["prob"].to_numpy(), [0.9, 0.2, 0.6, 0.4, 0.7])

    resultado = avaliar_previsoes(previsoes)
    assert resultado["agregado"]["ponto"]["n_pontos"] == 5, resultado["agregado"]["ponto"]
    assert resultado["por_trajetoria"]["n_trajetorias"] == 2
    assert np.isclose(resultado["agregado"]["ponto"]["f1_macro"], 1.0)

    try:
        construir_previsoes(dataset, [np.array([1.5, 0.2, 0.6]), np.array([0.4, 0.7])])
    except AssertionError:
        pass
    else:
        raise AssertionError("probabilidade fora de [0, 1] devia falhar")



def imprimir_avaliacao(avaliacao: dict, titulo: str) -> None:
    """Mostra o agregado e a média ± desvio por trajetória em três linhas."""
    ponto = avaliacao["agregado"]["ponto"]
    seg = avaliacao["agregado"]["segmentos"]
    pr_auc = "n/d" if ponto["pr_auc"] is None else f"{ponto['pr_auc']:.3f}"
    fronteira = (
        "n/d"
        if seg["fronteira_media_pontos"] is None
        else f"{seg['fronteira_media_pontos']:.1f} pts"
    )
    print(
        f"{titulo}: {ponto['n_pontos']} pontos | "
        f"pesca P {ponto['pesca']['precisao']:.3f} S {ponto['pesca']['sensibilidade']:.3f} "
        f"F1 {ponto['pesca']['f1']:.3f} IoU {ponto['pesca']['iou']:.3f} | "
        f"navegação P {ponto['navegacao']['precisao']:.3f} "
        f"S {ponto['navegacao']['sensibilidade']:.3f} F1 {ponto['navegacao']['f1']:.3f} "
        f"IoU {ponto['navegacao']['iou']:.3f} | F1 macro {ponto['f1_macro']:.3f} | "
        f"PR-AUC {pr_auc}"
    )
    print(
        f"  segmentos: real {seg['n_segmentos_real']} previsto {seg['n_segmentos_previsto']} "
        f"(razão {seg['razao_fragmentacao']:.3f}) | fronteira média {fronteira} | "
        f"{100 * seg['fracao_fronteiras_na_tolerancia']:.0f}% dentro da tolerância | "
        f"IoU segmento {seg['iou_segmento_medio']:.3f}"
    )
    media = avaliacao["por_trajetoria"]["media"]
    desvio = avaliacao["por_trajetoria"]["desvio"]
    print(
        f"  por trajetória ({avaliacao['por_trajetoria']['n_trajetorias']}): "
        f"F1 macro {media['f1_macro']:.3f} ± {desvio['f1_macro']:.3f} | "
        f"F1 pesca {media['pesca.f1']:.3f} ± {desvio['pesca.f1']:.3f} | "
        f"IoU pesca {media['pesca.iou']:.3f} ± {desvio['pesca.iou']:.3f}"
    )


def principal() -> None:
    """Ponto de entrada: corre os testes internos e avalia um parquet de previsões."""
    parser = argparse.ArgumentParser(
        description="Passo 8: métricas por ponto e por segmento sobre um parquet de previsões."
    )
    parser.add_argument("--previsoes", default=None, help="parquet com as colunas de previsões")
    parser.add_argument("--modelo", default=None, help="nome do modelo, só para o título")
    parser.add_argument("--tolerancia", type=int, default=TOLERANCIA_FRONTEIRA)
    parser.add_argument("--saida", default=None, help="JSON onde guardar a avaliação")
    parser.add_argument("--testes", action="store_true", help="só correr os testes internos")
    argumentos = parser.parse_args()

    verificar_metricas()
    verificar_segmentos()
    verificar_previsoes()
    print("Testes internos das métricas, segmentos, fronteiras e previsões: ok")
    if argumentos.testes:
        return
    assert argumentos.previsoes, "indicar --previsoes (ou --testes para só validar as métricas)"

    caminho = Path(argumentos.previsoes)
    assert caminho.exists(), f"não existe {caminho}"
    previsoes = pd.read_parquet(caminho)
    avaliacao = avaliar_previsoes(previsoes, argumentos.tolerancia)
    imprimir_avaliacao(avaliacao, argumentos.modelo or caminho.stem)
    if argumentos.saida:
        destino = guardar_avaliacao(avaliacao, Path(argumentos.saida))
        print(f"Avaliação guardada em {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    principal()
