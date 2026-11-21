"""Passo 6 da fase 1: baselines B0 (sempre pesca) e B1 (LogReg, HistGradientBoosting).

Os dois B1 veem **só as features do próprio ponto** (sem contexto temporal): são o que a fase 1 tem
de bater com o modelo sequencial. Ambos são ajustados **só ao treino** e a escolha do
hiperparâmetro é feita na **validação** (F1 macro; desempate por PR-AUC). O teste não é tocado.

Cada modelo escreve o parquet de previsões da validação (``data/previsoes/<EXP>/``) e o JSON da
avaliação (``data/resultados/<EXP>/``), no mesmo formato para todos os modelos da fase 1.

Executar a partir da raiz do repositório:

    python src/baselines.py                       # as experiências do config (EXP-001 e EXP-002)
    python src/baselines.py --exp EXP-001_features_com_xy
    python src/baselines.py --modelos B0_sempre_pesca
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from avaliar import (
    PASTA_PREVISOES,
    avaliar_previsoes,
    construir_previsoes,
    escrever_previsoes,
    guardar_avaliacao,
    imprimir_avaliacao,
    metricas_ponto,
)
from dataset import DIVISOES, carregar_dataset

RAIZ = Path(__file__).resolve().parents[1]
CONFIG_POR_DEFEITO = RAIZ / "configs" / "baselines.json"
PASTA_RESULTADOS = RAIZ / "data" / "resultados"

MODELOS = ("B0_sempre_pesca", "B1_logreg", "B1_hgb")


def ler_config(caminho: Path = CONFIG_POR_DEFEITO) -> dict:
    """Lê e valida a configuração dos baselines (grelhas, limiar, semente e experiências)."""
    assert caminho.exists(), f"configuração inexistente: {caminho}"
    config = json.loads(caminho.read_text(encoding="utf-8"))
    esperados = {
        "nome",
        "descricao",
        "divisao",
        "semente",
        "limiar",
        "experiencias",
        "logreg",
        "hist_gradient_boosting",
    }
    assert set(config) == esperados, f"campos inesperados em {caminho.name}: {sorted(config)}"
    assert config["divisao"] in DIVISOES, f"divisão desconhecida: {config['divisao']}"
    assert 0.0 < float(config["limiar"]) < 1.0, "limiar tem de estar em (0, 1)"
    assert int(config["semente"]) >= 0, "semente tem de ser >= 0"
    assert config["experiencias"], "sem experiências para correr"
    assert config["logreg"]["C"], "grelha de C vazia"
    assert config["hist_gradient_boosting"]["learning_rate"], "grelha de learning_rate vazia"
    return config


def empilhar(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Empilha as features e os rótulos de todas as trajetórias do conjunto (ponto a ponto)."""
    x = np.concatenate([t.x for t in dataset.trajetorias]).astype(np.float32)
    y = np.concatenate([t.y for t in dataset.trajetorias]).astype(np.int8)
    assert x.ndim == 2 and y.shape == (x.shape[0],), "formas inconsistentes ao empilhar"
    assert len(x) == dataset.n_pontos, f"{len(x)} pontos empilhados != {dataset.n_pontos}"
    return x, y


def prever_por_trajetoria(modelo, dataset) -> list[np.ndarray]:
    """Probabilidade de pesca por trajetória, na ordem dos itens do dataset."""
    probabilidades = []
    for trajetoria in dataset.trajetorias:
        prob = modelo.predict_proba(trajetoria.x)[:, 1].astype(np.float64)
        assert prob.shape == (trajetoria.comprimento,), "prob com forma errada"
        probabilidades.append(prob)
    return probabilidades


def prever_sempre_pesca(dataset) -> list[np.ndarray]:
    """B0: probabilidade 1,0 em todos os pontos (baseline trivial de referência)."""
    return [np.ones(t.comprimento, dtype=np.float64) for t in dataset.trajetorias]


def ajustar_logreg(x_treino: np.ndarray, y_treino: np.ndarray, config: dict) -> list[dict]:
    """Ajusta uma LogReg por valor de ``C`` e devolve os candidatos com tempo e convergência."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    candidatos = []
    for valor_c in config["C"]:
        modelo = LogisticRegression(
            C=float(valor_c), max_iter=int(config["max_iter"]), solver="lbfgs"
        )
        inicio = time.perf_counter()
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always", ConvergenceWarning)
            modelo.fit(x_treino, y_treino)
        candidatos.append(
            {
                "modelo": modelo,
                "parametros": {"C": float(valor_c)},
                "tempo_ajuste_s": time.perf_counter() - inicio,
                "iteracoes": int(modelo.n_iter_[0]),
                "convergiu": not any(issubclass(a.category, ConvergenceWarning) for a in avisos),
            }
        )
    return candidatos


def ajustar_hgb(
    x_treino: np.ndarray, y_treino: np.ndarray, config: dict, semente: int
) -> list[dict]:
    """Ajusta um HistGradientBoosting por combinação da grelha (sem early stopping interno)."""
    assert not config.get("early_stopping", False), (
        "early stopping interno tiraria pontos ao treino; a paragem é decidida na validação"
    )
    candidatos = []
    for learning_rate in config["learning_rate"]:
        for folhas in config["max_leaf_nodes"]:
            modelo = HistGradientBoostingClassifier(
                learning_rate=float(learning_rate),
                max_leaf_nodes=int(folhas),
                max_iter=int(config["max_iter"]),
                early_stopping=False,
                random_state=int(semente),
            )
            inicio = time.perf_counter()
            modelo.fit(x_treino, y_treino)
            candidatos.append(
                {
                    "modelo": modelo,
                    "parametros": {
                        "learning_rate": float(learning_rate),
                        "max_leaf_nodes": int(folhas),
                        "max_iter": int(config["max_iter"]),
                    },
                    "tempo_ajuste_s": time.perf_counter() - inicio,
                    "iteracoes": int(config["max_iter"]),
                    "convergiu": True,
                }
            )
    return candidatos


def escolher_na_validacao(
    candidatos: list[dict], validacao, y_val: np.ndarray, limiar: float
) -> dict:
    """Pontua cada candidato na validação (F1 macro, PR-AUC) e devolve o melhor."""
    assert candidatos, "sem candidatos para escolher"
    for candidato in candidatos:
        probabilidades = prever_por_trajetoria(candidato["modelo"], validacao)
        prob = np.concatenate(probabilidades)
        y_prev = (prob >= limiar).astype(np.int64)
        metricas = metricas_ponto(y_val, y_prev, prob)
        candidato["prob_val"] = probabilidades
        candidato["f1_macro_val"] = metricas["f1_macro"]
        candidato["pr_auc_val"] = metricas["pr_auc"]
    return max(
        candidatos,
        key=lambda c: (c["f1_macro_val"], c["pr_auc_val"] if c["pr_auc_val"] is not None else 0.0),
    )


def _resumo_candidato(candidato: dict) -> dict:
    """Resumo de um candidato sem o modelo nem as probabilidades (para o JSON)."""
    return {
        **candidato["parametros"],
        "f1_macro_val": candidato["f1_macro_val"],
        "pr_auc_val": candidato["pr_auc_val"],
        "tempo_ajuste_s": candidato["tempo_ajuste_s"],
        "iteracoes": candidato["iteracoes"],
        "convergiu": candidato["convergiu"],
    }


def guardar_resultado(
    nome: str,
    probabilidades: list[np.ndarray],
    validacao,
    experiencia: str,
    config: dict,
) -> dict:
    """Escreve o parquet de previsões e o JSON de avaliação de um modelo na validação."""
    limiar = float(config["limiar"])
    previsoes = construir_previsoes(validacao, probabilidades, limiar)
    parquet = escrever_previsoes(
        previsoes, PASTA_PREVISOES / experiencia / f"{nome}_validacao.parquet"
    )
    avaliacao = avaliar_previsoes(previsoes)
    avaliacao.update(
        {
            "modelo": nome,
            "experiencia": experiencia,
            "divisao": config["divisao"],
            "conjunto": "validacao",
            "limiar": limiar,
        }
    )
    json_saida = guardar_avaliacao(
        avaliacao, PASTA_RESULTADOS / experiencia / f"{nome}_validacao.json"
    )
    imprimir_avaliacao(avaliacao, f"{nome} [{experiencia}]")
    print(
        f"  previsões: {parquet.relative_to(RAIZ)} | avaliação: {json_saida.relative_to(RAIZ)}"
    )
    return avaliacao


def correr_experiencia(experiencia: str, config: dict, modelos: tuple[str, ...]) -> dict:
    """Corre B0 e/ou B1 numa experiência e devolve o registo dos hiperparâmetros e escolhas."""
    assert config["divisao"] == "principal", (
        "esta fase não toca na divisão por navio; a sensibilidade fica para o passo 9"
    )
    treino = carregar_dataset(experiencia, "treino", config["divisao"])
    validacao = carregar_dataset(experiencia, "validacao", config["divisao"])
    print(
        f"\n{experiencia} | treino {len(treino)} trajetórias / {treino.n_pontos} pontos | "
        f"validação {len(validacao)} / {validacao.n_pontos} | "
        f"{len(treino.features)} features ({', '.join(treino.features)})"
    )
    x_treino, y_treino = empilhar(treino)
    y_val = np.concatenate([t.y for t in validacao.trajetorias]).astype(np.int64)
    assert len(y_val) == validacao.n_pontos, "pontos da validação perdidos ao empilhar"
    print(
        f"  fração de pesca: treino {float(np.mean(y_treino)):.4f} | "
        f"validação {float(np.mean(y_val)):.4f}"
    )

    registo = {
        "experiencia": experiencia,
        "divisao": config["divisao"],
        "conjunto": "validacao",
        "limiar": float(config["limiar"]),
        "semente": int(config["semente"]),
        "pontos_treino": treino.n_pontos,
        "pontos_validacao": validacao.n_pontos,
        "normalizacao": f"data/features/{experiencia}/normalizacao.json (ajustada ao treino)",
        "threads_openmp": os.environ.get("OMP_NUM_THREADS", "não definido"),
        "modelos": {},
    }

    if "B0_sempre_pesca" in modelos:
        inicio = time.perf_counter()
        probabilidades = prever_sempre_pesca(validacao)
        guardar_resultado("B0_sempre_pesca", probabilidades, validacao, experiencia, config)
        registo["modelos"]["B0_sempre_pesca"] = {
            "grelha": [],
            "escolhido": {"regra": "prever sempre pesca (probabilidade 1,0)"},
            "tempo_ajuste_s": time.perf_counter() - inicio,
        }

    if "B1_logreg" in modelos:
        inicio = time.perf_counter()
        candidatos = ajustar_logreg(x_treino, y_treino, config["logreg"])
        melhor = escolher_na_validacao(candidatos, validacao, y_val, config["limiar"])
        total = time.perf_counter() - inicio
        for candidato in candidatos:
            resumo = _resumo_candidato(candidato)
            print(
                f"  B1_logreg C={resumo['C']}: F1 macro val {resumo['f1_macro_val']:.4f} | "
                f"PR-AUC {resumo['pr_auc_val']:.4f} | {resumo['tempo_ajuste_s']:.1f} s | "
                f"iterações {resumo['iteracoes']} | convergiu {resumo['convergiu']}"
            )
        guardar_resultado("B1_logreg", melhor["prob_val"], validacao, experiencia, config)
        registo["modelos"]["B1_logreg"] = {
            "grelha": [_resumo_candidato(c) for c in candidatos],
            "escolhido": _resumo_candidato(melhor),
            "tempo_ajuste_s": total,
        }

    if "B1_hgb" in modelos:
        inicio = time.perf_counter()
        candidatos = ajustar_hgb(
            x_treino, y_treino, config["hist_gradient_boosting"], config["semente"]
        )
        melhor = escolher_na_validacao(candidatos, validacao, y_val, config["limiar"])
        total = time.perf_counter() - inicio
        for candidato in candidatos:
            resumo = _resumo_candidato(candidato)
            print(
                f"  B1_hgb lr={resumo['learning_rate']} folhas={resumo['max_leaf_nodes']}: "
                f"F1 macro val {resumo['f1_macro_val']:.4f} | "
                f"PR-AUC {resumo['pr_auc_val']:.4f} | {resumo['tempo_ajuste_s']:.1f} s"
            )
        guardar_resultado("B1_hgb", melhor["prob_val"], validacao, experiencia, config)
        registo["modelos"]["B1_hgb"] = {
            "grelha": [_resumo_candidato(c) for c in candidatos],
            "escolhido": _resumo_candidato(melhor),
            "tempo_ajuste_s": total,
        }

    assert registo["modelos"], f"nenhum modelo escolhido para {experiencia}"
    destino = PASTA_RESULTADOS / experiencia / "baselines_parametros.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(registo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  hiperparâmetros e escolhas: {destino.relative_to(RAIZ)}")
    return registo


def principal() -> None:
    """Ponto de entrada: corre os baselines nas experiências pedidas, avaliando na validação."""
    parser = argparse.ArgumentParser(
        description="Passo 6: B0 (sempre pesca) e B1 (LogReg, HistGradientBoosting)."
    )
    parser.add_argument("--config", default=str(CONFIG_POR_DEFEITO))
    parser.add_argument("--exp", nargs="+", default=None, help="experiências a correr")
    parser.add_argument(
        "--modelos", nargs="+", choices=MODELOS, default=list(MODELOS), help="modelos a correr"
    )
    argumentos = parser.parse_args()
    config = ler_config(Path(argumentos.config))
    experiencias = argumentos.exp or config["experiencias"]
    for experiencia in experiencias:
        correr_experiencia(experiencia, config, tuple(argumentos.modelos))
    print("\nBaselines concluídos. O conjunto de teste não foi usado.")


if __name__ == "__main__":
    principal()
