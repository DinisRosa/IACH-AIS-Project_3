"""Passo 6 da fase 1: HMM de 2 estados sobre ``euc_speed`` e ``signed_turn`` (Peel e Good, 2011).

O modelo tem dois estados (navegação e pesca), transições de Markov entre pontos consecutivos e
emissões gaussianas diagonais. É ajustado por **EM (Baum-Welch) não supervisionado, só ao treino**,
trabalhando em cada **trajetória inteira** (o forward-backward nunca atravessa fronteiras de
trajetórias). A réplica de inicialização e o número de iterações são escolhidos na **validação**
(F1 macro), nunca no teste.

Duas decisões de desenho, ambas registadas:

- **Implementação própria**, em vez de ``hmmlearn``: o ``requirements.txt`` deixava a dependência em
  aberto, duas dimensões e dois estados não precisam de biblioteca, e assim o E-step fica auditável
  (teste com dados sintéticos gerados por um HMM conhecido, em ``verificar_sintetico``).
- **Viterbi para decidir** (a sequência de estados mais provável, como no artigo) e **posterior**
  (probabilidade de pesca) na coluna ``prob``; a versão com limiar 0,5 sobre o posterior é também
  medida e fica registada, para se ver o que a suavização do Viterbi muda.

Executar a partir da raiz do repositório:

    python src/hmm.py --testes                       # só os testes internos (rápidos)
    python src/hmm.py --exp EXP-001_features_com_xy
    python src/hmm.py                                # as experiências do config
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

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
CONFIG_POR_DEFEITO = RAIZ / "configs" / "hmm.json"
PASTA_RESULTADOS = RAIZ / "data" / "resultados"

N_ESTADOS = 2


def ler_config(caminho: Path = CONFIG_POR_DEFEITO) -> dict:
    """Lê e valida a configuração do HMM (features, iterações, réplicas, semente)."""
    assert caminho.exists(), f"configuração inexistente: {caminho}"
    config = json.loads(caminho.read_text(encoding="utf-8"))
    esperados = {
        "nome",
        "descricao",
        "divisao",
        "semente",
        "limiar",
        "experiencias",
        "features",
        "n_repeticoes_em",
        "tolerancia",
        "var_minima",
        "n_replicas",
    }
    assert set(config) == esperados, f"campos inesperados em {caminho.name}: {sorted(config)}"
    assert config["divisao"] in DIVISOES, f"divisão desconhecida: {config['divisao']}"
    assert config["features"], "sem features para o HMM"
    assert int(config["n_repeticoes_em"]) >= 1, "n_repeticoes_em tem de ser >= 1"
    assert int(config["n_replicas"]) >= 1, "n_replicas tem de ser >= 1"
    assert 0.0 < float(config["tolerancia"]) < 1.0, "tolerância fora de (0, 1)"
    assert float(config["var_minima"]) > 0.0, "var_minima tem de ser > 0"
    return config


def _lse2(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) estável, para dois valores."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    maior = a if a > b else b
    return maior + math.log1p(math.exp(-abs(a - b)))


def log_emissao(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Log-densidade gaussiana diagonal de cada ponto em cada estado: matriz ``[T, 2]``."""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == mu.shape[1], "features e médias com dimensões diferentes"
    desvio = x[:, None, :] - mu[None, :, :]
    return -0.5 * np.sum(np.log(2.0 * math.pi * var)[None, :, :] + desvio**2 / var[None, :, :], axis=2)


def forward(obs: list[list[float]], log_pi: list[float], log_a: list[list[float]]):
    """Passagem para a frente em log-espaço; devolve ``alfa [T, 2]`` e a log-verosimilhança."""
    a00, a01 = log_a[0]
    a10, a11 = log_a[1]
    alfa = [[log_pi[0] + obs[0][0], log_pi[1] + obs[0][1]]]
    for t in range(1, len(obs)):
        p0, p1 = alfa[-1]
        o = obs[t]
        alfa.append([o[0] + _lse2(a00 + p0, a10 + p1), o[1] + _lse2(a01 + p0, a11 + p1)])
    return alfa, _lse2(alfa[-1][0], alfa[-1][1])


def backward(obs: list[list[float]], log_a: list[list[float]]) -> list[list[float]]:
    """Passagem para trás em log-espaço; devolve ``beta [T, 2]``."""
    a00, a01 = log_a[0]
    a10, a11 = log_a[1]
    beta = [[0.0, 0.0] for _ in range(len(obs))]
    for t in range(len(obs) - 2, -1, -1):
        o = obs[t + 1]
        b0, b1 = beta[t + 1]
        beta[t][0] = _lse2(a00 + o[0] + b0, a01 + o[1] + b1)
        beta[t][1] = _lse2(a10 + o[0] + b0, a11 + o[1] + b1)
    return beta


def viterbi(obs: list[list[float]], log_pi: list[float], log_a: list[list[float]]) -> np.ndarray:
    """Sequência de estados mais provável (Viterbi) em log-espaço."""
    a00, a01 = log_a[0]
    a10, a11 = log_a[1]
    delta = [log_pi[0] + obs[0][0], log_pi[1] + obs[0][1]]
    ponteiros = []
    for t in range(1, len(obs)):
        o = obs[t]
        de0, de1 = delta
        candidato0 = de0 + a00
        candidato1 = de1 + a10
        if candidato0 >= candidato1:
            novo0, de0_origem = candidato0, 0
        else:
            novo0, de0_origem = candidato1, 1
        if de0 + a01 >= de1 + a11:
            novo1, de1_origem = de0 + a01, 0
        else:
            novo1, de1_origem = de1 + a11, 1
        ponteiros.append((de0_origem, de1_origem))
        delta = [novo0 + o[0], novo1 + o[1]]
    caminho = np.zeros(len(obs), dtype=np.int64)
    caminho[-1] = 0 if delta[0] >= delta[1] else 1
    for t in range(len(obs) - 2, -1, -1):
        caminho[t] = ponteiros[t][caminho[t + 1]]
    return caminho


def posteriores(
    x: np.ndarray, parametros: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Posteriores por ponto (``[T, 2]``, já normalizados) e a sequência de Viterbi."""
    obs = log_emissao(x, parametros["mu"], parametros["var"]).tolist()
    log_pi = np.log(parametros["pi"]).tolist()
    log_a = np.log(parametros["A"]).tolist()
    alfa, loglik = forward(obs, log_pi, log_a)
    beta = backward(obs, log_a)
    gama = np.empty((len(obs), N_ESTADOS), dtype=np.float64)
    for t in range(len(obs)):
        for i in range(N_ESTADOS):
            gama[t, i] = math.exp(alfa[t][i] + beta[t][i] - loglik)
    gama = np.clip(gama, 0.0, None)
    assert np.all(gama.sum(axis=1) > 0.0), "posterior nulo numa linha"
    gama /= gama.sum(axis=1, keepdims=True)
    gama = np.clip(gama, 0.0, 1.0)
    assert np.allclose(gama.sum(axis=1), 1.0, atol=1e-9), "posteriores não normalizados"
    assert gama.min() >= 0.0 and gama.max() <= 1.0, "posterior fora de [0, 1]"
    return gama, viterbi(obs, log_pi, log_a)


def inicializar(
    sequencias: list[np.ndarray], config: dict, rng: np.random.Generator, replica: int
) -> dict:
    """Parâmetros iniciais: réplica 0 por quantis, as restantes com perturbação aleatória."""
    todos = np.concatenate(sequencias).astype(np.float64)
    dimensao = todos.shape[1]
    var_global = todos.var(axis=0) + float(config["var_minima"])
    if replica == 0:
        mediana = float(np.median(todos[:, 0]))
        baixo = todos[todos[:, 0] <= mediana]
        alto = todos[todos[:, 0] > mediana]
        assert len(baixo) > 0 and len(alto) > 0, "quantis degenerados na inicialização"
        mu = np.vstack([baixo.mean(axis=0), alto.mean(axis=0)])
    else:
        mu = np.median(todos, axis=0)[None, :] + rng.normal(
            0.0, 0.75, size=(N_ESTADOS, dimensao)
        ) * np.sqrt(var_global)[None, :]
    var = np.tile(var_global, (N_ESTADOS, 1)) * (1.0 + 0.5 * rng.random((N_ESTADOS, dimensao)))
    transicao = np.full((N_ESTADOS, N_ESTADOS), 0.1 / (N_ESTADOS - 1))
    np.fill_diagonal(transicao, 0.9)
    return {
        "pi": np.full(N_ESTADOS, 1.0 / N_ESTADOS),
        "A": transicao,
        "mu": mu,
        "var": var,
    }


def acumular_suficientes(sequencias: list[np.ndarray], parametros: dict, config: dict) -> dict:
    """E-step: estatísticas suficientes (posteriores e transições) e log-verosimilhança."""
    dimensao = parametros["mu"].shape[1]
    log_pi = np.log(parametros["pi"]).tolist()
    log_a = np.log(parametros["A"]).tolist()
    soma = {
        "pi": np.zeros(N_ESTADOS),
        "A": np.zeros((N_ESTADOS, N_ESTADOS)),
        "gama": np.zeros(N_ESTADOS),
        "gama_ate_penultimo": np.zeros(N_ESTADOS),
        "gama_x": np.zeros((N_ESTADOS, dimensao)),
        "gama_x2": np.zeros((N_ESTADOS, dimensao)),
        "loglik": 0.0,
        "pontos": 0,
    }
    for x in sequencias:
        obs = log_emissao(x, parametros["mu"], parametros["var"]).tolist()
        alfa, loglik = forward(obs, log_pi, log_a)
        beta = backward(obs, log_a)
        total = len(obs)
        linha_x = x.tolist()
        soma["loglik"] += loglik
        soma["pontos"] += total
        for t in range(total):
            a0, a1 = alfa[t]
            b0, b1 = beta[t]
            g0 = math.exp(a0 + b0 - loglik)
            g1 = math.exp(a1 + b1 - loglik)
            if t == 0:
                soma["pi"][0] += g0
                soma["pi"][1] += g1
            if t < total - 1:
                proximo = obs[t + 1]
                soma["A"][0][0] += math.exp(a0 + log_a[0][0] + proximo[0] + beta[t + 1][0] - loglik)
                soma["A"][0][1] += math.exp(a0 + log_a[0][1] + proximo[1] + beta[t + 1][1] - loglik)
                soma["A"][1][0] += math.exp(a1 + log_a[1][0] + proximo[0] + beta[t + 1][0] - loglik)
                soma["A"][1][1] += math.exp(a1 + log_a[1][1] + proximo[1] + beta[t + 1][1] - loglik)
                soma["gama_ate_penultimo"][0] += g0
                soma["gama_ate_penultimo"][1] += g1
            soma["gama"][0] += g0
            soma["gama"][1] += g1
            for d, valor in enumerate(linha_x[t]):
                soma["gama_x"][0][d] += g0 * valor
                soma["gama_x"][1][d] += g1 * valor
                soma["gama_x2"][0][d] += g0 * valor * valor
                soma["gama_x2"][1][d] += g1 * valor * valor
    return soma


def atualizar(soma: dict, config: dict) -> dict:
    """M-step: máxima verosimilhança a partir das estatísticas suficientes."""
    assert np.all(soma["gama_ate_penultimo"] > 0), "estado sem pontos antes do último"
    pi = soma["pi"] / soma["pi"].sum()
    transicao = soma["A"] / soma["gama_ate_penultimo"][:, None]
    transicao = np.clip(transicao, 1e-12, None)
    transicao /= transicao.sum(axis=1, keepdims=True)
    mu = soma["gama_x"] / soma["gama"][:, None]
    var = soma["gama_x2"] / soma["gama"][:, None] - mu**2
    var = np.clip(var, float(config["var_minima"]), None)
    assert np.allclose(pi.sum(), 1.0) and np.allclose(transicao.sum(axis=1), 1.0)
    return {"pi": pi, "A": transicao, "mu": mu, "var": var}


def baum_welch(
    sequencias: list[np.ndarray], config: dict, semente: int, replica: int
) -> dict:
    """Ajusta o HMM por EM não supervisionado numa réplica de inicialização."""
    rng = np.random.default_rng(semente + replica)
    parametros = inicializar(sequencias, config, rng, replica)
    historico: list[float] = []
    loglik_anterior = -math.inf
    iteracoes = 0
    for iteracoes in range(1, int(config["n_repeticoes_em"]) + 1):
        soma = acumular_suficientes(sequencias, parametros, config)
        loglik_por_ponto = soma["loglik"] / soma["pontos"]
        historico.append(float(loglik_por_ponto))
        parametros = atualizar(soma, config)
        if (
            math.isfinite(loglik_anterior)
            and abs(loglik_por_ponto - loglik_anterior)
            < float(config["tolerancia"]) * abs(loglik_anterior)
        ):
            break
        loglik_anterior = loglik_por_ponto
    return {
        "parametros": parametros,
        "historico_loglik_por_ponto": historico,
        "iteracoes": iteracoes,
        "convergiu": len(historico) > 1 and iteracoes < int(config["n_repeticoes_em"]),
    }


def indices_features(dataset, config: dict) -> list[int]:
    """Posições das features do HMM na matriz de features do dataset (por nome, não por ordem)."""
    em_falta = [f for f in config["features"] if f not in dataset.features]
    assert not em_falta, f"features do HMM em falta no dataset: {em_falta}"
    return [dataset.features.index(f) for f in config["features"]]


def sequencias_de(dataset, config: dict) -> list[np.ndarray]:
    """Uma sequência (trajetória inteira) por trajetória, só com as features do HMM."""
    idx = indices_features(dataset, config)
    return [t.x[:, idx].astype(np.float64) for t in dataset.trajetorias]


def mapear_estado_pesca(parametros: dict, treino, config: dict) -> tuple[int, list[float]]:
    """Estado com maior fração de pontos de pesca no treino (o EM é não supervisionado)."""
    idx = indices_features(treino, config)
    pesos_estado = np.zeros(N_ESTADOS)
    pontos_estado = np.zeros(N_ESTADOS)
    for trajetoria in treino.trajetorias:
        x = trajetoria.x[:, idx].astype(np.float64)
        obs = log_emissao(x, parametros["mu"], parametros["var"]).tolist()
        caminho = viterbi(obs, np.log(parametros["pi"]).tolist(), np.log(parametros["A"]).tolist())
        for estado in range(N_ESTADOS):
            mascara = caminho == estado
            pesos_estado[estado] += float(np.sum(trajetoria.y[mascara]))
            pontos_estado[estado] += float(np.sum(mascara))
    assert np.all(pontos_estado > 0), "um estado ficou sem pontos no treino"
    fracao = pesos_estado / pontos_estado
    return int(np.argmax(fracao)), fracao.tolist()


def ajustar_replicas(treino, config: dict) -> list[dict]:
    """Ajusta uma réplica de EM por inicialização; falhas num estado colapsado não abortam."""
    sequencias = sequencias_de(treino, config)
    replicas = []
    for replica in range(int(config["n_replicas"])):
        inicio = time.perf_counter()
        try:
            resultado = baum_welch(sequencias, config, int(config["semente"]), replica)
        except AssertionError as erro:
            print(f"  réplica {replica}: colapsou ({erro})")
            replicas.append({"replica": replica, "falhou": str(erro)})
            continue
        resultado["replica"] = replica
        resultado["tempo_s"] = time.perf_counter() - inicio
        resultado["estado_pesca"], resultado["fracao_pesca_por_estado"] = mapear_estado_pesca(
            resultado["parametros"], treino, config
        )
        replicas.append(resultado)
    assert any("falhou" not in r for r in replicas), "todas as réplicas colapsaram"
    return replicas


def prever(dataset, parametros: dict, estado_pesca: int, config: dict):
    """Posterior do estado de pesca e decisão por Viterbi, trajetória a trajetória."""
    idx = indices_features(dataset, config)
    probabilidades = []
    decisoes = []
    for trajetoria in dataset.trajetorias:
        x = trajetoria.x[:, idx].astype(np.float64)
        gama, caminho = posteriores(x, parametros)
        probabilidades.append(gama[:, estado_pesca])
        decisoes.append((caminho == estado_pesca).astype(np.int8))
    return probabilidades, decisoes


def _resumo_replica(replica: dict) -> dict:
    """Resumo de uma réplica (parâmetros aprendidos e esforço) para o JSON."""
    if "falhou" in replica:
        return {"replica": replica["replica"], "falhou": replica["falhou"]}
    parametros = replica["parametros"]
    resumo = {
        "replica": replica["replica"],
        "iteracoes": replica["iteracoes"],
        "convergiu": replica["convergiu"],
        "tempo_s": replica["tempo_s"],
        "estado_pesca": replica["estado_pesca"],
        "fracao_pesca_por_estado": replica["fracao_pesca_por_estado"],
        "loglik_por_ponto_final": replica["historico_loglik_por_ponto"][-1],
        "pi": parametros["pi"].tolist(),
        "A": parametros["A"].tolist(),
        "mu": parametros["mu"].tolist(),
        "var": parametros["var"].tolist(),
    }
    for chave in ("f1_macro_val_viterbi", "f1_macro_val_limiar"):
        if chave in replica:
            resumo[chave] = replica[chave]
    return resumo


def correr_experiencia(experiencia: str, config: dict) -> dict:
    """Ajusta o HMM no treino, escolhe a réplica na validação e grava previsões e avaliação."""
    treino = carregar_dataset(experiencia, "treino", config["divisao"])
    validacao = carregar_dataset(experiencia, "validacao", config["divisao"])
    print(
        f"\nHMM [{experiencia}] | treino {len(treino)} trajetórias / {treino.n_pontos} pontos | "
        f"validação {len(validacao)} / {validacao.n_pontos} | "
        f"features {', '.join(config['features'])}"
    )
    replicas = ajustar_replicas(treino, config)
    y_val = np.concatenate([t.y for t in validacao.trajetorias]).astype(np.int64)
    limiar = float(config["limiar"])
    utilizaveis = [r for r in replicas if "falhou" not in r]
    for replica in utilizaveis:
        probabilidades, decisoes = prever(
            validacao, replica["parametros"], replica["estado_pesca"], config
        )
        prob = np.concatenate(probabilidades)
        replica["prob_val"] = probabilidades
        replica["decisoes_val"] = decisoes
        replica["f1_macro_val_viterbi"] = metricas_ponto(
            y_val, np.concatenate(decisoes).astype(np.int64)
        )["f1_macro"]
        replica["f1_macro_val_limiar"] = metricas_ponto(
            y_val, (prob >= limiar).astype(np.int64), prob
        )["f1_macro"]
        print(
            f"  réplica {replica['replica']}: {replica['iteracoes']} iterações "
            f"({'convergiu' if replica['convergiu'] else 'limite'}) | "
            f"{replica['tempo_s']:.1f} s | log-verosimilhança/ponto "
            f"{replica['historico_loglik_por_ponto'][-1]:.4f} | "
            f"F1 macro val: Viterbi {replica['f1_macro_val_viterbi']:.4f} / "
            f"limiar {replica['f1_macro_val_limiar']:.4f}"
        )
    melhor = max(utilizaveis, key=lambda r: r["f1_macro_val_viterbi"])

    previsoes = construir_previsoes(
        validacao, melhor["prob_val"], limiar, decisoes=melhor["decisoes_val"]
    )
    parquet = escrever_previsoes(previsoes, PASTA_PREVISOES / experiencia / "HMM_validacao.parquet")
    avaliacao = avaliar_previsoes(previsoes)
    avaliacao.update(
        {
            "modelo": "HMM",
            "experiencia": experiencia,
            "divisao": config["divisao"],
            "conjunto": "validacao",
            "limiar": limiar,
            "decisao": "viterbi",
            "features": config["features"],
            "replica_escolhida": melhor["replica"],
        }
    )
    imprimir_avaliacao(avaliacao, f"HMM [{experiencia}]")
    json_avaliacao = guardar_avaliacao(
        avaliacao, PASTA_RESULTADOS / experiencia / "HMM_validacao.json"
    )

    registo = {
        "experiencia": experiencia,
        "divisao": config["divisao"],
        "conjunto": "validacao",
        "features": config["features"],
        "semente": int(config["semente"]),
        "limiar": limiar,
        "n_repeticoes_em": int(config["n_repeticoes_em"]),
        "tolerancia": float(config["tolerancia"]),
        "var_minima": float(config["var_minima"]),
        "criterio_escolha": "F1 macro na validação com decisão por Viterbi",
        "replica_escolhida": melhor["replica"],
        "replicas": [_resumo_replica(r) for r in replicas],
    }
    destino = PASTA_RESULTADOS / experiencia / "hmm_parametros.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(registo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"  previsões: {parquet.relative_to(RAIZ)} | avaliação: "
        f"{json_avaliacao.relative_to(RAIZ)} | parâmetros: {destino.relative_to(RAIZ)}"
    )
    return registo


def verificar_lse() -> None:
    """O log-soma-exp de dois valores tem de coincidir com o caso trivial e com os infinitos."""
    assert _lse2(-math.inf, -math.inf) == -math.inf, "ambos -inf devia dar -inf"
    assert np.isclose(_lse2(0.0, 0.0), math.log(2.0)), _lse2(0.0, 0.0)
    assert np.isclose(_lse2(-1000.0, 0.0), 0.0), _lse2(-1000.0, 0.0)
    assert np.isclose(_lse2(1.0, -math.inf), 1.0), _lse2(1.0, -math.inf)


def verificar_viterbi() -> None:
    """Viterbi numa sequência cuja emissão separa os estados sem ambiguidade."""
    log_pi = [math.log(0.5), math.log(0.5)]
    log_a = [[math.log(0.9), math.log(0.1)], [math.log(0.1), math.log(0.9)]]
    obs = [[0.0, -50.0]] * 5 + [[-50.0, 0.0]] * 5
    caminho = viterbi(obs, log_pi, log_a)
    assert (caminho[:5] == 0).all() and (caminho[5:] == 1).all(), caminho.tolist()
    curto = viterbi([[0.0, -50.0]], log_pi, log_a)
    assert curto.tolist() == [0], curto.tolist()


def verificar_sintetico() -> None:
    """Recupera os estados de um HMM sintético conhecido e respeita a monotonia do EM."""
    rng = np.random.default_rng(7)
    total = 2000
    mu_real = np.array([[0.0, 0.0], [4.0, -3.0]])
    transicao_real = np.array([[0.97, 0.03], [0.03, 0.97]])
    estados = np.zeros(total, dtype=np.int64)
    for t in range(1, total):
        estados[t] = rng.choice(N_ESTADOS, p=transicao_real[estados[t - 1]])
    x = mu_real[estados] + rng.normal(0.0, 0.5, size=(total, 2))
    config = {"n_repeticoes_em": 15, "tolerancia": 1e-8, "var_minima": 1e-6}
    resultado = baum_welch([x], config, semente=1, replica=0)

    historico = resultado["historico_loglik_por_ponto"]
    assert np.all(np.diff(historico) > -1e-6), "log-verosimilhança desceu no EM"
    assert historico[-1] > historico[0], "o EM não melhorou a log-verosimilhança"

    gama, caminho = posteriores(x, resultado["parametros"])
    acerto = max(float(np.mean(caminho == estados)), float(np.mean(caminho != estados)))
    assert acerto > 0.95, f"acerto só {acerto:.3f} no HMM sintético"
    assert gama.shape == (total, N_ESTADOS), gama.shape
    assert np.all(gama >= 0.0) and np.all(gama <= 1.0), "posterior fora de [0, 1]"

    parametros = resultado["parametros"]
    assert np.allclose(parametros["A"].sum(axis=1), 1.0), parametros["A"]
    assert np.all(parametros["var"] > 0.0), parametros["var"]


def principal() -> None:
    """Ponto de entrada: corre os testes internos e o HMM nas experiências pedidas."""
    parser = argparse.ArgumentParser(description="Passo 6: HMM de 2 estados (euc_speed, signed_turn).")
    parser.add_argument("--config", default=str(CONFIG_POR_DEFEITO))
    parser.add_argument("--exp", nargs="+", default=None, help="experiências a correr")
    parser.add_argument("--testes", action="store_true", help="só correr os testes internos")
    argumentos = parser.parse_args()

    verificar_lse()
    verificar_viterbi()
    verificar_sintetico()
    print("Testes internos do HMM (log-soma-exp, Viterbi e EM sintético): ok")
    if argumentos.testes:
        return
    config = ler_config(Path(argumentos.config))
    assert config["divisao"] == "principal", "esta fase não toca na divisão por navio"
    for experiencia in argumentos.exp or config["experiencias"]:
        correr_experiencia(experiencia, config)
    print("\nHMM concluído. O conjunto de teste não foi usado.")


if __name__ == "__main__":
    principal()
