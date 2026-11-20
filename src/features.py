"""Passo 4 da fase 1: normalização mediana/IQR ajustada só ao treino e features por conjunto.

O que faz:

- lê a lista de features de um ficheiro em ``configs/`` (variantes: com ``x, y``, sem ``x, y`` e
  com ``t_rel``); trocar de variante é trocar de ficheiro, sem mexer no código;
- lê a divisão guardada por ``dividir.py`` (``data/dividir/conjuntos.json``) e separa o Parquet
  nos três conjuntos, sem partir nenhuma trajetória;
- ajusta mediana e IQR **só com as linhas do treino** e guarda-as em
  ``data/features/<experiência>/normalizacao.json``;
- aplica essas estatísticas aos três conjuntos e escreve
  ``data/features/<experiência>/{treino,validacao,teste}.parquet`` com as features em
  ``float32`` já normalizadas. Os outliers não são cortados: só há translação e escala;
- com ``--divisao por_navio`` faz o mesmo para a divisão agrupada por navio, ajustando outra vez a
  normalização ao treino **dessa** divisão e escrevendo em
  ``data/features/<experiência>__por_navio/``. Sem isto, o teste de sensibilidade usaria
  estatísticas calculadas com pontos que nessa divisão estão no teste (fuga). A divisão principal
  não é tocada;
- verifica que, no treino normalizado, a mediana é 0 e o IQR é 1 em cada coluna, que os extremos
  são os do bruto transformado (nada cortado) e que perturbar validação e teste não altera as
  estatísticas guardadas — prova de que nenhuma delas foi calculada com esses conjuntos;
- registra os percentis das features de cauda (``time_gap`` e ``euc_speed``) antes e depois da
  normalização, em ``distribuicoes.csv``.

Executar a partir da raiz do repositório:

    python src/features.py                                          # EXP-001 (com x, y)
    python src/features.py configs/EXP-002_features_sem_xy.json     # sem x, y
    python src/features.py configs/EXP-003_features_com_t_rel.json  # ablação com t_rel
    python src/features.py --divisao por_navio                      # EXP-001, divisão por navio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dividir import CAMINHO_SAIDA as CAMINHO_CONJUNTOS
from dividir import CAMINHO_SAIDA_NAVIO
from dividir import carregar_conjuntos

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_PARQUET = RAIZ / "data" / "ais_123.parquet"
PASTA_CONFIGS = RAIZ / "configs"
PASTA_SAIDA = RAIZ / "data" / "features"
CONFIG_POR_DEFEITO = PASTA_CONFIGS / "EXP-001_features_com_xy.json"

# Divisões que se podem preparar: a de referência e a de sensibilidade agrupada por navio.
CONJUNTOS_POR_DIVISAO = {"principal": CAMINHO_CONJUNTOS, "por_navio": CAMINHO_SAIDA_NAVIO}

# Features que o Parquet tem e que podem entrar no modelo. O ``t_rel`` existe mas fica fora por
# defeito: o modelo poderia aprender "início da viagem = navegação" em vez de comportamento.
FEATURES_DISPONIVEIS = [
    "x",
    "y",
    "signed_turn",
    "time_gap",
    "euc_speed",
    "distanceToShore",
    "t_rel",
]

# Colunas que acompanham as features nos Parquet normalizados.
COLUNAS_AUXILIARES = ["id_navio", "n_traj", "t_rel", "label"]

# Features cujas caudas interessa vigiar: escalas muito assimétricas.
FEATURES_CAUDAS = ("time_gap", "euc_speed")
PERCENTIS = (0, 1, 5, 25, 50, 75, 95, 99, 100)

# Tolerância das verificações feitas sobre valores guardados em float32.
TOLERANCIA = 1e-4


def ler_config(caminho: Path) -> dict:
    """Lê e valida o ficheiro JSON com a lista de features da experiência."""
    assert caminho.exists(), f"configuração inexistente: {caminho}"
    config = json.loads(caminho.read_text(encoding="utf-8"))
    assert set(config) == {"nome", "descricao", "features"}, (
        f"campos inesperados em {caminho.name}: {sorted(config)}"
    )
    assert config["nome"] == caminho.stem, (
        f"o nome «{config['nome']}» não bate com o ficheiro «{caminho.stem}»"
    )
    features = config["features"]
    assert isinstance(features, list) and features, "lista de features vazia"
    assert len(features) == len(set(features)), f"features repetidas: {features}"
    desconhecidas = sorted(set(features) - set(FEATURES_DISPONIVEIS))
    assert not desconhecidas, f"features desconhecidas: {desconhecidas}"
    return config


def ajustar_normalizacao(
    df_treino: pd.DataFrame, features: list[str]
) -> dict[str, dict[str, float]]:
    """Calcula a mediana e o IQR de cada feature a partir das linhas que recebe (treino)."""
    assert len(df_treino) > 0, "não há linhas de treino para ajustar a normalização"
    valores = df_treino[features].astype("float64")
    mediana = valores.median()
    iqr = valores.quantile(0.75) - valores.quantile(0.25)
    constantes = [f for f in features if not iqr[f] > 0]
    assert not constantes, f"IQR nulo ou negativo em: {constantes}"
    return {f: {"mediana": float(mediana[f]), "iqr": float(iqr[f])} for f in features}


def aplicar_normalizacao(
    df: pd.DataFrame, normalizacao: dict[str, dict[str, float]], features: list[str]
) -> pd.DataFrame:
    """Aplica (valor - mediana) / IQR e devolve as features em float32, sem cortar extremos."""
    assert set(normalizacao) == set(features), "normalização e lista de features não coincidem"
    # O t_rel fica como coluna auxiliar, exceto quando é ele próprio uma feature (EXP-003).
    auxiliares = [c for c in COLUNAS_AUXILIARES if c not in features]
    saida = df[auxiliares].copy()
    for f in features:
        p = normalizacao[f]
        valores = (df[f].astype("float64") - p["mediana"]) / p["iqr"]
        saida[f] = valores.astype("float32")
    assert not saida.isna().any().any(), "nulos depois da normalização"
    return saida


def mapear_conjuntos(
    df: pd.DataFrame, conjuntos: dict[str, set[tuple[int, int]]]
) -> np.ndarray:
    """Devolve, por linha, o nome do conjunto a que pertence a sua trajetória."""
    mapa = {id_completo: nome for nome, lista in conjuntos.items() for id_completo in lista}
    chaves = zip(df["id_navio"].tolist(), df["n_traj"].tolist())
    nomes = np.array([mapa.get(chave, "") for chave in chaves], dtype=object)
    fora = int((nomes == "").sum())
    assert fora == 0, f"{fora} pontos pertencem a trajetórias fora da divisão"
    return nomes


def separar_conjuntos(
    df: pd.DataFrame, conjuntos: dict[str, set[tuple[int, int]]]
) -> dict[str, pd.DataFrame]:
    """Separa o Parquet nos três conjuntos, mantendo a ordem original das linhas."""
    nomes = mapear_conjuntos(df, conjuntos)
    por_conjunto = {nome: df[nomes == nome] for nome in conjuntos}
    total = sum(len(parte) for parte in por_conjunto.values())
    assert total == len(df), f"{total} pontos separados != {len(df)}"
    return por_conjunto


def verificar_normalizacao(
    df_bruto: pd.DataFrame,
    df_normalizado: pd.DataFrame,
    features: list[str],
    normalizacao: dict[str, dict[str, float]],
) -> None:
    """Confirma mediana ≈ 0 e IQR ≈ 1 no treino normalizado e que nada foi cortado."""
    valores = df_normalizado[features].astype("float64")
    erro_mediana = float(valores.median().abs().max())
    iqr = valores.quantile(0.75) - valores.quantile(0.25)
    erro_iqr = float((iqr - 1.0).abs().max())
    assert erro_mediana <= TOLERANCIA, f"mediana do treino normalizado: {erro_mediana}"
    assert erro_iqr <= TOLERANCIA, f"IQR do treino normalizado afastado de 1: {erro_iqr}"
    for f in features:
        p = normalizacao[f]
        bruto = df_bruto[f].astype("float64")
        normalizado = valores[f]
        for designacao, metodo in (("mínimo", "min"), ("máximo", "max")):
            esperado = (getattr(bruto, metodo)() - p["mediana"]) / p["iqr"]
            obtido = getattr(normalizado, metodo)()
            assert np.isclose(obtido, esperado, rtol=1e-4, atol=TOLERANCIA), (
                f"{designacao} de {f} não corresponde ao bruto transformado: "
                f"{obtido:.6f} != {esperado:.6f}"
            )


def verificar_sem_fuga(
    df: pd.DataFrame,
    conjuntos: dict[str, set[tuple[int, int]]],
    normalizacao: dict[str, dict[str, float]],
    features: list[str],
) -> None:
    """Perturba validação e teste e confirma que as estatísticas do treino não se alteram.

    Se o ajuste usasse linhas de validação ou de teste, multiplicar essas linhas por 1000 e
    somar 1e6 mudaria as estatísticas. Confirma-se também que as estatísticas do treino não
    coincidem com as de todos os dados, para o teste não passar por acaso.
    """
    nomes = mapear_conjuntos(df, conjuntos)
    perturbado = df[["id_navio", "n_traj", *features]].copy()
    perturbado[features] = perturbado[features].astype("float64")
    fora_do_treino = nomes != "treino"
    perturbado.loc[fora_do_treino, features] = (
        perturbado.loc[fora_do_treino, features] * 1000.0 + 1e6
    )
    treino_perturbado = separar_conjuntos(perturbado, conjuntos)["treino"]
    normalizacao_perturbada = ajustar_normalizacao(treino_perturbado, features)
    assert normalizacao_perturbada == normalizacao, (
        "as estatísticas mudaram com validação/teste perturbados: há fuga entre conjuntos"
    )
    normalizacao_global = ajustar_normalizacao(df, features)
    diferentes = [f for f in features if normalizacao_global[f] != normalizacao[f]]
    assert diferentes, "as estatísticas do treino coincidem com as de todos os dados"


def distribuicoes(
    brutos: dict[str, pd.DataFrame],
    normalizados: dict[str, pd.DataFrame],
    features: list[str],
) -> pd.DataFrame:
    """Percentis das features de cauda antes e depois da normalização, por conjunto."""
    quantis = [p / 100 for p in PERCENTIS]
    linhas = []
    for momento, dados in (("antes", brutos), ("depois", normalizados)):
        for conjunto, df in dados.items():
            for f in features:
                valores = df[f].astype("float64").quantile(quantis).to_numpy()
                linha = {"conjunto": conjunto, "momento": momento, "feature": f}
                linha.update({f"p{p}": float(q) for p, q in zip(PERCENTIS, valores)})
                linhas.append(linha)
    return pd.DataFrame(linhas)


def verificar_parquet_normalizado(caminho: Path, features: list[str]) -> None:
    """Relê um Parquet escrito e confirma colunas, dtypes, ausência de nulos, mediana e IQR."""
    df = pd.read_parquet(caminho)
    esperadas = [c for c in COLUNAS_AUXILIARES if c not in features] + features
    assert list(df.columns) == esperadas, f"colunas de {caminho.name}: {list(df.columns)}"
    assert df[features].dtypes.eq("float32").all(), f"features não estão em float32: {caminho}"
    assert not df.isna().any().any(), f"nulos em {caminho.name}"
    if caminho.stem == "treino":
        valores = df[features].astype("float64")
        mediana = float(valores.median().abs().max())
        iqr = valores.quantile(0.75) - valores.quantile(0.25)
        assert mediana <= TOLERANCIA, f"mediana != 0 no treino escrito: {mediana}"
        assert float((iqr - 1.0).abs().max()) <= TOLERANCIA, f"IQR != 1 no treino escrito"


def carregar_normalizacao(pasta: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Lê o JSON de uma experiência e devolve (features, estatísticas de normalização)."""
    registo = json.loads((Path(pasta) / "normalizacao.json").read_text(encoding="utf-8"))
    return registo["features"], registo["normalizacao"]


def preparar(
    caminho_config: Path | str = CONFIG_POR_DEFEITO,
    caminho_parquet: Path = CAMINHO_PARQUET,
    caminho_conjuntos: Path | None = None,
    pasta_saida: Path = PASTA_SAIDA,
    divisao: str = "principal",
) -> Path:
    """Executa o passo 4 para uma configuração e uma divisão e devolve a pasta dos artefactos."""
    assert divisao in CONJUNTOS_POR_DIVISAO, f"divisão desconhecida: {divisao}"
    config = ler_config(Path(caminho_config))
    if caminho_conjuntos is None:
        caminho_conjuntos = CONJUNTOS_POR_DIVISAO[divisao]
    nome_saida = config["nome"] if divisao == "principal" else f"{config['nome']}__{divisao}"
    features = config["features"]
    colunas = list(dict.fromkeys([*COLUNAS_AUXILIARES, *features]))
    df = pd.read_parquet(caminho_parquet, columns=colunas)
    assert not df.isna().any().any(), (
        f"nulos no Parquet em: {list(df.columns[df.isna().any()])}"
    )
    conjuntos = carregar_conjuntos(caminho_conjuntos)
    brutos = separar_conjuntos(df, conjuntos)
    normalizacao = ajustar_normalizacao(brutos["treino"], features)
    normalizados = {
        nome: aplicar_normalizacao(parte, normalizacao, features)
        for nome, parte in brutos.items()
    }
    verificar_normalizacao(brutos["treino"], normalizados["treino"], features, normalizacao)
    verificar_sem_fuga(df, conjuntos, normalizacao, features)

    destino = pasta_saida / nome_saida
    destino.mkdir(parents=True, exist_ok=True)
    for nome, parte in normalizados.items():
        caminho = destino / f"{nome}.parquet"
        parte.to_parquet(caminho, index=False)
        verificar_parquet_normalizado(caminho, features)

    registo = {
        "experiencia": nome_saida,
        "experiencia_base": config["nome"],
        "divisao": divisao,
        "descricao": config["descricao"],
        "features": features,
        "ajustado_em": "treino",
        "n_pontos_treino": int(len(brutos["treino"])),
        "normalizacao": normalizacao,
    }
    (destino / "normalizacao.json").write_text(
        json.dumps(registo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    caudas = [f for f in FEATURES_CAUDAS if f in features]
    assert caudas, "nenhuma feature de cauda na configuração"
    tabela = distribuicoes(brutos, normalizados, caudas)
    tabela.to_csv(destino / "distribuicoes.csv", index=False)

    imprimir_resumo(config, brutos, normalizacao, tabela, destino)
    return destino


def imprimir_resumo(
    config: dict,
    brutos: dict[str, pd.DataFrame],
    normalizacao: dict[str, dict[str, float]],
    tabela: pd.DataFrame,
    destino: Path,
) -> None:
    """Mostra a experiência, os pontos por conjunto, as estatísticas e as caudas."""
    features = config["features"]
    print(f"Experiência {destino.name}: {len(features)} features ({', '.join(features)})")
    pontos = " | ".join(f"{nome}: {len(brutos[nome])} pontos" for nome in brutos)
    print(f"  {pontos}")
    print("  mediana e IQR ajustados só ao treino desta divisão:")
    for f in features:
        p = normalizacao[f]
        print(f"    {f:16s} mediana {p['mediana']:14.4f} | IQR {p['iqr']:14.4f}")
    print("  percentis antes/depois da normalização (depois, em unidades de IQR):")
    print(tabela.to_string(index=False))
    try:
        referencia = destino.relative_to(RAIZ)
    except ValueError:
        referencia = destino
    print(f"Artefactos em {referencia}")


def principal() -> None:
    """Ponto de entrada: prepara as features de uma configuração e de uma divisão."""
    parser = argparse.ArgumentParser(
        description="Passo 4: features normalizadas (mediana/IQR do treino) por conjunto."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(CONFIG_POR_DEFEITO),
        help="ficheiro de configs/ com a lista de features",
    )
    parser.add_argument(
        "--divisao",
        choices=tuple(CONJUNTOS_POR_DIVISAO),
        default="principal",
        help="divisão a preparar (por omissão, a de referência)",
    )
    argumentos = parser.parse_args()
    preparar(argumentos.config, divisao=argumentos.divisao)


if __name__ == "__main__":
    principal()

