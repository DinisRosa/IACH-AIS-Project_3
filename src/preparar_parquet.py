"""Passo 1 da fase 1: converter o CSV original de trajetórias AIS num Parquet validado.

Lê ``data/123_fishing_trajs.csv`` sem o alterar, reduz as colunas ao que a fase 1 usa,
separa o id completo em ``id_navio`` e ``n_traj``, binariza o rótulo, ordena por
``(id_navio, n_traj, t)`` e calcula ``t_rel`` em segundos desde o primeiro ponto de cada
trajetória. O resultado é escrito em ``data/ais_123.parquet`` e validado por asserts.

Decisões aplicadas (ver docs/markdown/plano_fase1.md, secção 8):
- rótulo binário: ``02-fishing`` e ``04-fishing`` -> 1; ``01/03/05-sailing`` -> 0;
- ``t_rel = t - t[0]`` (o ``t`` é um timestamp estritamente crescente em cada trajetória);
  o ``t`` fica em ``datetime64[us]``, que é a resolução nativa do Parquet e chega para uma
  base com resolução de segundos;
- o ``time_gap`` do primeiro ponto mantém-se com o valor original, sem correções;
- ``longitude`` e ``latitude`` ficam em ``float64`` (fora das features, para os mapas da
  fase 2); as seis características contínuas ficam em ``float32``, o que nas coordenadas
  projetadas custa cerca de 0,5 m de resolução — aceitável para pontos separados por
  vários metros e irrelevante depois da normalização por mediana/IQR.

Executar a partir da raiz do repositório:

    python src/preparar_parquet.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_CSV = RAIZ / "data" / "123_fishing_trajs.csv"
CAMINHO_PARQUET = RAIZ / "data" / "ais_123.parquet"

# Valores verificados no CSV original. Se a base mudar, o script tem de falhar em vez
# de produzir derivados silenciosamente diferentes.
N_TRAJETORIAS_ESPERADAS = 123
N_NAVIOS_ESPERADOS = 84
N_PONTOS_ESPERADOS = 1_026_279
N_PONTOS_PESCA_ESPERADOS = 775_521

FORMATO_T = "%Y-%m-%d %H:%M:%S"

# Colunas lidas do CSV (as restantes — bearing, distance_gap — não são usadas na fase 1).
COLUNAS_CSV = [
    "id",
    "t",
    "longitude",
    "latitude",
    "x",
    "y",
    "signed_turn",
    "time_gap",
    "euc_speed",
    "distanceToShore",
    "label",
]

# Rótulo binário: pesca -> 1, navegação -> 0.
MAPA_ROTULOS = {
    "01-sailing": 0,
    "02-fishing": 1,
    "03-sailing": 0,
    "04-fishing": 1,
    "05-sailing": 0,
}

# Características contínuas usadas pelo modelo. As coordenadas x, y entram numa variante
# e ficam de fora noutra; essa escolha é feita em features.py.
FEATURES = ["x", "y", "signed_turn", "time_gap", "euc_speed", "distanceToShore"]

# Ordem final das colunas no Parquet: identificação, tempo, features, rótulo e extras.
ORDEM_COLUNAS = [
    "id_navio",
    "n_traj",
    "t",
    "t_rel",
    "x",
    "y",
    "signed_turn",
    "time_gap",
    "euc_speed",
    "distanceToShore",
    "label",
    "longitude",
    "latitude",
    "label_original",
]



def ler_csv(caminho: Path) -> pd.DataFrame:
    """Lê o CSV original só com as colunas usadas, mantendo id e rótulo como texto."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"CSV original não encontrado em {caminho}. "
            "Colocar o ficheiro do Kaggle em data/123_fishing_trajs.csv."
        )
    return pd.read_csv(
        caminho,
        usecols=COLUNAS_CSV,
        dtype={"id": "string", "label": "string"},
    )


def separar_id(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Separa o id completo ``<navio>-<n>`` em ``id_navio`` e ``n_traj`` inteiros."""
    partes = df["id"].str.split("-", n=1, expand=True)
    assert partes.shape[1] == 2, "o id não tem o formato <navio>-<n>"
    assert partes[0].notna().all() and partes[1].notna().all(), "há ids vazios ou sem '-'"
    # Reconstruir o id garante que a separação não perdeu nem inventou caracteres.
    assert (
        (partes[0] + "-" + partes[1]) == df["id"]
    ).all(), "a separação do id não é reversível"
    assert partes[0].str.fullmatch(r"\d+").all(), "id_navio não é numérico"
    assert partes[1].str.fullmatch(r"\d+").all(), "n_traj não é numérico"
    return partes[0].astype("int32"), partes[1].astype("int16")


def binarizar_rotulo(serie: pd.Series) -> pd.Series:
    """Converte as cinco classes originais no rótulo binário pesca (1) / navegação (0)."""
    desconhecidos = sorted(set(serie.unique()) - set(MAPA_ROTULOS))
    assert not desconhecidos, f"rótulos originais não previstos: {desconhecidos}"
    binario = serie.map(MAPA_ROTULOS)
    assert binario.notna().all(), "ficaram pontos sem rótulo binário"
    return binario.astype("int8")


def calcular_t_rel(df: pd.DataFrame) -> pd.Series:
    """Segundos desde o primeiro ponto de cada trajetória (``t - t[0]``)."""
    assert df["t"].notna().all(), "há timestamps nulos"
    primeiro = df.groupby(["id_navio", "n_traj"], sort=False)["t"].transform("min")
    segundos = (df["t"] - primeiro).dt.total_seconds()
    assert (segundos >= 0).all(), "t_rel negativo"
    assert (segundos % 1 == 0).all(), "t_rel com parte fracionária"
    assert segundos.max() < 2**31, "t_rel não cabe em int32"
    return segundos.astype("int32")


def validar(df: pd.DataFrame) -> None:
    """Confirma tudo o que o resto da fase 1 assume sobre o Parquet."""
    assert len(df) == N_PONTOS_ESPERADOS, f"pontos: {len(df)} != {N_PONTOS_ESPERADOS}"
    assert not df.isna().any().any(), (
        f"existem nulos em: {list(df.columns[df.isna().any()])}"
    )

    trajetorias = df.groupby(["id_navio", "n_traj"], sort=False, observed=True)
    assert trajetorias.ngroups == N_TRAJETORIAS_ESPERADAS, (
        f"trajetórias: {trajetorias.ngroups} != {N_TRAJETORIAS_ESPERADAS}"
    )
    assert df["id_navio"].nunique() == N_NAVIOS_ESPERADOS, (
        f"navios: {df['id_navio'].nunique()} != {N_NAVIOS_ESPERADOS}"
    )

    # Ordem e tempo: o modelo sequencial lê os pontos pela ordem do ficheiro. A chave é
    # o trio (id_navio, n_traj, t), por isso o `t` cresce dentro de cada trajetória e
    # recomeça na seguinte — não é monótono na coluna toda.
    chaves = df[["id_navio", "n_traj", "t"]]
    assert chaves.equals(
        chaves.sort_values(["id_navio", "n_traj", "t"], kind="stable").reset_index(drop=True)
    ), "o Parquet não está ordenado por (id_navio, n_traj, t)"
    assert not chaves.duplicated().any(), "há dois pontos com o mesmo (id, t)"
    assert (trajetorias["t"].diff().dropna() > pd.Timedelta(0)).all(), (
        "t não é estritamente crescente dentro de uma trajetória"
    )
    assert (df["t_rel"] >= 0).all(), "t_rel negativo"
    assert trajetorias["t_rel"].first().eq(0).all(), (
        "há trajetórias cujo t_rel não começa em 0"
    )
    assert (trajetorias["t_rel"].diff().dropna() > 0).all(), (
        "t_rel não cresce dentro da trajetória"
    )

    # Rótulo: binário e coerente com a etiqueta original de cinco classes.
    assert set(df["label"].unique()) <= {0, 1}, "label não é binário"
    assert int(df["label"].sum()) == N_PONTOS_PESCA_ESPERADOS, (
        f"pontos de pesca: {int(df['label'].sum())} != {N_PONTOS_PESCA_ESPERADOS}"
    )
    esperado = df["label_original"].astype("string").map(MAPA_ROTULOS)
    assert (esperado == df["label"]).all(), "label não corresponde a label_original"

    # Nenhum valor original foi alterado nem limitado.
    assert (df["time_gap"] >= 0).all(), "time_gap negativo"
    assert (df["time_gap"] < 86_400).all(), "time_gap de um dia ou mais"
    assert (df["euc_speed"] >= 0).all(), "euc_speed negativa"

    # Tipos finais, para que os derivados não cresçam sem necessidade.
    assert str(df["id_navio"].dtype) == "int32"
    assert str(df["n_traj"].dtype) == "int16"
    assert str(df["t_rel"].dtype) == "int32"
    assert str(df["label"].dtype) == "int8"
    assert str(df["t"].dtype) == "datetime64[us]", "t não está em datetime64[us]"
    for coluna in FEATURES:
        assert str(df[coluna].dtype) == "float32", f"{coluna} não é float32"
    for coluna in ("longitude", "latitude"):
        assert str(df[coluna].dtype) == "float64", f"{coluna} não é float64"
    assert str(df["label_original"].dtype) == "category", "label_original não é categórico"
    assert sorted(df["label_original"].cat.categories) == sorted(MAPA_ROTULOS), (
        "label_original não tem as cinco classes originais"
    )


def imprimir_resumo(df: pd.DataFrame, caminho: Path) -> None:
    """Imprime o essencial do Parquet gerado, com números verificáveis."""
    trajetorias = df.groupby(["id_navio", "n_traj"], sort=False, observed=True)
    comprimentos = trajetorias.size()
    duracoes = trajetorias["t_rel"].max().astype("float64") / 3600.0
    pesca = int(df["label"].sum())
    total = len(df)
    print(
        f"Parquet escrito em {caminho.relative_to(RAIZ)} "
        f"({caminho.stat().st_size / 1e6:.1f} MB)"
    )
    print(
        f"trajetórias: {trajetorias.ngroups} | navios: {df['id_navio'].nunique()} "
        f"| pontos: {total}"
    )
    print(
        f"pesca: {pesca} ({100 * pesca / total:.2f} %) | "
        f"navegação: {total - pesca} ({100 * (total - pesca) / total:.2f} %)"
    )
    print(
        f"pontos por trajetória: mín {comprimentos.min()}, "
        f"mediana {int(comprimentos.median())}, máx {comprimentos.max()}"
    )
    print(
        f"duração por trajetória (h): mín {duracoes.min():.2f}, "
        f"mediana {duracoes.median():.2f}, máx {duracoes.max():.2f}"
    )
    print(f"período coberto: {df['t'].min()} a {df['t'].max()}")


def preparar(
    caminho_csv: Path = CAMINHO_CSV, caminho_parquet: Path = CAMINHO_PARQUET
) -> pd.DataFrame:
    """Executa o passo 1 completo: CSV -> DataFrame validado -> Parquet."""
    df = ler_csv(caminho_csv)

    # As categorias são guardadas com valores `str` (object) para que o Parquet devolva
    # exatamente o mesmo tipo de categoria que fica em memória.
    df["label_original"] = df["label"].astype("str").astype("category")
    df["label"] = binarizar_rotulo(df["label"])
    df["id_navio"], df["n_traj"] = separar_id(df)
    df["t"] = pd.to_datetime(df["t"], format=FORMATO_T, errors="raise").astype(
        "datetime64[us]"
    )

    df = df.sort_values(["id_navio", "n_traj", "t"], kind="stable").reset_index(drop=True)
    df["t_rel"] = calcular_t_rel(df)

    for coluna in FEATURES:
        df[coluna] = df[coluna].astype("float32")
    for coluna in ("longitude", "latitude"):
        df[coluna] = df[coluna].astype("float64")

    df = df[ORDEM_COLUNAS]
    validar(df)

    caminho_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(caminho_parquet, index=False, engine="pyarrow", compression="snappy")

    # Reler o ficheiro escrito garante que o Parquet devolve exatamente o que foi validado.
    relido = pd.read_parquet(caminho_parquet)
    assert list(relido.columns) == ORDEM_COLUNAS, "as colunas do Parquet não coincidem"
    assert len(relido) == len(df), "o Parquet relido tem outro número de linhas"
    assert relido.dtypes.equals(df.dtypes), "o Parquet relido tem outros tipos"
    assert relido[["id_navio", "n_traj", "t"]].equals(df[["id_navio", "n_traj", "t"]]), (
        "o Parquet relido tem outra ordem de pontos"
    )
    assert int(relido["label"].sum()) == N_PONTOS_PESCA_ESPERADOS

    imprimir_resumo(relido, caminho_parquet)
    return relido


if __name__ == "__main__":
    preparar()


