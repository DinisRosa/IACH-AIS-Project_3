"""Passo 5 da fase 1: dataset de trajetórias completas, collate com máscara e bucketing.

O que faz:

- lê os Parquet já normalizados por ``features.py`` (``data/features/<experiência>/``) e os ids de
  cada conjunto em ``data/dividir/`` (a divisão principal ou a agrupada por navio);
- cada item é **uma trajetória completa**, sem truncar nem janelar: ``x`` ``[L, F]``, ``y`` ``[L]``,
  o id ``(id_navio, n_traj)`` e os índices dos pontos no conjunto, para reconstruir as previsões
  ponto a ponto na avaliação;
- o ``collate`` preenche até ao máximo do batch e devolve a máscara booleana dos pontos reais, os
  comprimentos e os índices; o padding leva ``-1`` no ``y`` e nos índices, para que um uso
  indevido do padding falhe em vez de passar despercebido;
- o bucketing agrupa trajetórias de comprimento parecido com semente fixa, e o tamanho do batch
  vem de ``configs/dataset.json``;
- valida tudo o que assume: contiguidade e ordem temporal dos pontos, ids de cada conjunto,
  máscara coerente com os comprimentos, cobertura dos batches, e igualdade entre a loss mascarada
  de um batch com padding e a loss das mesmas trajetórias sem padding.

Executar a partir da raiz do repositório:

    python src/dataset.py                                # valida as duas divisões (EXP-001)
    python src/dataset.py --divisao principal            # só a divisão de referência
    python src/dataset.py --exp EXP-002_features_sem_xy  # outra variante de features
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dividir import carregar_conjuntos
from features import carregar_normalizacao

RAIZ = Path(__file__).resolve().parents[1]
PASTA_FEATURES = RAIZ / "data" / "features"
PASTA_DIVISOES = RAIZ / "data" / "dividir"
CONFIG_POR_DEFEITO = RAIZ / "configs" / "dataset.json"

# Nome da divisão -> ficheiro escrito por ``dividir.py``. As duas divisões têm a mesma interface.
DIVISOES = {
    "principal": "conjuntos.json",
    "por_navio": "conjuntos_por_navio.json",
}
NOMES_CONJUNTOS = ("treino", "validacao", "teste")

# Colunas que os Parquet normalizados têm de trazer além das features.
COLUNAS_OBRIGATORIAS = ("id_navio", "n_traj", "t_rel", "label")

# Valor do padding no ``y`` e nos índices: -1 nunca é uma classe nem uma posição válida.
VALOR_PADDING = -1

N_TRAJETORIAS_ESPERADAS = 123
N_PONTOS_ESPERADOS = 1_026_279

# Tolerância das comparações numéricas da loss (float32).
TOLERANCIA_LOSS = 1e-6


def ler_config(caminho: Path = CONFIG_POR_DEFEITO) -> dict:
    """Lê e valida o ficheiro com a experiência, a divisão e o tamanho do batch."""
    assert caminho.exists(), f"configuração inexistente: {caminho}"
    config = json.loads(caminho.read_text(encoding="utf-8"))
    esperados = {"nome", "descricao", "experiencia", "divisao", "tamanho_batch", "semente"}
    assert set(config) == esperados, f"campos inesperados em {caminho.name}: {sorted(config)}"
    assert config["divisao"] in DIVISOES, (
        f"divisão «{config['divisao']}» desconhecida; opções: {sorted(DIVISOES)}"
    )
    assert int(config["tamanho_batch"]) >= 1, "tamanho de batch tem de ser >= 1"
    assert int(config["semente"]) >= 0, "semente tem de ser >= 0"
    pasta = PASTA_FEATURES / config["experiencia"]
    assert pasta.is_dir(), f"não há features normalizadas em {pasta.relative_to(RAIZ)}"
    return config


@dataclass(frozen=True)
class Trajetoria:
    """Uma trajetória completa, com o id e os índices dos pontos no conjunto de origem."""

    id_navio: int
    n_traj: int
    x: np.ndarray  # [L, F] float32, já normalizado
    y: np.ndarray  # [L] int64 com 0 ou 1
    indices: np.ndarray  # [L] int64, posição ordinal do ponto no Parquet do conjunto

    @property
    def id_completo(self) -> tuple[int, int]:
        """Id completo da trajetória: (id_navio, n_traj)."""
        return (self.id_navio, self.n_traj)

    @property
    def comprimento(self) -> int:
        """Número de pontos da trajetória."""
        return int(self.x.shape[0])


def carregar_trajetorias(caminho: Path, features: list[str]) -> list[Trajetoria]:
    """Lê um Parquet normalizado e devolve as trajetórias inteiras, pela ordem do ficheiro.

    Confirma que as linhas de cada trajetória são contíguas e que o ``t_rel`` cresce dentro de
    cada uma: sem isso, o modelo receberia os pontos fora de ordem sem aviso.
    """
    df = pd.read_parquet(caminho)
    faltam = [c for c in (*COLUNAS_OBRIGATORIAS, *features) if c not in df.columns]
    assert not faltam, f"colunas em falta em {caminho.name}: {faltam}"
    assert len(df) > 0, f"{caminho.name} está vazio"
    assert not df.isna().any().any(), (
        f"nulos em {caminho.name}: {list(df.columns[df.isna().any()])}"
    )
    etiquetas = set(int(v) for v in np.unique(df["label"]))
    assert etiquetas <= {0, 1}, f"etiquetas fora de {{0, 1}} em {caminho.name}: {etiquetas}"

    chaves = np.column_stack(
        [df["id_navio"].to_numpy(np.int64), df["n_traj"].to_numpy(np.int64)]
    )
    muda = np.r_[True, np.any(chaves[1:] != chaves[:-1], axis=1)]
    inicios = np.flatnonzero(muda)
    fins = np.r_[inicios[1:], len(df)]
    ids_distintos = {tuple(int(v) for v in chave) for chave in chaves}
    assert len(inicios) == len(ids_distintos), (
        f"as linhas de uma trajetória não são contíguas em {caminho.name}: "
        f"{len(inicios)} blocos para {len(ids_distintos)} trajetórias"
    )

    valores = np.ascontiguousarray(df[features].to_numpy(np.float32))
    rotulos = np.ascontiguousarray(df["label"].to_numpy(np.int64))
    tempos = df["t_rel"].to_numpy(np.int64)
    posicoes = np.arange(len(df), dtype=np.int64)

    trajetorias = []
    for inicio, fim in zip(inicios, fins):
        id_navio, n_traj = (int(v) for v in chaves[inicio])
        assert np.all(np.diff(tempos[inicio:fim]) > 0), (
            f"t_rel não é estritamente crescente em ({id_navio}, {n_traj}) de {caminho.name}"
        )
        trajetorias.append(
            Trajetoria(
                id_navio=id_navio,
                n_traj=n_traj,
                x=valores[inicio:fim],
                y=rotulos[inicio:fim],
                indices=posicoes[inicio:fim],
            )
        )
    assert sum(t.comprimento for t in trajetorias) == len(df), "pontos perdidos ao agrupar"
    return trajetorias


class DatasetTrajetorias(Dataset):
    """Dataset de trajetórias completas: um item por trajetória, sem truncar nem janelar."""

    def __init__(self, trajetorias: list[Trajetoria], features: list[str], nome: str = ""):
        assert trajetorias, "dataset sem trajetórias"
        self.trajetorias = trajetorias
        self.features = list(features)
        self.nome = nome
        larguras = {t.x.shape[1] for t in trajetorias}
        assert larguras == {len(self.features)}, (
            f"largura das features inconsistente: {sorted(larguras)} != {len(self.features)}"
        )
        assert len(self.ids) == len(set(self.ids)), "trajetória repetida no mesmo conjunto"

    def __len__(self) -> int:
        """Número de trajetórias."""
        return len(self.trajetorias)

    def __getitem__(self, indice: int) -> dict:
        """Trajetória ``indice`` como tensores, com o id e os índices dos pontos."""
        t = self.trajetorias[indice]
        return {
            "id": t.id_completo,
            "x": torch.from_numpy(t.x),
            "y": torch.from_numpy(t.y),
            "indices": torch.from_numpy(t.indices),
            "comprimento": t.comprimento,
        }

    @property
    def n_features(self) -> int:
        """Número de features por ponto."""
        return len(self.features)

    @property
    def comprimentos(self) -> np.ndarray:
        """Comprimento de cada trajetória, pela ordem dos itens."""
        return np.array([t.comprimento for t in self.trajetorias], dtype=np.int64)

    @property
    def ids(self) -> list[tuple[int, int]]:
        """Ids completos das trajetórias, pela ordem dos itens."""
        return [t.id_completo for t in self.trajetorias]

    @property
    def n_pontos(self) -> int:
        """Total de pontos do conjunto."""
        return int(self.comprimentos.sum())


def collate_trajetorias(itens: list[dict]) -> dict:
    """Junta trajetórias num batch com padding até ao máximo, máscara e comprimentos.

    O padding leva 0 nas features (que já estão normalizadas, logo 0 é a mediana), ``-1`` no ``y``
    e nos índices. A máscara booleana marca os pontos reais: a loss só olha para esses.
    """
    assert itens, "batch vazio"
    comprimentos = torch.tensor([item["comprimento"] for item in itens], dtype=torch.long)
    n_features = itens[0]["x"].shape[1]
    n_batch = len(itens)
    n_max = int(comprimentos.max())
    x = torch.zeros((n_batch, n_max, n_features), dtype=torch.float32)
    y = torch.full((n_batch, n_max), VALOR_PADDING, dtype=torch.long)
    indices = torch.full((n_batch, n_max), VALOR_PADDING, dtype=torch.long)
    mascara = torch.zeros((n_batch, n_max), dtype=torch.bool)
    for linha, item in enumerate(itens):
        assert item["x"].shape[1] == n_features, "features de largura diferente no mesmo batch"
        comprimento = item["comprimento"]
        x[linha, :comprimento] = item["x"]
        y[linha, :comprimento] = item["y"]
        indices[linha, :comprimento] = item["indices"]
        mascara[linha, :comprimento] = True
    assert int(mascara.sum()) == int(comprimentos.sum()), "máscara não bate com os comprimentos"
    return {
        "x": x,
        "y": y,
        "mascara": mascara,
        "comprimentos": comprimentos,
        "indices": indices,
        "ids": [item["id"] for item in itens],
    }


def construir_batches(
    comprimentos: np.ndarray, tamanho_batch: int, semente: int
) -> list[list[int]]:
    """Agrupa índices de trajetórias de comprimento parecido, com semente fixa.

    A ordem dos itens é sorteada, depois ordenada por comprimento (empates pela ordem sorteada) e
    cortada em blocos; a ordem dos blocos é sorteada outra vez, para o treino não ver sempre as
    trajetórias mais curtas primeiro. Assim o padding de cada batch é pequeno e a sequência é
    reprodutível.
    """
    comprimentos = np.asarray(comprimentos, dtype=np.int64)
    assert tamanho_batch >= 1, "tamanho de batch tem de ser >= 1"
    gerador = np.random.default_rng(semente)
    ordem = gerador.permutation(len(comprimentos))
    ordem = ordem[np.argsort(comprimentos[ordem], kind="stable")]
    blocos = [ordem[i : i + tamanho_batch] for i in range(0, len(ordem), tamanho_batch)]
    return [blocos[i].tolist() for i in gerador.permutation(len(blocos))]


def iterar_batches(
    dataset: DatasetTrajetorias, tamanho_batch: int, semente: int
) -> Iterator[dict]:
    """Gera os batches do conjunto, por ordem de comprimento e com a semente indicada."""
    for indices in construir_batches(dataset.comprimentos, tamanho_batch, semente):
        yield collate_trajetorias([dataset[i] for i in indices])


def pasta_da_experiencia(experiencia: str, nome_divisao: str) -> Path:
    """Pasta das features normalizadas de uma experiência numa divisão.

    A divisão de referência fica em ``data/features/<experiência>/``; a agrupada por navio fica em
    ``data/features/<experiência>__por_navio/``, porque tem normalização própria.
    """
    assert nome_divisao in DIVISOES, f"divisão desconhecida: {nome_divisao}"
    sufixo = "" if nome_divisao == "principal" else f"__{nome_divisao}"
    return PASTA_FEATURES / f"{experiencia}{sufixo}"


def carregar_dataset(
    experiencia: str,
    conjunto: str,
    nome_divisao: str,
    conjuntos: dict[str, set[tuple[int, int]]] | None = None,
) -> DatasetTrajetorias:
    """Carrega um conjunto (treino/validação/teste) de uma experiência e de uma divisão."""
    assert conjunto in NOMES_CONJUNTOS, f"conjunto desconhecido: {conjunto}"
    assert nome_divisao in DIVISOES, f"divisão desconhecida: {nome_divisao}"
    pasta = pasta_da_experiencia(experiencia, nome_divisao)
    features, _ = carregar_normalizacao(pasta)
    caminho = pasta / f"{conjunto}.parquet"
    assert caminho.exists(), (
        f"falta {caminho.relative_to(RAIZ)}; correr src/features.py "
        f"configs/{experiencia}.json --divisao {nome_divisao}"
    )
    if conjuntos is None:
        conjuntos = carregar_conjuntos(PASTA_DIVISOES / DIVISOES[nome_divisao])
    trajetorias = carregar_trajetorias(caminho, features)
    dataset = DatasetTrajetorias(trajetorias, features, nome=f"{experiencia}/{conjunto}")
    assert len(dataset) == len(conjuntos[conjunto]), (
        f"{len(dataset)} trajetórias em {conjunto} != {len(conjuntos[conjunto])} na divisão"
    )
    verificar_ids(dataset, conjuntos, conjunto)
    return dataset


def verificar_ids(
    dataset: DatasetTrajetorias,
    conjuntos: dict[str, set[tuple[int, int]]],
    conjunto: str,
) -> None:
    """Confirma que as trajetórias do dataset são exatamente as do conjunto certo."""
    assert set(dataset.ids) == conjuntos[conjunto], (
        f"ids de {dataset.nome} não coincidem com os de {conjunto} na divisão"
    )
    outros = [nome for nome in NOMES_CONJUNTOS if nome != conjunto]
    intrusos = sorted(set(dataset.ids) & set().union(*(conjuntos[n] for n in outros)))
    assert not intrusos, f"trajetórias de outros conjuntos em {dataset.nome}: {intrusos}"


def verificar_formas(dataset: DatasetTrajetorias) -> None:
    """Confirma as formas e os tipos de cada item e a contagem total de pontos."""
    for indice in range(len(dataset)):
        item = dataset[indice]
        comprimento = item["comprimento"]
        trajetoria = dataset.trajetorias[indice]
        assert item["id"] == trajetoria.id_completo, f"id trocado no item {indice}"
        assert item["x"].shape == (comprimento, dataset.n_features), (
            f"x com forma {tuple(item['x'].shape)} no item {indice}"
        )
        assert item["y"].shape == (comprimento,), f"y com forma {tuple(item['y'].shape)}"
        assert item["indices"].shape == (comprimento,), "índices com forma errada"
        assert item["x"].dtype == torch.float32, "x não está em float32"
        assert item["y"].dtype == torch.int64, "y não está em int64"
        assert item["indices"].dtype == torch.int64, "índices não estão em int64"
        assert set(item["y"].tolist()) <= {0, 1}, f"etiquetas inválidas no item {indice}"
        assert comprimento == trajetoria.comprimento == len(trajetoria.x)
        assert item["indices"][0] == trajetoria.indices[0]
        assert item["indices"][-1] == trajetoria.indices[-1]
    assert dataset.n_pontos == sum(dataset.comprimentos), "contagem de pontos inconsistente"


def verificar_ordem_pontos(dataset: DatasetTrajetorias, caminho_parquet: Path) -> None:
    """Confirma que cada item tem os pontos do ficheiro, pela ordem original e sem faltas."""
    df = pd.read_parquet(caminho_parquet)
    assert len(df) == dataset.n_pontos, (
        f"{len(df)} linhas no Parquet != {dataset.n_pontos} pontos no dataset"
    )
    valores = np.ascontiguousarray(df[dataset.features].to_numpy(np.float32))
    rotulos = df["label"].to_numpy(np.int64)
    for trajetoria in dataset.trajetorias:
        indices = trajetoria.indices
        assert np.array_equal(trajetoria.x, valores[indices]), (
            f"pontos fora de ordem em {trajetoria.id_completo}"
        )
        assert np.array_equal(trajetoria.y, rotulos[indices]), (
            f"etiquetas fora de ordem em {trajetoria.id_completo}"
        )
        assert np.all(np.diff(indices) > 0), f"índices não crescentes em {trajetoria.id_completo}"


def verificar_collate(
    dataset: DatasetTrajetorias, tamanho_batch: int, semente: int
) -> list[dict]:
    """Confirma formas, máscara, comprimentos e cobertura dos batches de um conjunto."""
    batches = list(iterar_batches(dataset, tamanho_batch, semente))
    vistos: list[tuple[int, int]] = []
    pontos = 0
    for batch in batches:
        n_batch = len(batch["ids"])
        n_max = int(batch["comprimentos"].max())
        assert 1 <= n_batch <= tamanho_batch, f"batch com {n_batch} trajetórias"
        assert batch["x"].shape == (n_batch, n_max, dataset.n_features), "forma de x no batch"
        assert batch["y"].shape == (n_batch, n_max), "forma de y no batch"
        assert batch["indices"].shape == (n_batch, n_max), "forma dos índices no batch"
        assert batch["mascara"].shape == (n_batch, n_max), "forma da máscara no batch"
        assert batch["mascara"].dtype == torch.bool, "máscara não é booleana"
        assert int(batch["x"].shape[1]) == int(batch["comprimentos"].max())
        for linha, comprimento in enumerate(batch["comprimentos"].tolist()):
            mascara = batch["mascara"][linha]
            assert int(mascara.sum()) == comprimento, "máscara não bate com o comprimento"
            assert bool(mascara[:comprimento].all()), "há padding dentro da trajetória"
            assert not bool(mascara[comprimento:].any()), "há máscara no padding"
            assert bool((batch["x"][linha, comprimento:] == 0).all()), "padding de x não é 0"
            assert bool((batch["y"][linha, comprimento:] == VALOR_PADDING).all()), (
                "padding de y não é -1"
            )
            assert bool((batch["indices"][linha, comprimento:] == VALOR_PADDING).all()), (
                "padding dos índices não é -1"
            )
            assert bool((batch["y"][linha, :comprimento] >= 0).all()), "y real com -1"
            indices = batch["indices"][linha, :comprimento]
            assert bool((indices[1:] > indices[:-1]).all()), "pontos do batch fora de ordem"
        vistos.extend(batch["ids"])
        pontos += int(batch["comprimentos"].sum())
    assert len(vistos) == len(dataset.ids), (
        f"{len(vistos)} trajetórias em batches != {len(dataset)} no conjunto"
    )
    assert set(vistos) == set(dataset.ids), "batches não cobrem as trajetórias do conjunto"
    assert pontos == dataset.n_pontos, f"{pontos} pontos em batches != {dataset.n_pontos}"

    repetidos = construir_batches(dataset.comprimentos, tamanho_batch, semente)
    assert repetidos == construir_batches(dataset.comprimentos, tamanho_batch, semente), (
        "o bucketing não é reprodutível com a mesma semente"
    )
    outra = construir_batches(dataset.comprimentos, tamanho_batch, semente + 1)
    assert outra != repetidos, "sementes diferentes deram o mesmo bucketing"
    return batches


def verificar_loss_padding(
    dataset: DatasetTrajetorias, tamanho_batch: int = 2, semente: int = 0
) -> dict[str, tuple[float, float]]:
    """Compara a loss mascarada de um batch com padding com a loss das trajetórias isoladas.

    Usa uma projeção linear (ponto a ponto, portanto insensível ao contexto) para que as duas
    contas tenham de dar o mesmo: a loss do batch é a média sobre os pontos reais e a loss sem
    padding é a média ponderada pelos comprimentos das trajetórias.
    """
    torch.manual_seed(0)
    modelo = torch.nn.Linear(dataset.n_features, 1)
    comprimentos = dataset.comprimentos
    curta, longa = int(np.argmin(comprimentos)), int(np.argmax(comprimentos))
    cenarios = {"mais curta + mais longa": collate_trajetorias([dataset[curta], dataset[longa]])}
    for batch in iterar_batches(dataset, tamanho_batch, semente):
        if len(set(batch["comprimentos"].tolist())) > 1:
            cenarios["primeiro batch com padding"] = batch
            break
    cenarios["batch só de uma trajetória"] = collate_trajetorias([dataset[longa]])

    resultados: dict[str, tuple[float, float]] = {}
    for nome, batch in cenarios.items():
        with torch.no_grad():
            logits = modelo(batch["x"]).squeeze(-1)
            alvo = batch["y"].to(torch.float32)
            mascara = batch["mascara"]
            loss_batch = float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[mascara], alvo[mascara]
                )
            )
            soma, total = 0.0, 0
            for linha, comprimento in enumerate(batch["comprimentos"].tolist()):
                loss_linha = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[linha, :comprimento], alvo[linha, :comprimento]
                )
                soma += float(loss_linha) * comprimento
                total += comprimento
            loss_sem_padding = soma / total
        assert abs(loss_batch - loss_sem_padding) <= TOLERANCIA_LOSS, (
            f"loss com padding ({loss_batch:.8f}) != loss sem padding "
            f"({loss_sem_padding:.8f}) em «{nome}»"
        )
        resultados[nome] = (loss_batch, loss_sem_padding)
    return resultados


def preparar(
    nome_divisao: str,
    experiencia: str,
    config: dict,
) -> dict:
    """Carrega e valida os três conjuntos de uma divisão e devolve um resumo."""
    conjuntos = carregar_conjuntos(PASTA_DIVISOES / DIVISOES[nome_divisao])
    tamanho_batch = int(config["tamanho_batch"])
    semente = int(config["semente"])
    dados: dict[str, DatasetTrajetorias] = {}
    resumo: dict[str, dict[str, float]] = {}
    for conjunto in NOMES_CONJUNTOS:
        dataset = carregar_dataset(experiencia, conjunto, nome_divisao, conjuntos)
        verificar_formas(dataset)
        verificar_ordem_pontos(
            dataset, pasta_da_experiencia(experiencia, nome_divisao) / f"{conjunto}.parquet"
        )
        batches = verificar_collate(dataset, tamanho_batch, semente)
        dados[conjunto] = dataset
        comprimentos = dataset.comprimentos
        pesca = sum(int(t.y.sum()) for t in dataset.trajetorias)
        resumo[conjunto] = {
            "trajetorias": len(dataset),
            "pontos": dataset.n_pontos,
            "minimo": int(comprimentos.min()),
            "mediana": float(np.median(comprimentos)),
            "maximo": int(comprimentos.max()),
            "batches": len(batches),
            "fracao_pesca": pesca / dataset.n_pontos,
        }

    total_trajetorias = sum(int(r["trajetorias"]) for r in resumo.values())
    total_pontos = sum(int(r["pontos"]) for r in resumo.values())
    assert total_trajetorias == N_TRAJETORIAS_ESPERADAS, (
        f"{total_trajetorias} trajetórias na divisão {nome_divisao}"
    )
    assert total_pontos == N_PONTOS_ESPERADOS, (
        f"{total_pontos} pontos na divisão {nome_divisao}"
    )

    if nome_divisao == "por_navio":
        navios = [{t.id_navio for t in dados[c].trajetorias} for c in NOMES_CONJUNTOS]
        for i, primeiro in enumerate(navios):
            for segundo in navios[i + 1 :]:
                assert not (primeiro & segundo), "há navios em dois conjuntos na divisão por navio"

    return {
        "divisao": nome_divisao,
        "experiencia": experiencia,
        "features": dados["treino"].features,
        "conjuntos": resumo,
        "loss": verificar_loss_padding(dados["treino"], tamanho_batch, semente),
    }


def imprimir_resumo(resumo: dict, config: dict) -> None:
    """Mostra o resumo de uma divisão: trajetórias, pontos, comprimentos e batches."""
    print(
        f"Divisão {resumo['divisao']} | {resumo['experiencia']} | "
        f"{len(resumo['features'])} features | batch {config['tamanho_batch']} "
        f"| semente {config['semente']}"
    )
    cabecalho = (
        f"  {'conjunto':10s} {'traj':>5s} {'pontos':>9s} {'min':>6s} "
        f"{'mediana':>8s} {'max':>7s} {'batches':>7s} {'pesca':>6s}"
    )
    print(cabecalho)
    for nome in NOMES_CONJUNTOS:
        r = resumo["conjuntos"][nome]
        print(
            f"  {nome:10s} {int(r['trajetorias']):5d} {int(r['pontos']):9d} "
            f"{int(r['minimo']):6d} {r['mediana']:8.1f} {int(r['maximo']):7d} "
            f"{int(r['batches']):7d} {100 * r['fracao_pesca']:5.1f}%"
        )
    for nome, (com_padding, sem_padding) in resumo["loss"].items():
        print(
            f"  loss mascarada «{nome}»: {com_padding:.8f} "
            f"(sem padding {sem_padding:.8f})"
        )


def principal() -> None:
    """Ponto de entrada: valida as divisões pedidas para uma experiência."""
    parser = argparse.ArgumentParser(
        description="Passo 5: dataset de trajetórias completas, collate e bucketing."
    )
    parser.add_argument(
        "--config", default=str(CONFIG_POR_DEFEITO), help="ficheiro de configs/ a usar"
    )
    parser.add_argument(
        "--exp", default=None, help="experiência de features (por omissão, a do config)"
    )
    parser.add_argument(
        "--divisao",
        choices=(*DIVISOES, "ambas"),
        default="ambas",
        help="divisão a validar (por omissão, as duas)",
    )
    argumentos = parser.parse_args()
    config = ler_config(Path(argumentos.config))
    experiencia = argumentos.exp or config["experiencia"]
    escolhidas = list(DIVISOES) if argumentos.divisao == "ambas" else [argumentos.divisao]
    for nome_divisao in escolhidas:
        imprimir_resumo(preparar(nome_divisao, experiencia, config), config)
    print("Dataset validado: formas, máscara, ordem dos pontos, ids e loss mascarada.")


if __name__ == "__main__":
    principal()
