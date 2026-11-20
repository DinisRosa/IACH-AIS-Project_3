# Plano — Fase 1: trajetória completa → previsão ponto a ponto

Projeto: segmentação de trajetórias AIS para deteção de pesca (IA Centrada no Humano, MIACD, Coimbra).
Estado: rascunho para validação. Os pontos marcados com **[A CONFIRMAR]** dependem de verificação nos dados ou de decisão do Dinis.

---

## 1. Objetivo da fase 1

Construir um modelo que recebe **uma trajetória completa** (identificada pelo id completo = id do navio + nº da trajetória) e devolve, **para cada ponto**, se o navio está a pescar ou a navegar.

Esta fase **não usa janelas, sobreposição nem pós-processamento**. O resultado serve de **baseline de referência**: na fase 2, o janelamento, a segmentação e o pós-processamento são avaliados contra os números desta fase, sobre os mesmos pontos.

## 2. Princípios

- **Unidade de trabalho:** o id completo. Uma trajetória nunca é partida nem dividida entre conjuntos.
- **Rótulo por ponto.** Classe positiva = pesca (`1`); navegação = `0`. Confirmado na base: existem apenas as cinco classes `01-sailing`, `02-fishing`, `03-sailing`, `04-fishing` e `05-sailing`.
- **Sem fuga de informação:** tudo o que é estimado a partir dos dados (normalização, limiares, hiperparâmetros) usa só o treino e a validação. O teste é usado uma única vez, no fim.
- **Reprodutibilidade:** semente fixa, parâmetros em ficheiros de configuração, derivados regeneráveis a partir do CSV.
- **Desequilíbrio natural mantido:** sem reamostragem nem pesos de classe (decisão da proposta). A pesca é cerca de 75% dos pontos, por isso a *accuracy* engana.

## 3. Dados de partida

Fonte: *Detailed Labelled Fishing Trajectories from AIS* (Kaggle, `thedevastator`).
Colunas originais conhecidas: `id, t, longitude, latitude, x, y, signed_turn, bearing, time_gap, distance_gap, euc_speed, distanceToShore, label`.
Colunas a manter: `id, t, label, x, y, signed_turn, time_gap, euc_speed, distanceToShore`.
Rótulos originais: `02-fishing`, `04-fishing` → 1; `01-sailing`, `03-sailing`, `05-sailing` → 0.
Valores esperados: 123 trajetórias, cerca de 1 026 279 pontos.

## 4. Passos

### Passo 1 — CSV → Parquet (`preparar_parquet.py`)

**Estado: concluído** (`src/preparar_parquet.py`, `data/ais_123.parquet`, 49,1 MB).

1. Ler só as colunas úteis.
2. Separar o id completo em `id_navio` e `n_traj`. Formato verificado e confirmado: `"211477000-2"` → `id_navio = 211477000`, `n_traj = 2` (inteiros); ordenar por `(id_navio, n_traj, t)`.
3. Binarizar o rótulo.
4. Ordenar por `(id, t)`. Um modelo sequencial lê os pontos pela ordem em que aparecem; sem ordenação, as features que dependem do ponto vizinho deixam de fazer sentido, e o erro é silencioso.
5. Criar `t_rel` (segundos desde o primeiro ponto da trajetória), guardado em bruto e fora das features. `t` é um timestamp estritamente crescente em todas as trajetórias, por isso `t_rel = t - t[0]` (**decidido**). Verificado: só 1 linha em 1 026 279 (em `219429000-2`) tem `Δt ≠ time_gap` (30 s vs 10 s). Consequência medida: nessa trajetória, os 8 614 pontos a partir de 2021-11-15 17:17:11 ficam com `t_rel` 20 s acima da soma dos `time_gap` (diferença constante até ao fim da trajetória); todas as outras 122 trajetórias coincidem exatamente. A proposta usava o `time_gap` acumulado; um desvio de 20 s em 8 614 pontos de 1 026 279 (0,84 %) fica documentado como limitação dos dados, sem correção.
6. Converter para tipos compactos (`float32`, `int8`, id como inteiro).
7. Validar com asserts:
   - 123 trajetórias e 1 026 279 pontos (nada perdido nem duplicado);
   - sem nulos;
   - `t` estritamente crescente dentro de cada trajetória (sem dois pontos no mesmo segundo);
   - `time_gap` do primeiro ponto de cada trajetória: no CSV atual varia entre 2 e 901 s (não existe o artefacto ≈ 7 860 do projeto antigo). **Decidido: manter o valor original, sem alterar.**
8. Colunas do Parquet (**decidido**): `id_navio, n_traj, t, t_rel, x, y, signed_turn, time_gap, euc_speed, distanceToShore, label`, mais três colunas fora das features: `longitude` e `latitude` (em `float64`, para os mapas da fase 2) e `label_original` (a etiqueta de cinco classes, para auditoria).

### Passo 2 — Análise de comprimentos

Distribuição do nº de pontos por trajetória (mín., mediana, máx.) e da duração. Serve para decidir se um modelo recorrente é viável em CPU ou se é preciso uma CNN dilatada. Já verificado: mín. 901, mediana 5 610 e máx. 45 251 pontos; durações de 5,99 h, 15,76 h e 85,10 h; 84 navios para 123 trajetórias; sem GPU. Uma BiGRU em sequências de até 45 mil pontos em CPU é lenta: medir cedo o tempo por época.

### Passo 3 — Divisão por id completo (`dividir.py`)

**Estado: concluído** (`src/dividir.py`, `data/dividir/conjuntos.json`).

- 70/15/15, semente 42 → **86 / 19 / 18** trajetórias (treino / validação / teste). A proposta indicava exatamente 86/19/18 e foi confirmada: a divisão reproduz os seus conjuntos ao ponto, com 706 522 / 173 532 / 146 225 pontos e 75,6 % / 77,1 % / 73,5 % de pesca. Não é preciso corrigir o texto da proposta.
- Esquema que reproduz a proposta (único entre os testados): `numpy.random.default_rng(42).permutation` sobre os ids ordenados por `(id_navio, n_traj)`, com cortes `[:86]` treino, `[86:104]` teste e `[104:]` validação. `random.shuffle`, `numpy.random.shuffle` e `rng.choice` davam outros conjuntos (por exemplo, validação com 83,1 % de pesca).
- Verificado: nenhum id em dois conjuntos, os 123 ids cobertos, cada conjunto inteiro (nenhuma trajetória partida) e a soma dos pontos igual a 1 026 279. O `dividir.py` falha se a divisão deixar de reproduzir os números da proposta.
- **Fuga entre navios:** 65 / 17 / 18 navios nos três conjuntos, mas só 84 distintos — **16 navios aparecem em mais de um conjunto** (9 em treino e teste ao mesmo tempo). A divisão é por id completo, como decidido; isto fica registado como limitação e justifica o teste de sensibilidade agrupado por navio (por fazer).
- Com só 18 trajetórias no teste, os resultados são ruidosos: prever validação cruzada por trajetória no fim.

### Passo 4 — Normalização e features (`features.py`)

- Mediana/IQR por coluna, ajustados só ao treino, guardados em JSON e aplicados aos três conjuntos.
- `time_gap` mantém-se como feature. O `t_rel` fica de fora das features por defeito (risco de o modelo aprender "início da viagem = navegação"); pode testar-se depois como ablação. **[A CONFIRMAR]**
- Duas versões das features: **com** `x, y` e **sem** `x, y` (evita decorar zonas geográficas).

### Passo 5 — Dataset (`dataset.py`)

- Cada item é uma trajetória completa: `X [L, F]` e `y [L]`.
- `collate`: padding até ao máximo do batch, devolve máscara e comprimentos.
- Loss ignora o padding. Batches pequenos, agrupados por comprimento (bucketing).
- Sem truncar trajetórias.

### Passo 6 — Baselines (`modelos.py`)

| Nome | Modelo | Nota |
| --- | --- | --- |
| B0 | Prever sempre pesca | referência trivial |
| B1 | LogReg, HistGradientBoosting | veem só as features do próprio ponto |
| HMM | HMM sobre `euc_speed` e `signed_turn` (Peel e Good, 2011) | prometido na proposta; trabalha sobre a trajetória inteira. **Decidido: entra na fase 1** |
| Wu et al. | Reimplementação ou aproximação de Wu et al. (2022) | prometido na proposta. **Decidido: fica registado para depois da fase 1** |
| Principal | BiGRU sobre a trajetória inteira, saída por ponto | alternativa: 1D-CNN com convoluções dilatadas |

O antigo B2 (baselines com médias/desvios móveis e ponto anterior) foi retirado: em contexto de trajetória completa, o modelo sequencial cobre esse papel e a fase 1 mede-se contra B0, B1, HMM e o principal.

Risco: o PyTorch está em CPU. Se o passo 2 mostrar sequências muito longas, a CNN dilatada passa a modelo principal.

### Passo 7 — Treino (`treinar.py`)

- BCE mascarada, sem pesos de classe.
- Early stopping pelo F1 na validação.
- 3 a 5 sementes por modelo.

### Passo 8 — Avaliação (`avaliar.py`)

- **Por ponto:** precisão, sensibilidade, F1 e IoU das duas classes, F1 macro, PR-AUC.
- **Por segmento:** nº de trocas pesca↔navegação previstas vs reais (fragmentação), erro de fronteira e IoU por segmento (prometido na proposta).
- Resultados por trajetória (média ± desvio), além do agregado.
- Validação para escolher; teste uma única vez.

### Passo 9 — Saída da fase 1

- Tabela com todos os modelos e métricas.
- Parquet de previsões por ponto (`id, índice, y_real, y_prev, prob`), para a fase 2 comparar exatamente sobre os mesmos pontos.
- Parâmetros de normalização guardados junto do modelo final.

## 5. Estrutura sugerida do repositório

```
src/
  preparar_parquet.py   dividir.py   features.py
  dataset.py   modelos.py   treinar.py   avaliar.py
configs/     (um ficheiro por experiência)
data/        (ignorado pelo Git)
docs/        (este plano e notas)
```

## 6. Fora de âmbito nesta fase

Janelamento, sobreposição, corte em gaps, pós-processamento temporal, ablações de janelas, SHAP, confiança calibrada e dashboard. Tudo isto fica para a fase 2 e seguintes, medido contra a tabela do passo 9.

## 7. Critérios de conclusão da fase 1

- Parquet validado por asserts.
- Divisão verificada (sem fuga entre conjuntos).
- Baselines B0, B1, HMM e modelo sequencial treinados e avaliados.
- Tabela de resultados e parquet de previsões guardados.
- Decisões em aberto deste documento resolvidas e registadas.

## 8. Registo de decisões (validadas pelo Dinis)

| # | Decisão | Consequência |
| --- | --- | --- |
| D1 | Rótulo binário: `02-fishing`, `04-fishing` → 1; `01-sailing`, `03-sailing`, `05-sailing` → 0 | 775 521 pontos de pesca (75,57 %) |
| D2 | `id` no formato `MMSI-n`; `id_navio` e `n_traj` inteiros; ordem `(id_navio, n_traj, t)` | 123 trajetórias, 84 navios |
| D3 | `t_rel = t - t[0]` | 1 ponto em `219429000-2` fica com 20 s de diferença face ao `time_gap` acumulado; documentado |
| D4 | `time_gap` do primeiro ponto mantém-se original (2 a 901 s) | nenhum valor alterado na base |
| D5 | HMM (Peel e Good, 2011) entra na fase 1 | novo baseline a implementar em `modelos.py` |
| D6 | Wu et al. (2022) fica registado para depois da fase 1 | não bloqueia a fase 1 |
| D7 | B2 (baselines com contexto) retirado do plano | tabela de resultados com B0, B1, HMM e principal |
| D8 | IoU acrescentado à avaliação, por ponto e por segmento | cumpre o prometido na proposta |
| D9 | Parquet com 14 colunas: 11 do plano + `longitude`, `latitude` e `label_original` | permite mapas e auditoria na fase 2 |
| D10 | `requirements.txt` começa com `numpy`, `pandas` e `pyarrow`; o resto entra quando for preciso | ambiente mínimo e reprodutível |
