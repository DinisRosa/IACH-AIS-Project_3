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
- **Fuga entre navios:** 65 / 17 / 18 navios nos três conjuntos, mas só 84 distintos — **16 navios aparecem em mais de um conjunto** (9 em treino e teste ao mesmo tempo). A divisão é por id completo, como decidido; isto fica registado como limitação e é medido pelo teste de sensibilidade agrupado por navio.
- **Divisão de sensibilidade, agrupada por navio** (`data/dividir/conjuntos_por_navio.json`): mesma semente e mesma regra de cortes, mas a unidade sorteada é o navio. Resultado: 59 / 12 / 13 navios e **80 / 21 / 22 trajetórias** (treino / validação / teste), com 776 216 / 114 749 / 135 314 pontos e 74,9 % / 80,0 % / 75,8 % de pesca. Verificado: nenhum navio em dois conjuntos, os 84 navios cobertos, as 123 trajetórias cobertas e a soma dos pontos igual a 1 026 279. A divisão principal não muda; esta serve só para medir a diferença.
- Com só 18 trajetórias no teste, os resultados são ruidosos: prever validação cruzada por trajetória no fim.

### Passo 4 — Normalização e features (`features.py`)

**Estado: concluído** (`src/features.py`, `configs/EXP-00{1,2,3}_*.json`, `data/features/`).

- A lista de features vem de um ficheiro em `configs/`, um por experiência; trocar de variante é passar outro ficheiro ao script, sem alterar código. Três variantes: **EXP-001** com `x, y` (referência: `x, y, signed_turn, time_gap, euc_speed, distanceToShore`), **EXP-002** sem `x, y` (`signed_turn, time_gap, euc_speed, distanceToShore`) e **EXP-003** com `t_rel` (as seis de EXP-001 mais `t_rel`).
- `time_gap` é feature. O `t_rel` fica fora por defeito (risco de o modelo aprender "início da viagem = navegação") e entra apenas na EXP-003, como ablação. **Decidido.**
- Normalização mediana/IQR por coluna, ajustada **só ao treino**, guardada em `data/features/<experiência>/normalizacao.json` e aplicada aos três conjuntos. As features ficam em `float32`; o `t_rel`, quando não é feature, fica como coluna auxiliar.
- Valores ajustados no treino (EXP-001), mediana / IQR: `x` 544 820,7 / 200 743,3; `y` 6 337 198,8 / 163 109,5; `signed_turn` 0,0 / 5,327; `time_gap` 10 s / 6 s; `euc_speed` 2,7223 / 2,3814; `distanceToShore` 20 505,2 / 31 691,8. Na EXP-003, `t_rel` 44 263 s / 95 195 s.
- **Outliers preservados**: não há corte, só translação e escala. Efeito das caudas no treino, em unidades de IQR já normalizadas: `time_gap` p99 = 5,17 e máximo = 7 860 (máximo bruto de 47 172 s ≈ 13,1 h, o maior gap do treino); `euc_speed` p99 = 2,97 e máximo = 19,2 (bruto 48,5). Como a mediana de `time_gap` é 10 s e o IQR 6 s, qualquer gap grande aparece com dezenas ou milhares de IQR: essa cauda é medida, não escondida.
- Testes no próprio script: no treino normalizado a mediana é exatamente 0 e |IQR − 1| ≤ 3 × 10⁻⁸ em cada coluna (tolerância 1e-4); mínimo e máximo normalizados coincidem com os do bruto transformado (prova de que nada foi cortado); e a prova de ausência de fuga entre conjuntos — perturbar validação e teste (×1000 + 1e6) não altera as estatísticas guardadas. O teste foi verificado ao contrário: ajustar as estatísticas a todos os dados faz a asserção falhar.
- Percentis de `time_gap` e `euc_speed` antes e depois, por conjunto, em `data/features/<experiência>/distribuicoes.csv`. Os três conjuntos ficam em `treino.parquet`, `validacao.parquet` e `teste.parquet` (4,8 + 19 + 5,6 MB na EXP-001).

### Passo 5 — Dataset (`dataset.py`)

**Estado: concluído** (`src/dataset.py`, `configs/dataset.json`; medição de custo em `src/modelos.py`).

- Cada item é uma trajetória completa, **sem truncar nem janelar**: `x [L, F]` (float32, já normalizado), `y [L]` (int64), o id `(id_navio, n_traj)` e os índices dos pontos no conjunto, para reconstruir as previsões ponto a ponto na avaliação.
- `collate`: padding até ao máximo do batch, com máscara booleana, comprimentos e índices. O padding leva 0 nas features (0 é a mediana da normalização), `-1` no `y` e nos índices, para que um uso indevido falhe em vez de passar; a loss mascarada só olha para os pontos reais.
- Bucketing por comprimento com semente fixa: ordem sorteada, ordenada por comprimento (empates pela ordem sorteada), cortada em blocos de `tamanho_batch` e ordem dos blocos sorteada outra vez. `configs/dataset.json`: EXP-001, divisão principal, `tamanho_batch = 2`, semente 42. Batches pequenos porque cada item tem milhares de pontos.
- A divisão principal lê `data/features/<EXP>/`; a agrupada por navio lê `data/features/<EXP>__por_navio/`, com normalização própria ajustada ao treino **dessa** divisão (D15), para o teste de sensibilidade não usar estatísticas de pontos que nele estão no teste.
- Testes no próprio script: formas e tipos de cada item; máscara a bater certo com os comprimentos (nenhum ponto real mascarado, nenhum padding com máscara); padding de `x`/`y`/índices com 0/−1/−1; cobertura exata das trajetórias e dos pontos pelos batches; bucketing reprodutível com a mesma semente e diferente com outra; ordem dos pontos preservada (comparação com o Parquet e `t_rel` estritamente crescente); ids iguais aos do conjunto certo e sem intrusos; e a loss mascarada de um batch com padding igual à média ponderada das losses das trajetórias isoladas (tolerância 10⁻⁶; na prática ≤ 1,4 × 10⁻⁷). As duas divisões passam os mesmos testes com a mesma interface.
- Números: principal 86 / 19 / 18 trajetórias, 706 522 / 173 532 / 146 225 pontos, comprimentos 901–45 251 (mediana 5 540,5 no treino), 43 / 10 / 9 batches; por navio 80 / 21 / 22 trajetórias, 776 216 / 114 749 / 135 314 pontos, 40 / 11 / 11 batches. Soma 123 trajetórias e 1 026 279 pontos nas duas.
- **Custo em CPU** (`python src/modelos.py`: BiGRU de brinquedo de 6 features → 64 unidades, 1 camada bidirecional, 50 497 parâmetros; forward+backward+passo de SGD com loss mascarada; 3 repetições; 1 thread):

| Cenário | Pontos | Tempo (mediana) | Memória adicional | Pico do processo |
| --- | --- | --- | --- | --- |
| batch mediano (2 trajetórias) | 11 081 | 0,87 s | +340 MB | 900 MB |
| trajetória mais longa | 45 251 | 8,57 s | +1 398 MB | 2 297 MB |
| duas mais longas (com padding) | 76 023 | 8,66 s | +291 MB | 2 588 MB |

  Com 10 threads os tempos pioram e ficam instáveis (14,55 s de mediana na mais longa, 10,0–19,6 s): a GRU é sequencial e o paralelismo acrescenta contenção. **Decisão (D16): 1 thread no torch.** Extrapolando linearmente em pontos, uma época de treino (706 522 pontos) fica em ≈ 2,2 min e 30 épocas em ≈ 1,1 h. A memória do pior caso (2,3–2,6 GB) cabe nos 14 GB da máquina, mas limita o batch das trajetórias muito longas a 2.
  **Conclusão: a BiGRU é viável em CPU** e mantém-se como modelo principal do passo 6; a CNN dilatada continua a ser a alternativa, não passa a principal. Números brutos em `data/benchmarks/bigru_cpu.json` (1 thread) e `bigru_cpu_10threads.json`.

### Passo 6 — Baselines (`modelos.py`)

| Nome | Modelo | Nota |
| --- | --- | --- |
| B0 | Prever sempre pesca | referência trivial |
| B1 | LogReg, HistGradientBoosting | veem só as features do próprio ponto |
| HMM | HMM sobre `euc_speed` e `signed_turn` (Peel e Good, 2011) | prometido na proposta; trabalha sobre a trajetória inteira. **Decidido: entra na fase 1** |
| Wu et al. | Reimplementação ou aproximação de Wu et al. (2022) | prometido na proposta. **Decidido: fica registado para depois da fase 1** |
| Principal | BiGRU sobre a trajetória inteira, saída por ponto | alternativa: 1D-CNN com convoluções dilatadas |

O antigo B2 (baselines com médias/desvios móveis e ponto anterior) foi retirado: em contexto de trajetória completa, o modelo sequencial cobre esse papel e a fase 1 mede-se contra B0, B1, HMM e o principal.

Estado: **concluído**. Implementado em `src/baselines.py` (B0 e B1, config `configs/baselines.json`) e `src/hmm.py` (HMM em log-espaço, EM próprio, init por quantis, decisão por Viterbi; config `configs/hmm.json`). Métricas de validação das duas experiências no `registo_experiencias.md`: B1 é o melhor baseline (F1 macro 0,877 na EXP-001 e 0,893 na EXP-002) e o HMM fica em 0,708 por ser conservador na sensibilidade (0,663).

Risco: o PyTorch está em CPU. Medido no passo 5: a BiGRU de brinquedo custa 8,57 s por forward+backward na trajetória mais longa e ≈ 2,2 min por época estimada, logo é viável (D16) e mantém-se principal; a CNN dilatada é a alternativa se o tempo real de treino não couber no orçamento.

### Passo 7 — Treino (`treinar.py`)

**Estado: concluído** com a ressalva do orçamento de épocas (ver abaixo). `src/modelos.py` (classe `BiGRU` com packing), `src/treinar.py` e `configs/bigru.json`; seis corridas (2 experiências × 3 sementes) na validação, resultados no `registo_experiencias.md`. O teste continua intocado.

- BCE mascarada, sem pesos de classe.
- Early stopping pelo F1 macro na validação; o estado final é o da melhor época, não o da última.
- 3 a 5 sementes por modelo: **decidido 3 sementes (42, 43, 44) por experiência**, 8 épocas no máximo e paciência 3 (D18), porque a época custa ~3 min em CPU.
- Custo medido: ~173 s por época na EXP-001 (1 thread, hidden 32, batch 1); a avaliação da validação por época passou a batch 1 para não pagar o packing (D17).
- **Custo real de ponta a ponta (semente 42):** 203,0 s/época na EXP-001 (27,7 min) e 201,3 s/época na EXP-002 (27,4 min), com 8 épocas corridas nas duas e a melhor época na 8.ª.
- **As 3 sementes confirmaram o problema:** 8/8 épocas em todas as seis corridas (195 a 221 s/época, 26,8 a 30,1 min por semente) e a melhor época foi a 6.ª, 7.ª ou 8.ª; a paciência 3 **nunca disparou**. O orçamento de 8 épocas é o que limita o treino, não o *early stopping*. **Decisão em aberto (Dinis):** subir o máximo de épocas obriga a repetir as seis corridas (~3 h em CPU a 1 thread), por isso não se mexeu. Resultado já obtido: F1 macro na validação **0,940 ± 0,011** (EXP-001) e **0,933 ± 0,005** (EXP-002), contra 0,877 / 0,893 do melhor B1 e 0,708 do HMM; detalhe no `registo_experiencias.md`.
- `--sanidade` (overfit de 5 trajetórias, espera-se F1 ≈ 1) e `--determinismo` (duas corridas iguais bit a bit) são testes do próprio script.
- **Verificações feitas (EXP-001, 1 thread):** `--sanidade` dá loss 0,623 → 0,017 e F1 macro 0,998 na própria amostra (5 trajetórias, 5 706 pontos, 124 s), logo o modelo memoriza quando deve; `--determinismo` dá losses idênticas (0,630039 / 0,595168 duas vezes) e diferença máxima nas probabilidades 0,0, logo o treino é reprodutível. `python src/hmm.py --testes` também passa (log-soma-exp, Viterbi e EM sintético).

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
  dataset.py   modelos.py   baselines.py   hmm.py   treinar.py   avaliar.py
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
| D11 | Divisão principal por id completo: **86 / 19 / 18 = treino / validação / teste** | confere com o texto da proposta |
| D12 | Divisão agrupada por navio entra já na fase 1, só como teste de sensibilidade | 59 / 12 / 13 navios → 80 / 21 / 22 trajetórias; a divisão principal não muda |
| D13 | `time_gap` é feature; `t_rel` fica de fora por defeito e só entra na ablação EXP-003 | variantes escolhidas em `configs/`, sem alterar código |
| D14 | PDFs de terceiros em `docs/bibliografia/` ficam fora do Git | `.gitignore` e nota no `AGENTS.md`; versão local mantém-se |
| D15 | A divisão por navio tem features e normalização próprias, em `data/features/<EXP>__por_navio/`, ajustadas ao treino dessa divisão | o teste de sensibilidade não usa estatísticas de pontos que nele estão no teste; a divisão principal não muda (verificado por md5, bit a bit) |
| D16 | BiGRU em CPU com 1 thread no torch e `tamanho_batch = 2` | medido no passo 5: 8,57 s por passo na trajetória mais longa (vs 14,55 s com 10 threads), pico de 2,3 GB, época ≈ 2,2 min; a CNN dilatada fica como alternativa |
| D17 | `pack_padded_sequence` só quando o batch tem mesmo padding; treino e avaliação com `tamanho_batch = 1` | medido: com packing 7,39 s por batch de 9 000 pontos (0,82 s/1000), sem padding 1,30 s (0,14 s/1000), ou seja ~6× mais caro em CPU; o modo determinístico não custa nada (1,30 vs 1,33). Sem janelas, batch de 1 não introduz padding e garante que a previsão não depende do batch. Época medida em EXP-001: 173 s |
| D18 | BiGRU com `epocas = 8`, `paciencia = 3` e **3 sementes** (42, 43, 44) por experiência | a época real custa ~3 min (706 522 pontos de treino + 173 532 de validação, 1 thread); 15 épocas × 5 sementes × 2 experiências não cabia no orçamento de tempo. O F1 macro da validação já é 0,867 na 1.ª época, logo 8 épocas chegam para a curva estabilizar. Três sementes ficam no intervalo "3 a 5" do plano e permitem média ± desvio |
