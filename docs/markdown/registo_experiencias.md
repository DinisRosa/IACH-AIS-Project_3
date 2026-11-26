# Registo de experiências

Uma entrada por experiência (`EXP-00N`), com objetivo, configuração, métricas e conclusão. As
métricas por ponto e por segmento entram aqui quando os modelos forem treinados (passos 6 a 8 do
plano de fase 1); até lá fica registado o que já se pode medir.

Semente 42 nos baselines e no HMM; a BiGRU corre com três sementes (42, 43 e 44, D18) e reporta
média ± desvio. A divisão de referência é a por id completo
(`data/dividir/conjuntos.json`); a divisão agrupada por navio
(`data/dividir/conjuntos_por_navio.json`) serve só para o teste de sensibilidade.

Cada experiência existe nas duas divisões: `data/features/<EXP>/` (principal) e
`data/features/<EXP>__por_navio/` (por navio), esta última com normalização ajustada ao treino
**dessa** divisão (D15). Os números abaixo são da divisão principal; na por navio o treino tem
776 216 pontos e a mediana/IQR de `x` é 548 546,7 / 182 032,1 (vs 544 820,7 / 200 743,3).

## EXP-001 — features com `x, y` (referência)

- **Objetivo:** variante de referência do passo 4, com as coordenadas projetadas, para comparação
  com a variante sem coordenadas e com a ablação do `t_rel`.
- **Configuração:** `configs/EXP-001_features_com_xy.json` → `x, y, signed_turn, time_gap,
  euc_speed, distanceToShore`. Mediana/IQR ajustados ao treino (706 522 pontos) e guardados em
  `data/features/EXP-001_features_com_xy/normalizacao.json`.
- **Métricas:** ainda nenhuma de classificação (depende do treino). Verificado que no treino
  normalizado a mediana é 0 e o IQR é 1 (|IQR − 1| ≤ 3 × 10⁻⁸) e que nenhuma estatística usou
  validação ou teste.
- **Conclusão:** variante pronta a treinar; todos os modelos da fase 1 comparam-se sobre ela.

## EXP-002 — features sem `x, y`

- **Objetivo:** medir quanto da qualidade vem da posição geográfica (risco de decorar zonas de
  pesca) e não do comportamento do navio.
- **Configuração:** `configs/EXP-002_features_sem_xy.json` → `signed_turn, time_gap, euc_speed,
  distanceToShore`. Mesma normalização só ao treino.
- **Métricas:** aguarda treino; a comparar com a EXP-001 ponto a ponto (F1 macro, PR-AUC, IoU).
- **Conclusão:** pendente de treino.

## EXP-003 — features com `t_rel` (ablação)

- **Objetivo:** testar o risco de atalho temporal ("início da viagem = navegação") que motivou
  deixar o `t_rel` de fora por defeito.
- **Configuração:** `configs/EXP-003_features_com_t_rel.json` → as seis de EXP-001 mais `t_rel`
  (mediana 44 263 s, IQR 95 195 s no treino).
- **Métricas:** aguarda treino; se subir muito face à EXP-001, é sinal de atalho e não de sinal.
- **Conclusão:** pendente de treino.

## Baselines e HMM — validação (passo 6)

Conjunto de validação: 19 trajetórias, 173 532 pontos, 77,12 % de pesca. Todos os números vêm dos
JSON em `data/resultados/<EXP>/` (`B0_sempre_pesca`, `B1_logreg`, `B1_hgb`, `HMM`), limiar 0,5 e
sem qualquer pós-processamento temporal.

| Exp | Modelo | Precisão pesca | Sensib. pesca | F1 pesca | F1 navegação | F1 macro | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-001 | B0 sempre pesca | 0,771 | 1,000 | 0,871 | 0,000 | 0,435 | 0,771 |
| EXP-001 | B1 regressão logística | 0,958 | 0,924 | 0,941 | 0,814 | **0,877** | 0,975 |
| EXP-001 | B1 HistGradientBoosting | 0,982 | 0,851 | 0,912 | 0,774 | 0,843 | **0,988** |
| EXP-001 | HMM (`euc_speed`, `signed_turn`) | 0,986 | 0,663 | 0,793 | 0,624 | 0,708 | 0,964 |
| EXP-002 | B1 regressão logística | 0,939 | 0,935 | 0,937 | 0,789 | 0,863 | 0,967 |
| EXP-002 | B1 HistGradientBoosting | 0,976 | 0,918 | 0,946 | 0,840 | **0,893** | **0,991** |
| EXP-002 | HMM (idêntico à EXP-001) | 0,986 | 0,663 | 0,793 | 0,624 | 0,708 | 0,964 |

Leituras:

- **O baseline trivial não chega.** Prever sempre pesca dá F1 macro 0,435 (0 na navegação), muito
  abaixo de qualquer modelo aprendido; a *accuracy* de 77 % esconderia isto, como previsto na regra 6.
- **Sem `x, y` o gradiente boosting melhora** (F1 macro 0,843 → 0,893): a posição ajudava a regressão
  logística (0,877 vs 0,863) mas prejudicava o HGB. Nenhum dos dois colapsa, o que sugere que o sinal
  comportamental (`euc_speed`, `signed_turn`, `distanceToShore`, `time_gap`) transporta a maior parte
  da informação.
- **O HMM é conservador:** precisão 0,986 mas sensibilidade 0,663 — só marca 51,9 % dos pontos como
  pesca quando a realidade é 77,1 %. O Viterbi beneficia da persistência temporal (as fronteiras
  emparelhadas são 37/37, todas), mas a emissão gaussiana diagonal com duas features saturam. Fica
  como referência interpretável, não como candidato a melhor modelo.
- **O HMM dá exatamente o mesmo resultado nas duas experiências** porque só usa `euc_speed` e
  `signed_turn` (`features` no JSON); não é um erro de cópia, é o esperado.
- **Fragmentação:** B1 prevê 3 486 (EXP-001) e 4 934 (EXP-002) segmentos para 56 reais
  (`razao_fragmentacao` 62× e 88×), o que baixa o IoU por segmento apesar do bom desempenho por
  ponto. É a motivação natural para o pós-processamento temporal da fase 2; na fase 1 reporta-se
  tal e qual.

## BiGRU — validação (passo 7)

Config `configs/bigru.json`: projeção linear + GRU bidirecional (hidden 32, 1 camada, dropout 0),
Adam com lr 0,002, BCE mascarada, `grad_clip` 1,0, batch 1, até 8 épocas com paciência 3, limiar
0,5, 1 thread. Sem pesos de classe, sem reamostragem e sem pós-processamento. O estado final é o da
melhor época na validação (D18). Números de `data/resultados/<EXP>/bigru_semente<k>_treino.json` e
`BiGRU_semente<k>_validacao.json`.

| Exp | Semente | Precisão pesca | Sensib. pesca | F1 pesca | F1 navegação | F1 macro | PR-AUC | IoU pesca | IoU navegação |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-001 | 42 | 0,983 | 0,973 | 0,978 | 0,927 | 0,952 | 0,996 | 0,956 | 0,863 |
| EXP-001 | 43 | 0,983 | 0,952 | 0,967 | 0,897 | 0,932 | 0,995 | 0,936 | 0,812 |
| EXP-001 | 44 | 0,978 | 0,962 | 0,970 | 0,902 | 0,936 | 0,995 | 0,941 | 0,821 |
| EXP-001 | **média ± desvio** | 0,981 ± 0,003 | 0,962 ± 0,011 | 0,972 ± 0,006 | 0,908 ± 0,016 | **0,940 ± 0,011** | 0,995 ± 0,000 | 0,945 ± 0,011 | 0,832 ± 0,027 |
| EXP-002 | 42 | 0,983 | 0,956 | 0,969 | 0,903 | 0,936 | 0,994 | 0,941 | 0,823 |
| EXP-002 | 43 | 0,986 | 0,953 | 0,969 | 0,903 | 0,936 | 0,994 | 0,940 | 0,823 |
| EXP-002 | 44 | 0,979 | 0,952 | 0,965 | 0,890 | 0,927 | 0,994 | 0,933 | 0,801 |
| EXP-002 | **média ± desvio** | 0,983 ± 0,004 | 0,954 ± 0,002 | 0,968 ± 0,002 | 0,898 ± 0,008 | **0,933 ± 0,005** | 0,994 ± 0,000 | 0,938 ± 0,004 | 0,816 ± 0,013 |

Custo e épocas: 8/8 épocas em todas as seis corridas, 195 a 221 s por época (1 thread) e 26,8 a
30,1 min por semente; a melhor época foi a 6.ª, 7.ª ou 8.ª, sempre perto do fim, e o *early stopping*
com paciência 3 **nunca disparou**. Ou seja, quem limita é o orçamento de 8 épocas, não a paciência.

Leituras:

- **Supera os baselines na validação, com folga e com pouco desvio:** F1 macro 0,940 ± 0,011
  (EXP-001) e 0,933 ± 0,005 (EXP-002), contra 0,877 / 0,893 do melhor B1 e 0,708 do HMM. Ganha
  sobretudo na classe minoritária: o F1 da navegação passa de 0,774–0,840 (B1) para 0,908 ± 0,016,
  mantendo a precisão da pesca em ~0,98.
- **O contexto sequencial substitui a posição geográfica:** com `x, y` (EXP-001) o F1 macro é
  0,007 melhor do que sem eles, valor dentro de um desvio-padrão de 0,011. Não há sinal de que a
  posição seja o que sustenta o resultado — o mesmo que o B1 já sugeria, agora com mais margem.
- **Fragmentação muito menor do que no B1:** 128, 239 e 288 segmentos previstos para 56 reais
  (razão 2,3×, 4,3× e 5,1×) na EXP-001, contra 3 486 e 4 934 (62× e 88×) do B1; IoU por segmento
  0,879–0,922. É o efeito esperado de ver a trajetória inteira em vez de um ponto de cada vez.
- **As fronteiras são encontradas mas não são exatas:** 37/37 fronteiras reais emparelhadas nas seis
  corridas, mas apenas 24 % a 35 % dentro da tolerância, com a fronteira prevista em média a 46–78
  pontos da real. A BiGRU acerta *que* há uma troca; *onde* ela começa continua a ser o alvo natural
  do pós-processamento da fase 2 (fora do âmbito desta fase).
- **Há trajetórias difíceis:** por trajetória, o F1 macro fica em 0,945–0,957 em média, mas o mínimo
  desce a 0,685–0,823, ou seja, o agregado de 0,93–0,95 esconde casos individuais fracos. Fica
  registado para a análise de erros típicos.
- **A decisão em aberto é o orçamento de épocas:** como o *early stopping* não dispara e a melhor
  época é sempre uma das três últimas, é provável que 8 épocas cortem o treino a meio. Subir o
  máximo de épocas obriga a repetir as seis corridas (~3 h em CPU) e por isso ficou como decisão do
  Dinis, não como ajuste silencioso.

## BiGRU — orçamento de épocas (passo 7b)

Mesma configuração da secção anterior com duas alterações: máximo de **30 épocas** e **paciência 5**
(o resto igual: hidden 32, 1 camada, dropout 0, lr 0,002, BCE mascarada, `grad_clip` 1,0, batch 1,
limiar 0,5, 1 thread, sementes 42/43/44, divisão principal). Objetivo: fechar a decisão em aberto do
passo 7 — perceber se o F1 macro estabiliza antes das 30 épocas e se o *early stopping* chega a
disparar. Continua a ser validação apenas: o teste não é usado e não se escolhe variante de
referência.

Artefactos por corrida: `data/resultados/<EXP>/bigru_semente<k>_treino.json` (histórico completo),
`bigru_semente<k>_curvas.csv` (loss no treino e F1 macro na validação, época a época) e
`BiGRU_semente<k>_validacao.json`. As corridas de 8 épocas ficaram arquivadas em
`data/arquivo/epocas8/` e são o termo de comparação.

