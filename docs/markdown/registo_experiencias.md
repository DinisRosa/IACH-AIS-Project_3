# Registo de experiências

Uma entrada por experiência (`EXP-00N`), com objetivo, configuração, métricas e conclusão. As
métricas por ponto e por segmento entram aqui quando os modelos forem treinados (passos 6 a 8 do
plano de fase 1); até lá fica registado o que já se pode medir.

Semente 42 em todas. A divisão de referência é a por id completo
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
