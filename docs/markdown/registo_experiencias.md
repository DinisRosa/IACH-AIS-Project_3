# Registo de experiências

Uma entrada por experiência (`EXP-00N`), com objetivo, configuração, métricas e conclusão. As
métricas por ponto e por segmento entram aqui quando os modelos forem treinados (passos 6 a 8 do
plano de fase 1); até lá fica registado o que já se pode medir.

Semente 42 em todas. A divisão de referência é a por id completo
(`data/dividir/conjuntos.json`); a divisão agrupada por navio
(`data/dividir/conjuntos_por_navio.json`) serve só para o teste de sensibilidade.

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
