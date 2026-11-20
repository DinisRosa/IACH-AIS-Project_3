# AGENTS.md — Segmentação de trajetórias AIS para deteção de pesca

Instruções para agentes e colaboradores. Língua do projeto: **português europeu** (código, comentários, documentação e commits). Termos técnicos consagrados podem ficar em inglês.

## 1. Contexto e objetivo

Projeto da unidade curricular *IA Centrada no Humano* (MIACD, Universidade de Coimbra). Classificar, ao longo de uma trajetória AIS, os pontos de **pesca** e de **navegação**, de forma robusta e interpretável, como apoio à decisão de inspetores humanos (*human-on-the-loop*): o sistema apoia, a decisão final é sempre humana.

Dados: *Detailed Labelled Fishing Trajectories from AIS* (Kaggle, `thedevastator`): 123 trajetórias, cerca de 1 026 279 pontos. A classe pesca é cerca de 75% dos pontos.

## 2. Fase atual: FASE 1

**Uma trajetória completa entra no modelo e sai uma previsão por ponto (pesca vs navegação).**

- Unidade de trabalho: o **id completo** (id do navio + nº da trajetória). Nunca partir uma trajetória.
- **Sem janelas, sem sobreposição, sem pós-processamento.** O resultado é a baseline de referência.
- Fora de âmbito (fase 2 em diante): janelamento, sobreposição, corte em gaps, pós-processamento temporal, SHAP, confiança calibrada, dashboard. **Não implementar sem pedido explícito.**

Plano detalhado: `docs/markdown/plano_fase1.md`. Os pontos marcados **[A CONFIRMAR]** nesse plano são decisões em aberto: perguntar ao Dinis em vez de assumir.

## 3. Regras que não se quebram

1. **Divisão por id completo**, 70/15/15, semente 42. Nenhum id em mais de um conjunto.
2. **Sem fuga de informação:** a normalização (mediana/IQR) é ajustada **só ao treino**; limiares e hiperparâmetros são escolhidos na validação. O **teste é usado uma única vez**, no fim.
3. **Não remover nem limitar outliers.**
4. **Sem reamostragem nem pesos de classe** (decisão da proposta).
5. **Ordenar por `(id, t)`** antes de qualquer cálculo que dependa da ordem.
6. **Avaliar sempre as duas classes** (precisão, sensibilidade, F1 e IoU de cada uma, F1 macro, PR-AUC). Nunca reportar só *accuracy*; comparar com o baseline trivial "prever sempre pesca".
7. **Reprodutibilidade:** semente fixa, parâmetros em `configs/`, derivados regeneráveis a partir do CSV.
8. Se o deep learning não superar um baseline simples, isso é um resultado válido e reporta-se tal e qual.

## 4. Estrutura do repositório

```
src/
  preparar_parquet.py   CSV -> Parquet (colunas úteis, label binário, t_rel)
  dividir.py            divisão por id completo
  features.py           normalização (só treino) e seleção de features
  dataset.py            Dataset/collate: trajetória completa, máscara, bucketing
  modelos.py            baselines e modelos sequenciais
  treinar.py            treino (loss mascarada, early stopping)
  avaliar.py            métricas por ponto e por segmento
configs/                um ficheiro por experiência
data/                   entrada e derivados (ignorado pelo Git)
docs/markdown/          plano da fase 1, notas, registo de experiências
docs/latex/             proposta entregue (LaTeX e PDF)
docs/bibliografia/      artigos em PDF consultados (fora do Git, ver .gitignore)
docs/FormattingGuidelines-IJCAI-ECAI-26/   template e guidelines obrigatórias para o PDF
AGENTS.md               este ficheiro
.gitignore              ignora data/ (incluindo o CSV), a venv IACH-AIS-venv-3, caches e modelos
requirements.txt
```

## 5. Ambiente e comandos

Python 3.14.4 (a instalada), PyTorch 2.15.0 em **CPU** (sem GPU). Executar a partir da raiz do repositório.

```
python3 -m venv IACH-AIS-venv-3 && source IACH-AIS-venv-3/bin/activate
pip install -r requirements.txt
python src/preparar_parquet.py     # CSV -> Parquet
python src/dividir.py              # treino/validação/teste (principal e agrupada por navio)
python src/features.py             # features e normalização (uma variante de configs/)
python src/features.py configs/EXP-002_features_sem_xy.json   # outra variante
```

**[A COMPLETAR]** comandos de treino e avaliação quando os scripts existirem.

## 6. Convenções de código

- Nomes de funções, variáveis e comentários em português; docstring curta em cada função pública.
- Sem lógica em notebooks: os notebooks só chamam funções de `src/`.
- Cada script deve poder correr isolado e escrever o que gera em `data/`.
- Validar com `assert` (ou verificações explícitas) tudo o que a seguir se assume: nº de trajetórias, nº de pontos, ausência de nulos, `t` monótono, ids sem sobreposição entre conjuntos.
- Não commitar `data/`, modelos treinados nem ambientes virtuais.

## 7. Documentação e registo

- Atualizar o estado das tarefas em `docs/markdown/plano_fase1.md` no mesmo commit que as conclui.
- Cada experiência tem um identificador (`EXP-001`, `EXP-002`, …) e uma linha no registo em `docs/` com: objetivo, configuração, métricas e conclusão.
- Decisões de desenho novas ficam escritas (o quê e porquê) no mesmo commit.

## 8. Commits

**Objetivo:** o Dinis tem de perceber o que foi feito **sem abrir o commit**. Por isso cada commit tem um título curto e um corpo bem explicado, dado com dois `-m`:

```
git commit -m "Título curto no imperativo" -m "Corpo explicado"
```

Regras:

- **Commits pequenos e atómicos:** uma ideia por commit.
- **Título:** até cerca de 72 caracteres, no imperativo, em português, a dizer o que muda (ex.: "Adicionar validação de ordem temporal ao Parquet").
- **Corpo**, por esta ordem (omitir o que não se aplica):
  1. **O quê:** o que foi feito, em linguagem clara.
  2. **Porquê:** o problema ou a decisão que levou a isto.
  3. **Como:** abordagem escolhida e alternativas descartadas, se relevante.
  4. **Impacto:** ficheiros ou artefactos novos/alterados, mudanças em dados ou resultados (com números, se houver).
  5. **Como foi validado:** comando executado ou verificação feita e o resultado.
  6. **Pendente / próximo passo:** o que ficou por fazer.
- Nunca usar mensagens vagas ("update", "fix", "alterações").
- Se o commit conclui uma tarefa do plano, dizê-lo no corpo.

Exemplo:

```
git commit -m "Validar ordem temporal e nulos no Parquet" -m "O quê: adicionados asserts em preparar_parquet.py que confirmam 123 trajetórias, ausência de nulos e t monótono dentro de cada id.
Porquê: um modelo sequencial lê os pontos pela ordem do ficheiro; sem esta garantia, o erro seria silencioso.
Impacto: data/ais_123.parquet é regenerado; nenhuma linha removida.
Validado: python src/preparar_parquet.py termina sem erros e imprime 123 trajetórias.
Pendente: decidir o tratamento do time_gap do primeiro ponto."
```

**[A CONFIRMAR]** esta secção reflete o objetivo descrito pelo Dinis; ajustar ao estilo exato dos commits dele quando houver exemplos reais.

## 9. Como os agentes devem trabalhar

- Perguntar quando há ambiguidade ou uma decisão em aberto; não inventar.
- Não correr treinos pesados nem apagar ficheiros em `data/` sem confirmação.
- Não alterar a divisão, a semente ou a política de outliers sem pedido explícito.
- Dizer sempre o que foi verificado e o que não foi.
- Manter este ficheiro curto: se uma regra passa a ser detalhe de uma tarefa, vai para `docs/`.
