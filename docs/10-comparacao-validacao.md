# Comparação entre Random Forest e CNN na validação

Avaliamos o primeiro Random Forest e o primeiro treino completo da CNN com o mesmo
programa oficial do LUNA16, nos mesmos 89 exames de validação, com os mesmos 75.063
candidatos V2 e as mesmas marcações excluídas. O teste não foi lido. Medido em
13/10/2026.

```bash
.venv/bin/dvc repro comparar_validacao diagnosticar_random_forest
```

O estágio `avaliar_random_forest` (`scripts/22_avaliar_random_forest.py`) reutiliza os
escores salvos pelo treino de 04/10/2026, sem treinar de novo, e confere que cada escore
corresponde ao mesmo candidato da lista V2 em identificador e coordenada. O run do MLflow
é `e3ed9e0c20ac4494aa6d97af2c779baa`, com saída em `modelos/random_forest_v2_validacao`.
A CNN é o run `4f52dc350af44d099574a26ac1567731` (`docs/09-cnn-mlflow.md`).

Antes de criar os estágios novos, registramos com `dvc commit -f registrar_preparacao
treinar_random_forest` a mudança em `src/experimentos/rastreamento.py` trazida pela
integração da PR #5. Ela só acrescenta arquivos à cópia de código guardada no MLflow e
não altera o treino; sem o registro, o DVC retreinaria o Random Forest.

## FROC oficial

O avaliador sorteia as reamostras por exame com a semente 42 sobre a mesma lista de
exames, então as 1.000 reamostras são as mesmas para os dois modelos. O intervalo da
diferença de CPM vem dessas reamostras pareadas (`scripts/23_comparar_validacao.py`,
saída em `relatorios/experimentos/comparacao_validacao/`).

| FP por exame | Random Forest | IC95% | CNN | IC95% |
|---|---|---|---|---|
| 0,125 | 0,000 | 0,000 a 0,000 | 0,054 | 0,000 a 0,124 |
| 0,25 | 0,000 | 0,000 a 0,000 | 0,090 | 0,039 a 0,233 |
| 0,5 | 0,000 | 0,000 a 0,000 | 0,225 | 0,127 a 0,306 |
| 1 | 0,000 | 0,000 a 0,000 | 0,270 | 0,175 a 0,367 |
| 2 | 0,000 | 0,000 a 0,000 | 0,333 | 0,235 a 0,452 |
| 4 | 0,000 | 0,000 a 0,000 | 0,405 | 0,306 a 0,525 |
| 8 | 0,018 | 0,000 a 0,049 | 0,495 | 0,386 a 0,611 |

| | Random Forest | CNN |
|---|---|---|
| CPM | 0,003 | 0,268 |
| IC95% do CPM | 0,000 a 0,007 | 0,198 a 0,358 |

A diferença de CPM, CNN menos Random Forest, foi 0,265, com IC95% pareado de 0,196 a
0,354. A CNN teve CPM maior nas 1.000 reamostras. A curva completa dos dois está em
`froc_validacao.png`.

O avaliador considera no máximo 100 candidatos por exame. Dentro desse limite, a CNN
alcança 90 dos 111 nódulos e o Random Forest, 23 (`CADAnalysis.txt` de cada avaliação).

## As hipóteses do professor para o Random Forest

O feedback da Sprint 3 listou quatro hipóteses para o Random Forest praticamente não
funcionar. Testamos cada uma (`scripts/24_diagnosticar_random_forest.py`, saída em
`relatorios/experimentos/diagnostico_random_forest/diagnostico.json`):

**Informações diferentes das da CNN.** Recalculamos as sete features a partir dos
recortes NPZ que a CNN lê, em 2.655 candidatos de três exames de validação. A maior
diferença contra a tabela do Random Forest foi da ordem de 1e-16, arredondamento de
ponto flutuante. Os dois modelos olham os mesmos cubos de 34³. Escala não se aplica,
porque árvore de decisão não depende de escala.

**Rótulo deslocado.** A T3 já tinha conferido as 604.563 linhas de treino e validação
contra a lista V2. O script 22 repetiu a conferência de identificador e coordenada nos
75.063 candidatos de validação.

**Coluna de probabilidade invertida.** O código escolhe a coluna em que
`classes_ == 1`. Nos dados, a mediana do escore dos positivos foi 0,104 contra 0,064 dos
negativos.

**Conjuntos diferentes de candidatos.** O script 23 recusa comparar se exames,
candidatos, semente ou número de reamostras divergirem. A comparação passou.

O teste rápido que o professor sugeriu foi medir a ordenação fora da FROC: a chance de um
candidato positivo sorteado ter escore maior que um negativo sorteado, em que 0,5 é
moeda jogada. É um diagnóstico, não métrica do projeto.

| | Random Forest | CNN |
|---|---|---|
| Separação no treino | 1,000 | não medido |
| Separação na validação | 0,593 | 0,901 |
| Positivos entre os 100 maiores escores do próprio exame | 25 de 120 | 99 de 120 |

Cada feature sozinha separou na validação entre 0,426 e 0,544. Nenhuma das quatro
hipóteses se confirmou. O Random Forest decorou o treino e ordena a validação pouco
acima do acaso; por inferência, as sete estatísticas de intensidade do cubo inteiro
carregam pouca informação para distinguir nódulo dos candidatos negativos da lista V2.
