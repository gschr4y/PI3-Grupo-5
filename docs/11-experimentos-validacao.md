# Experimentos na validação, 13/10/2026

Depois da primeira comparação (`docs/10-comparacao-validacao.md`), testamos mudanças na CNN e
no Random Forest usando só treino e validação. O teste não foi lido.

## Regras fixadas antes de rodar

Uma mudança por vez na CNN, cada variante partindo da melhor anterior. A mudança fica se o CPM
pontual da validação subir. No máximo cinco variantes de CNN e uma de Random Forest, para não
ajustar demais aos 89 exames de validação. Para todas, registramos a diferença pareada de CPM
com IC95%, calculada nas mesmas 1.000 reamostras por exame do avaliador oficial.

Cada variante é um nome em `experimentos_cnn` no YAML, que sobrescreve chaves de `cnn`, e um
estágio do DVC:

```bash
.venv/bin/dvc repro treinar_cnn_variante@<nome>
.venv/bin/dvc repro treinar_random_forest_centro
.venv/bin/dvc repro comparar_validacao
```

## O que mudou no código

`src/experimentos/cnn.py` ganhou aumento de dados opcional: espelhamento ao acaso em cada um dos
três eixos e rotação do lote em múltiplos de 90 graus no plano axial. Os recortes têm 1 mm nos três
eixos, então a rotação não distorce. Num cubo de 32 voxels o centro fica entre os índices 15 e 16,
e o espelhamento troca um pelo outro; o nódulo se desloca no máximo 1 voxel. Com a chave ausente
ou falsa, o treino é idêntico ao anterior: um teste treina duas épocas em dados sintéticos e exige
os mesmos pesos (`testes/test_cnn.py`).

`scripts/18_treinar_cnn.py` aceita `--variante` e recusa variante inexistente ou chave
desconhecida. O run do MLflow registra o nome da variante.

O Random Forest centro (`scripts/25_treinar_random_forest_centro.py`) calcula as mesmas sete
estatísticas do Random Forest v2 em três cubos concêntricos do recorte, de 8, 16 e 34 voxels, 21
features no total, com os mesmos hiperparâmetros. As features saem dos recortes NPZ, depois de
conferir identificador e classe de cada candidato contra a lista V2. Num teste com dois exames,
a média do cubo de 34 bateu com a feature do Random Forest v2 com diferença da ordem de 1e-16.

`scripts/23_comparar_validacao.py` passou a comparar a lista de modelos declarada em
`comparacao` no YAML, contra a referência `cnn_v2` e nos pares declarados.

Como `cnn.py`, o script 18 e `src/detection/features.py` são dependências de estágios já
executados, registramos esses estágios com `dvc commit -f` em vez de reexecutá-los: `treinar_cnn`,
`features@treino`, `features@validacao`, `registrar_preparacao`, `treinar_random_forest` e
`diagnosticar_random_forest`. Nos três arquivos, o que já existia não mudou de comportamento,
e os testes antigos continuaram passando.

## Resultados

Medidos pelo avaliador oficial nos 89 exames de validação
(`relatorios/experimentos/comparacao_validacao/resumo.csv` e `pares.csv`):

| Modelo | Run MLflow | CPM | IC95% | Sens. 1 FP | Sens. 4 FP |
|---|---|---|---|---|---|
| CNN v2, 15 épocas | `4f52dc35` | 0,268 | 0,198 a 0,358 | 0,270 | 0,405 |
| CNN 40 épocas | `a90b6d29` | 0,523 | 0,439 a 0,605 | 0,550 | 0,730 |
| + aumento de dados | `7b2c2f43` | 0,525 | 0,438 a 0,608 | 0,568 | 0,712 |
| + 20 negativos por positivo | `5529e76d` | 0,696 | 0,625 a 0,771 | 0,748 | 0,838 |
| + learning rate 3e-4 | `2f8b0a3c` | 0,752 | 0,681 a 0,819 | 0,784 | 0,883 |
| Random Forest v2 | `e3ed9e0c` | 0,003 | 0,000 a 0,007 | 0,000 | 0,000 |
| Random Forest centro | `5169e10d` | 0,409 | 0,323 a 0,505 | 0,405 | 0,559 |

Na variante com learning rate 3e-4, o IC95% da sensibilidade a 4 FP vai de 0,805 a 0,945, e a
1 FP, de 0,699 a 0,870 (`modelos/cnn_lr_3e-4/avaliacao_validacao/pontos_operacao.csv`). Na de 20
negativos, de 0,741 a 0,915 a 4 FP e de 0,650 a 0,835 a 1 FP.

Diferenças pareadas de CPM:

| Comparação | Diferença | IC95% | Reamostras a favor |
|---|---|---|---|
| 40 épocas contra 15 | 0,255 | 0,173 a 0,322 | 100% |
| + aumento contra 40 épocas | 0,003 | -0,054 a 0,055 | 52% |
| + 20 negativos contra + aumento | 0,171 | 0,107 a 0,256 | 100% |
| RF centro contra RF v2 | 0,407 | 0,322 a 0,501 | 100% |
| + learning rate 3e-4 contra + 20 negativos | 0,055 | 0,014 a 0,085 | 99,9% |
| CNN com learning rate 3e-4 contra RF centro | 0,342 | 0,252 a 0,437 | 100% |

Pela regra fixada, o aumento de dados entrou na cadeia porque o CPM pontual subiu, mas a
diferença não se distingue de zero.

O avaliador oficial considera os 100 maiores escores de cada exame. Quantos nódulos ficam dentro
desse corte depende da ordenação de cada modelo: 90 de 111 na CNN v2, 99 na CNN de 40 épocas, 99
com aumento, 103 com 20 negativos, 107 com learning rate 3e-4 e 92 no Random Forest centro (`CADAnalysis.txt` de cada pasta).

A loss de treino na época 40 foi 0,237 sem aumento, 0,340 com aumento, 0,136 com 20 negativos e
0,066 com learning rate 3e-4 (`historico.csv` de cada pasta). Os treinos com 20 negativos usaram
23.058 candidatos e levaram cerca de 1.300 segundos na GPU cada.

Usamos quatro das cinco variantes permitidas. A configuração com melhor CPM na validação é 40
épocas, aumento de dados, 20 negativos por positivo e learning rate 3e-4.
