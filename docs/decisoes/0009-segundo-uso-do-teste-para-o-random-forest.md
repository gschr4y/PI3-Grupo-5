# 0009: segundo uso do teste, só para medir o Random Forest centro congelado

**Estado:** decidido antes de abrir o teste pela segunda vez; executado uma vez em 26/10/2026
**Data:** 26/10/2026

## Contexto

O teste, 176 exames de 176 pacientes com 223 nódulos, foi usado uma vez em 13/10/2026 para
medir a CNN `cnn_lr_3e-4`, congelada pela decisão 0008 (CPM 0,724, IC95% 0,670 a 0,778;
`docs/12-avaliacao-final-teste.md`). Até aqui, o Random Forest centro só tem resultado na
validação (CPM 0,409, IC95% 0,323 a 0,505), e a comparação entre os dois modelos com intervalo
existe só nos 89 exames de validação (`docs/13-dossie-modelagem.md`, seção 11).

Para comparar a rede com o modelo simples nos mesmos exames novos, precisamos medir o Random
Forest centro no teste. Isso é um segundo uso do mesmo conjunto.

## Decisão

**Usamos o teste uma segunda vez, uma única execução, só para avaliar o Random Forest centro já
congelado.** O texto abaixo vai para o relatório:

> O conjunto de teste foi utilizado duas vezes. A primeira avaliação, em 13/10/2026, mediu a rede
> convolucional escolhida, que já tem seu resultado final registrado e não será alterada. A
> segunda avaliação mede o Random Forest centro, cujo modelo foi treinado e congelado em
> 13/10/2026, antes da primeira abertura do teste, e cuja identidade (hash SHA-256 do arquivo) foi
> registrada antes desta segunda avaliação. A segunda avaliação tem objetivo exclusivamente
> comparativo: não haverá treinamento, ajuste de hiperparâmetros, escolha de features, seleção de
> limiar ou escolha entre versões do Random Forest com base no resultado do teste, e o resultado da
> rede não será revisto. Ambos os modelos foram avaliados com o mesmo programa oficial do LUNA16,
> os mesmos 176 exames, os mesmos candidatos e as mesmas 1.000 reamostras por exame. Registramos o
> reuso do teste como limitação metodológica: o resultado da comparação é descritivo e não serviu
> para escolher modelo.

## Identidade congelada, registrada antes da avaliação

- `modelos/random_forest_centro/modelo.joblib`, SHA-256
  `58ba90fde1c7f6bbd7073ce8e6b6e6a9c6232591bddc03f7c758e57f27ff9b57`, versionado no commit
  `e1b51a5`. Run de treino `5169e10d13874969804d14f29cca6ea6`, 13/10/2026.
- Hiperparâmetros fixos desde o treino: 200 árvores, profundidade 20, 2 amostras por folha,
  `max_features="sqrt"`, `class_weight="balanced"`, semente 42, 21 features em cubos de 8, 16 e
  34 voxels, scikit-learn 1.9.1.
- Mesma divisão e mesma lista V2 que o checkpoint da CNN gravou: `divisao.csv` SHA-256
  `45b22b5e1d5468417243b8f8de5dd94d864a35e28f34e4759a8bef238a10d300` e `candidates_V2.csv`
  SHA-256 `8c168d9113f97a3cfd609d18703c50a62067f8f61fe4833be1650a02baa43c82`.
- Avaliação: programa oficial do LUNA16 (`src/experimentos/avaliacao_oficial.py`), no máximo
  100 marcas por exame, 1.000 reamostras por exame, IC95%, semente de bootstrap 42, a mesma da
  avaliação da CNN no teste. Com a mesma semente e a mesma lista de exames, as reamostras são as
  mesmas nos dois modelos, o que permite a diferença pareada de CPM.

`scripts/26_avaliar_teste_random_forest.py` confere os três hashes e a versão do scikit-learn
antes de prever, recusa pasta de saída já existente e grava o run do MLflow com
`teste_utilizado=true`.

## O que descartamos

**Não avaliar o Random Forest no teste.** Mantém o teste com um uso só, mas deixa a comparação
entre rede e modelo simples restrita à validação, onde a rede foi escolhida entre cinco versões e
o número dela tende a ser otimista.

**Retreinar ou ajustar o Random Forest antes de levá-lo ao teste.** Qualquer mudança agora seria
feita depois de conhecermos o teste da rede; o modelo que vai ao teste é o de 13/10/2026.

**Reportar o Random Forest só se ele ficar acima da meta, ou rodar mais de uma vez.** Transformaria
o teste em validação.

## Resultado

Executado uma vez em 26/10/2026, run do MLflow `41ecf929305b4edf9e14fc5839d8513b`, com os hashes
acima conferidos pelo script: **CPM 0,420 (IC95% 0,349 a 0,490)**, sensibilidade 0,381 a 1 FP e
0,520 a 4 FP por exame. Nas mesmas reamostras, a CNN ficou 0,304 de CPM acima (IC95% 0,247 a
0,360). Detalhes, dados usados e verificações em
`docs/14-baseline-literatura-sementes-e-teste-rf.md`, seção 3.

## Consequências

O teste deixa de estar intocado para qualquer escolha futura. Se o grupo mudar algum modelo a
partir de agora, o número novo no teste não estima mais desempenho em exames novos e precisa ser
reportado como tal. A comparação entre os dois modelos no teste é descritiva; o resultado da rede
continua sendo o de 13/10/2026.
