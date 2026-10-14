# 0008: o modelo congelado para a avaliação final no teste

**Estado:** decidido antes de abrir o teste
**Data:** 13/10/2026

## Contexto

Os experimentos de 13/10/2026 (`docs/11-experimentos-validacao.md`) compararam cinco versões da
CNN e dois Random Forest nos 89 exames de validação, com o avaliador oficial do LUNA16. O teste,
176 exames de 176 pacientes, não foi lido em nenhuma dessas execuções.

## Decisão

**Congelamos a CNN `cnn_lr_3e-4` e só ela vai para o teste.** Ela teve o maior CPM na validação,
0,752 (IC95% de 0,681 a 0,819), e a diferença pareada contra a versão anterior excluiu zero.

Configuração: 40 épocas, batch 32, learning rate 0,0003, dropout 0,3, weight decay 0,00001,
20 negativos por positivo, aumento de dados ligado, recorte de 32 voxels, semente 42. Run de
treino `2f8b0a3c57c84d9e8f51dea2eab94b7f`.

Identidade dos pesos, registrada antes da avaliação:

- hash dos tensores (`cnn.hash_pesos`): `7e8a9a706ff1a796e0c92b930508034b227db8c4ffdf41c9e52be83cc68d6308`
- SHA-256 de `modelos/cnn_lr_3e-4/modelo.pt`: `f84156769a8e06023450105dc94561dfd23e7a3f7d0a9b4836344a996cdfd58f`

O teste roda uma vez, por `scripts/20_avaliar_teste_cnn.py`, que não treina, confere os hashes
da divisão e da lista V2 gravados no checkpoint e grava a execução num run separado do MLflow.
O resultado do teste não volta para escolher configuração.

## O que descartamos

**Levar mais de uma versão ao teste e reportar a melhor.** Escolher pelo teste o transforma em
validação e o número deixa de estimar o desempenho em exames novos.

**Usar a quinta variante permitida antes de congelar.** Cada escolha a mais na mesma validação
aumenta o otimismo do número escolhido.

## Consequências

O valor da validação foi escolhido entre várias versões e tende a ser otimista; o do teste é a
estimativa que vai para o resumo e para a banca.
