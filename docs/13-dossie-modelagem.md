# Dossiê da modelagem: dos dados ao teste final

Registro único de como chegamos ao modelo do Grupo 5: o que usamos, o que medimos, o que
testamos, o que deu errado e como corrigimos. Cada número vem com o arquivo ou o comando que o
mediu. Os detalhes de cada etapa estão nos documentos citados; este dossiê os junta em ordem.
Escrito em 14/10/2026.

## 1. O problema e a régua

Somos a etapa de detecção: recebemos a tomografia de tórax, isolamos o pulmão e apontamos os
pontos suspeitos de nódulo. Não classificamos benigno contra maligno.

A régua é a curva FROC do desafio LUNA16: sensibilidade em função da média de falsos positivos
por exame. Reportamos a sensibilidade a 1 e a 4 FP por exame e o CPM, que é a média das
sensibilidades em 0,125, 0,25, 0,5, 1, 2, 4 e 8 FP por exame. Todo número leva intervalo de
95% por bootstrap. Acurácia não serve: com 0,2062% de positivos entre os candidatos, responder
sempre "não é nódulo" acerta 99,79% (`docs/02-analise-exploratoria.md`).

As metas do enunciado estão em `configuracao/config.yaml`, bloco `metas`: sensibilidade de 0,70
a 4 FP por exame para o baseline e de 0,80 para a CNN.

## 2. Linha do tempo

| Data | O que aconteceu | Registro |
|---|---|---|
| 25/08 a 08/09/2026 | Base escolhida, coordenadas conferidas, pré-processamento definido e rodado nos 888 exames | `docs/01` a `docs/03`, decisões 0001 a 0006 |
| 03/09 e 09/09 | Lista de candidatos V2 adotada | `docs/criterios_inclusao_luna16.md` |
| 20/09 | Calibração do detector próprio; limiar 0,20 reprovado | `docs/05-calibracao-detector.md` |
| 28/09 | Divisão fixa por paciente | decisão 0003 |
| Sprint 3 | Primeira FROC, com implementação própria | deck da Sprint 3 no Drive |
| 03/10 | Detector próprio nos 888 exames, cobertura de 97,6% | `docs/04-deteccao-de-candidatos.md` |
| 04/10 | Primeiro Random Forest treinado | `docs/08-primeiro-treino.md` |
| 05 e 06/10 | Integração da CNN e 754.975 recortes com procedência | `docs/09-cnn-mlflow.md` |
| 12/10 | Primeiro treino completo da CNN, na GPU | `docs/09-cnn-mlflow.md`, decisão 0007 |
| 13/10 | Random Forest avaliado e diagnosticado; quatro variantes da CNN; Random Forest centro | `docs/10` e `docs/11` |
| 13/10 | CNN congelada e avaliada uma vez no teste | `docs/12`, decisão 0008 |

## 3. Os dados

**LUNA16**, derivado do LIDC-IDRI: 888 exames, 1.186 nódulos de 3 mm ou mais marcados por pelo
menos três de quatro radiologistas, em 601 exames. 287 exames não têm nódulo e entram na conta
de falsos positivos. Escolhemos o LUNA16 porque o enunciado põe nele o plano e o baseline do
Grupo 5 (decisão 0001).

**Candidatos.** Usamos o `candidates_V2.csv`: 754.975 candidatos, 1.557 positivos. Ele alcança
1.166 dos 1.186 nódulos, contra 1.120 da lista antiga; os 20 restantes não têm candidato nenhum
e nenhum modelo os recupera (`docs/criterios_inclusao_luna16.md`).

**Marcações excluídas.** O desafio tira da conta 35.192 achados; um candidato sobre eles não é
acerto nem alarme falso. Na base, isso ignora 7.972 candidatos do V2 e 26.747 da lista própria
(`dados/intermediario/marcacoes_excluidas.csv`).

**Coordenadas.** As anotações estão em milímetro no espaço do mundo, e o volume, em índice de
voxel. 14 exames têm orientação LPI, que espelha x e y; a fórmula sem a matriz de direção joga
os 24 nódulos desses exames para fora do volume. A conversão usa origem, espaçamento e direção,
e o array vem em (z, y, x) enquanto o cabeçalho vem em (x, y, z). Há teste com exames de
espaçamento diferente e com um exame LPI (`testes/test_coordenadas.py`) e figura de conferência
(`relatorios/figuras/coordenada_lpi.png`).

## 4. Pré-processamento

Quatro passos, nesta ordem (`docs/03-pre-processamento.md`):

1. **Janela de HU de −1000 a 400.** Guarda 98,3% dos voxels do pulmão numa amostra de 30
   exames; corta osso e metal, que dominariam a escala. Vem antes da reamostragem para o osso
   não vazar para o pulmão na interpolação.
2. **Reamostragem para 1 mm isotrópico.** O espaçamento entre fatias vai de 0,45 a 2,5 mm.
   Comparamos oito alvos (decisão 0004): de 0,5 a 1,0 mm o contraste do nódulo não se distingue
   (1,0 mm: 245 HU, IC95% 192 a 282) e 0,5 mm custa oito vezes mais voxels; 2,0 mm perde
   contraste (121 HU, IC95% 94 a 165).
3. **Máscara de pulmão do desafio, sem dilatar, lida como `> 0`.** Pelo critério de acerto do
   LUNA16, 1.185 dos 1.186 nódulos continuam alcançáveis; ler só os rótulos 3 e 4 perderia
   quatro (decisão 0002). Fora do pulmão fica −1000.
4. **Normalização para [0, 1]**, gravada no volume (decisão 0006).

Rodada nos 888 exames em 10/09/2026: zero falhas, 39,8 minutos, 8,6 GiB. Em 37 exames a máscara
do desafio passa de 8 litros e vaza para o ar em volta do paciente; isso aumenta a área de busca,
mas não corta nódulo.

## 5. Divisão por paciente

O LUNA16 não identifica o paciente. Pelo metadata do LIDC-IDRI, os 888 exames são de 887
pacientes; o `LIDC-IDRI-0332` tem duas séries, nos subsets 2 e 6 (decisão 0003).

Divisão fixa (`dados/processado/divisao.csv`): treino nos subsets 0 a 6, com 623 exames de 622
pacientes; validação no subset 7, com 89; teste nos subsets 8 e 9, com 176. As duas séries do
paciente repetido ficam no treino. Os scripts de treino e avaliação recusam paciente
compartilhado entre partições (`cnn.conferir_divisao`).

| Partição | Exames | Candidatos V2 positivos | Candidatos V2 negativos | Nódulos avaliados |
|---|---|---|---|---|
| Treino | 623 | 1.098 | 528.402 | não avaliado |
| Validação | 89 | 120 | 74.943 | 111 |
| Teste | 176 | 339 | 150.073 | 223 |

Os candidatos positivos são mais numerosos que os nódulos porque um nódulo pode ter mais de um
candidato em cima. Os nódulos avaliados saem do `CADAnalysis.txt` do avaliador oficial.

## 6. A busca própria de candidatos

O detector do grupo usa blob detection 3D por Laplaciano de Gaussiana, com sigma de 1 a 5 e
limiar 0,10 (`docs/04-deteccao-de-candidatos.md`). Nos 888 exames, em 03/10/2026: 14.310.042
candidatos, 16.114,9 por exame, alcançando 1.158 dos 1.186 nódulos, 97,6% (IC95% 96,7% a 98,6%,
bootstrap por exame). O V2 alcança 1.166 com 850,2 candidatos por exame.

A calibração de 20/09/2026 testou limiares maiores. O limiar 0,20 cortou 67,1% dos candidatos
na validação, mas perdeu um nódulo de 4,96 mm; a regra fixada antes reprovou essa configuração
(`docs/05-calibracao-detector.md`).

Os modelos deste dossiê foram treinados e avaliados sobre a lista V2, não sobre a lista própria.
A lista própria tem a cobertura medida, mas ainda não passou por classificador.

## 7. Recortes

Para cada candidato V2, um cubo de 34 voxels (34 mm a 1 mm) em volta do centro. Com 32 voxels,
um nódulo de 32,27 mm não cabia; com 34, nenhum fica de fora (decisão 0004). Fora do volume o
recorte é preenchido com zero.

A extração completa terminou em 05/10/2026 e foi conferida em 06/10: 888 arquivos NPZ, 754.975
recortes, zero falhas, 39.690.009.514 bytes, 5.444,36 segundos
(`dados/intermediario/resumo_patches.json`). Cada NPZ guarda o SHA-256 do volume e do código do
extrator; o treino recusa recorte cuja origem não bata com o volume atual
(`src/detection/procedencia_recortes.py`). A CNN usa o cubo central de 32 voxels de cada recorte.

## 8. Como avaliamos

Usamos o programa oficial do LUNA16, `noduleCADEvaluationLUNA16.py`, guardado sem alteração em
`third_party/luna16`. Ele foi escrito em Python 2; `src/experimentos/avaliacao_oficial.py` gera
em pasta temporária uma cópia com os ajustes de sintaxe para Python 3 e registra o hash do
original e da cópia em cada avaliação. Dois acréscimos estão declarados no código:

- quando nenhum nódulo é detectado, o scikit-learn devolve sensibilidade indefinida; tratamos
  como zero, porque o denominador oficial já conta os nódulos sem candidato;
- o intervalo do CPM é calculado nas mesmas 1.000 reamostras por exame que o programa usa para a
  curva, e não pela média dos limites de cada ponto.

O programa considera os 100 maiores escores de cada exame e usa as marcações excluídas. A
semente do sorteio é 42, sobre a lista ordenada de exames; com os mesmos exames, dois modelos
recebem as mesmas reamostras, e por isso calculamos a diferença de CPM reamostra a reamostra,
com intervalo pareado (`scripts/23_comparar_validacao.py`).

Toda escolha foi feita na validação. O teste foi lido uma única vez, para o modelo congelado.

## 9. Os modelos, em ordem

### 9.1 Primeira FROC, Sprint 3

O deck da Sprint 3 registra uma avaliação em 105 exames de teste, com implementação própria da
FROC e 500 reamostras: Random Forest com CPM 0,013 e sensibilidade de 0,022 a 4 FP por exame;
CNN com CPM 0,593 (IC95% 0,524 a 0,663) e 0,896 a 4 FP (IC95% 0,836 a 0,959). O próprio slide
registra que o número não era comparável ao ranking sem o programa oficial e sem as marcações
excluídas. Essa execução não foi reproduzida no repositório, e os 105 exames não correspondem à
divisão fixa da seção 5. Os números dela não se comparam com os desta seção em diante.

### 9.2 Ensaios técnicos da CNN, 6 e 4 exames

O ensaio original da integração da PR #5 treinou com 18 candidatos de poucos exames e teve CPM 0
na validação (`relatorios/experimentos/cnn_reproducao/execucao.json`). Repetimos em 05/10/2026
com 2 exames de treino e 2 de validação, uma época, run `9002b29b2f0d4c06a1388e8bd300baf1`: CPM
0, também. Os dois ensaios só verificam que o fluxo treina, salva, recarrega e avalia; com 18
candidatos de treino não há o que aprender.

### 9.3 Random Forest v2: sete features do cubo inteiro

Treinado em 04/10/2026 (`docs/08-primeiro-treino.md`, run `1663d6b906ee486bb0cfff9fdd09457d`):
200 árvores, profundidade máxima 20, mínimo de 2 candidatos por folha, `max_features` raiz
quadrada, pesos `balanced`, semente 42. Entrada: média, desvio padrão, curtose, mínimo, máximo e
percentis 25 e 75 do cubo de 34 voxels, para os 529.500 candidatos de treino.

Avaliado pelo programa oficial em 13/10/2026 (run `e3ed9e0c20ac4494aa6d97af2c779baa`): **CPM 0,003**
(IC95% 0,000 a 0,007), sensibilidade zero até 4 FP por exame.

O professor listou quatro hipóteses para o modelo quase não funcionar. Testamos as quatro
(`docs/10-comparacao-validacao.md`, `scripts/24_diagnosticar_random_forest.py`):

| Hipótese | Teste | Resultado |
|---|---|---|
| Dados diferentes dos da CNN | Features recalculadas dos NPZ que a CNN lê, 2.655 candidatos | Diferença máxima da ordem de 1e-16 |
| Rótulo deslocado | Identificador e coordenada contra a lista V2, 75.063 candidatos | Todos alinhados |
| Coluna de probabilidade invertida | Coluna escolhida por `classes_ == 1`; mediana dos positivos contra negativos | 0,104 contra 0,064 |
| Conjuntos de candidatos diferentes | Script 23 recusa comparar se divergirem | Comparação passou |

Nenhuma se confirmou. O que os números mostram é outra coisa: o modelo separa positivo de
negativo com 1,000 no treino e 0,593 na validação, onde 0,5 é o acaso, e cada feature sozinha
separa entre 0,426 e 0,544. Ou seja, decorou o treino. Só 25 dos 120 positivos de validação
ficam entre os 100 maiores escores do próprio exame, e o resto o avaliador descarta.

### 9.4 CNN v2: primeiro treino completo

Arquitetura preservada da pesquisa (`src/experimentos/cnn.py`): três blocos de convolução 3D com
16, 32 e 64 canais, cada um com normalização em lote, ReLU e max pooling 2, seguidos de média
global, dropout 0,3 e camadas densas de 64 para 32 e de 32 para 2. Entrada: cubo de 32 voxels.
Perda de entropia cruzada, otimizador Adam.

Amostragem: todos os 1.098 positivos de treino e um sorteio fixo de negativos, com semente 42, na
proporção configurada. Em cada época, um sorteio com reposição ponderado pelo inverso da
frequência de cada classe equilibra os lotes. A saída é a probabilidade da classe nódulo após
softmax.

Configuração registrada antes de treinar: 15 épocas, lote 32, learning rate 0,0001, weight decay
0,00001, 5 negativos por positivo. Rodamos em 12/10/2026 numa RTX 4050 (run
`4f52dc350af44d099574a26ac1567731`, 13 min 17 s no `dvc repro`). A loss de treino caiu de 0,659
para 0,404. **CPM de validação 0,268** (IC95% 0,198 a 0,358); 0,270 a 1 FP e 0,405 a 4 FP.

Sobre a GPU (decisão 0007): o modo determinístico estrito do PyTorch interrompe o treino no
gradiente do max pooling 3D, que não tem versão determinística em CUDA. Na GPU ele passou a só
avisar. Dois treinos curtos com a mesma semente deram pesos idênticos, e a época 10 da variante
de 40 épocas repetiu a loss da época 10 do `cnn_v2` (0,466619), mas o PyTorch não garante essa
igualdade. Um lote de treino levou cerca de 22 ms na GPU contra cerca de 500 ms na CPU
(`scripts/21_medir_velocidade_cnn.py`).

### 9.5 Variantes da CNN, só na validação

Regras fixadas antes de rodar (`docs/11-experimentos-validacao.md`): uma mudança por vez sobre a
melhor anterior, a mudança fica se o CPM pontual subir, no máximo cinco variantes, diferença
pareada com IC95% para todas.

| Modelo | Run | CPM | IC95% | Sens. 1 FP | Sens. 4 FP | Nódulos nos 100 maiores |
|---|---|---|---|---|---|---|
| CNN v2, 15 épocas | `4f52dc35` | 0,268 | 0,198 a 0,358 | 0,270 | 0,405 | 90 de 111 |
| 40 épocas | `a90b6d29` | 0,523 | 0,439 a 0,605 | 0,550 | 0,730 | 99 |
| + aumento de dados | `7b2c2f43` | 0,525 | 0,438 a 0,608 | 0,568 | 0,712 | 99 |
| + 20 negativos por positivo | `5529e76d` | 0,696 | 0,625 a 0,771 | 0,748 | 0,838 | 103 |
| + learning rate 0,0003 | `2f8b0a3c` | 0,752 | 0,681 a 0,819 | 0,784 | 0,883 | 107 |

| Mudança | Diferença de CPM | IC95% pareado |
|---|---|---|
| 15 para 40 épocas | 0,255 | 0,173 a 0,322 |
| + aumento de dados | 0,003 | −0,054 a 0,055 |
| + 20 negativos por positivo | 0,171 | 0,107 a 0,256 |
| + learning rate 0,0003 | 0,055 | 0,014 a 0,085 |

O que cada mudança mostrou, com a leitura marcada como inferência quando for:

- **40 épocas.** Na época 15 a loss ainda caía, sinal de treino interrompido cedo. Treinar mais
  dobrou o CPM, com intervalo pareado acima de zero.
- **Aumento de dados** (espelhamento nos três eixos e rotação de 90 graus no plano axial). A
  diferença não se distingue de zero; entrou na cadeia só porque o CPM pontual subiu 0,003. Por
  inferência, com 40 épocas o modelo ainda não decorava o treino a ponto de o aumento ajudar.
- **20 negativos por positivo.** Foi o maior ganho depois das épocas. Com 5 negativos por
  positivo o treino via 5.490 dos 528.402 negativos; com 20, 21.960. Por inferência, ver mais
  variedade de não nódulo ensinou a rede a separar os casos parecidos com nódulo, que são os que
  ocupam os 100 primeiros lugares de cada exame.
- **Learning rate 0,0003.** A loss terminou em 0,066, contra 0,136, e o CPM subiu 0,055 com
  intervalo acima de zero. Por inferência, com 0,0001 o otimizador ainda não tinha convergido em
  40 épocas.

O número de nódulos que entram nos 100 maiores escores de cada exame subiu de 90 para 107 de
111. O corte de 100 marcas não impõe um teto fixo: o teto depende de como cada modelo ordena os
candidatos.

### 9.6 Random Forest centro: o baseline corrigido

Mesmo modelo e mesmos hiperparâmetros do Random Forest v2, mas com as sete estatísticas
calculadas em três cubos concêntricos de 8, 16 e 34 voxels, 21 features
(`scripts/25_treinar_random_forest_centro.py`, run `5169e10d13874969804d14f29cca6ea6`).

**CPM 0,409** (IC95% 0,323 a 0,505), 0,405 a 1 FP e 0,559 a 4 FP. A diferença pareada contra o
Random Forest v2 foi 0,407 (IC95% 0,322 a 0,501). O nódulo típico tem 6,4 mm e fica no centro do
recorte de 34 mm; por inferência, a estatística do cubo inteiro diluía o nódulo no tecido em
volta, e o cubo de 8 voxels mede quase só o nódulo. A meta do baseline, 0,70 a 4 FP, não foi
atingida.

Na validação, a CNN final supera o Random Forest centro por 0,342 de CPM (IC95% 0,252 a 0,437).

### 9.7 Modelo congelado e teste

Congelamos a variante de learning rate 0,0003 antes de abrir o teste, com o hash dos pesos
registrado na decisão 0008. Avaliamos uma vez nos 176 exames de teste, com 223 nódulos (run
`d754f0738780451a8a734be0ac58a435`, `docs/12-avaliacao-final-teste.md`):

| FP por exame | 0,125 | 0,25 | 0,5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| Sensibilidade | 0,457 | 0,574 | 0,641 | 0,735 | 0,852 | 0,883 | 0,924 |

**CPM 0,724** (IC95% 0,670 a 0,778). Sensibilidade de 0,883 a 4 FP por exame (IC95% 0,843 a
0,929), acima da meta de 0,80 em todo o intervalo, e de 0,735 a 1 FP (IC95% 0,659 a 0,819).
Dentro do corte de 100 marcas, 218 dos 223 nódulos tiveram candidato. O CPM do teste ficou 0,028
abaixo do da validação.

O Random Forest centro não foi avaliado no teste.

## 10. Erros encontrados e como corrigimos

Os erros que mais pesaram foram os silenciosos: os que não levantam exceção e produzem um
resultado com cara de válido.

| Erro | Como apareceu | Correção | Registro |
|---|---|---|---|
| Conversão de coordenada sem a matriz de direção | Os 24 nódulos dos 14 exames LPI caíam fora do volume | Conversão com direção, teste com exame LPI e figura | `docs/01-dados.md` |
| Máscara lida com os rótulos 3 e 4 | Perderia quatro nódulos em vez de um | Leitura `> 0` | decisão 0002 |
| Limite de 2,5 mm comparado por igualdade | Dois exames passavam do limite por um bit de float32 | Comparação com tolerância | `docs/criterios_inclusao_luna16.md` |
| Afirmação de que cada `seriesuid` é um paciente | Um paciente tem duas séries em subsets diferentes | Mapa de pacientes e divisão por paciente | decisão 0003 |
| Normalização dividindo por escalar | Volume promovido a float64, dobrando o arquivo | Multiplicação pelo inverso, com teste | decisão 0006 |
| Figura de antes e depois com a mesma faixa de cinza | Painel virou retângulo chapado depois da normalização | Faixa por painel, com teste que mede a figura | decisão 0006 |
| Chave de configuração que nenhum código lia | Pôr `false` mantinha a máscara sem aviso | Chave removida | `docs/historico-do-projeto.md` |
| Testes que não pegavam o que prometiam | Teste de vazamento passava com a ordem trocada; o das máscaras largas só contava | Testes reescritos | `docs/historico-do-projeto.md`, auditoria de 09/09 |
| DVC apagando figuras versionadas | Com o HD desconectado, o estágio saía com código zero e quatro figuras sumiam | `persist: true` nas nove figuras e o script de EDA falhando quando não gera as cinco | commit `83f5f6b` |
| Cache do detector aceitando arquivos inconsistentes | A retomada podia publicar conclusão com exames ausentes | Conferência de identidade e interrupção em falha | `docs/historico-do-projeto.md`, 05/10 |
| Recortes sem procedência | Um recorte de volume antigo seria aceito no treino | Hash do volume e do extrator em cada NPZ | `docs/09-cnn-mlflow.md` |
| Treino e teste no mesmo comando | A integração original avaliava o teste junto do treino | Treino só com treino e validação; teste em comando separado | `docs/09-cnn-mlflow.md` |
| Modo determinístico estrito na GPU | O treino parava no gradiente do max pooling 3D | Modo que avisa na GPU, estrito na CPU | decisão 0007 |
| Features do cubo inteiro no Random Forest | CPM 0,003 sem nenhum erro de pipeline | Estatísticas em cubos concêntricos, CPM 0,409 | `docs/11-experimentos-validacao.md` |
| Corte de 100 marcas tratado como teto fixo | Uma leitura inicial pôs o teto da validação em 0,811 | O teto depende da ordenação; a melhor CNN alcança 107 de 111 | `docs/11-experimentos-validacao.md` |
| Número de FROC de implementação própria comparado ao oficial | 0,896 a 4 FP na Sprint 3 contra 0,405 na primeira avaliação oficial | Só números do programa oficial se comparam entre si | seção 9.1 |

Também registramos perdas que não são erro, mas limitam o sistema: um nódulo de 5 mm sai com a
máscara do desafio, 20 nódulos não têm candidato no V2, e no teste 5 dos 223 nódulos ficaram
fora dos 100 maiores escores.

## 11. O que este resultado não diz

- A configuração final foi escolhida entre cinco versões olhando a mesma validação de 89 exames;
  o CPM de validação dela tende a ser otimista. O número que estima o desempenho em exames novos
  é o do teste.
- O teste já foi usado. Uma escolha nova feita a partir dele perde o valor de estimativa.
- O Random Forest centro não tem avaliação no teste; a comparação entre os dois modelos com
  intervalo existe só na validação.
- Os modelos usam a lista V2. A lista própria do grupo tem cobertura medida, mas não foi
  classificada.
- A comparação com trabalhos publicados no LUNA16 está em
  `docs/14-baseline-literatura-sementes-e-teste-rf.md`, com fontes conferidas em 26/10/2026.
  Nenhum dos artigos usa o nosso protocolo de teste fixo; os números são referência, não
  comparação direta.
- A reprodutibilidade na GPU é parcial: o PyTorch não garante pesos idênticos entre execuções.

## 12. Como reproduzir

Ambientes: `.venv` para o pipeline e o Random Forest, `.venv-cnn` para os recortes e
`.venv-cnn-gpu` para o treino da CNN (`docs/09-cnn-mlflow.md`).

```bash
.venv/bin/dvc repro patches
.venv/bin/dvc repro treinar_random_forest avaliar_random_forest treinar_random_forest_centro
.venv/bin/dvc repro treinar_cnn treinar_cnn_variante@epocas_40 treinar_cnn_variante@aumento treinar_cnn_variante@negativos_20 treinar_cnn_variante@lr_3e-4
.venv/bin/dvc repro comparar_validacao diagnosticar_random_forest
.venv-cnn-gpu/bin/python scripts/20_avaliar_teste_cnn.py --modelo modelos/cnn_lr_3e-4/modelo.pt --saida <pasta-nova>
.venv-cnn-gpu/bin/python -m pytest testes -q
```

Pesos, probabilidades e avaliações ficam em `modelos/`. A rede escolhida, a avaliação no teste e o
Random Forest centro estão versionados no Git desde 25/10/2026; as demais variantes ficam fora, e
cada execução tem cópia no MLflow com código, ambiente e hashes das entradas. A tabela de modelos está em
`modelos/README.md`, e as curvas de validação dos sete modelos em
`relatorios/experimentos/comparacao_validacao/froc_validacao.png`.
