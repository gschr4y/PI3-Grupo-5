# Baseline da literatura, segundo uso do teste e sementes da rede, 26/10/2026

Este documento fecha a etapa de baseline. Ele separa três tipos de número: **medido** (com
arquivo no repositório), **pendente** (script pronto, execução não feita) e **da literatura**
(publicado por outros, com outro protocolo). Nenhum número pendente foi estimado.

## 1. Pesquisa de baseline na literatura

Conferimos título, autores, DOI e tabelas no texto completo de cada artigo, em 26/10/2026.

### 1.1 As duas referências citadas pelo grupo

**Zheng et al. (2022) — confirmada.** Zheng S, Kong S, Huang Z, Pan L, Zeng T, Zheng B, Yang M,
Liu Z. *A Lower False Positive Pulmonary Nodule Detection Approach for Early Lung Cancer
Screening*. Diagnostics, 12(11):2660, 2022. DOI
[10.3390/diagnostics12112660](https://doi.org/10.3390/diagnostics12112660). É o mesmo trabalho
que a decisão 0002 já cita pela dilatação da máscara com esfera de raio 6.

**Li et al. (2025) — localizada, identificação a confirmar pelo grupo.** Li Y, Hui L, Wang X,
Zou L, Chua S. *Lung nodule detection using a multi-scale convolutional neural network and global
channel spatial attention mechanisms*. Scientific Reports, 15:12313, 2025. DOI
[10.1038/s41598-025-97187-w](https://doi.org/10.1038/s41598-025-97187-w). Foi o único artigo de
2025 com primeiro autor Li, LUNA16 e CPM que encontramos, e a tabela de comparação dele cita
Zheng et al. com os mesmos sete valores do artigo de 2022, o que sugere que o grupo tirou as
duas referências daí. **Isso é inferência:** quem trouxe a referência precisa confirmar o DOI.
Não confundir com o "Li et al. (2023)" da decisão 0002, que é outro trabalho, sobre perda de
nódulos na segmentação.

### 1.2 Tabela

Sensibilidades e CPM copiados das tabelas dos artigos. "Sens. 1 FP" é a sensibilidade a 1 falso
positivo por exame.

| Artigo | Ano | Método | Dataset | Protocolo | CPM | Sens. 1 FP | Observações |
|---|---:|---|---|---|---:|---:|---|
| Li et al., Sci. Rep. 15:12313 | 2025 | Sistema completo: detector 3D Res2Net com atenção GCSAM e RPN (CNDNet) + rede de redução de FP em cubos de 32³ (FPRNet) | LUNA16, 888 exames, 1.186 nódulos; 754.975 candidatos para a redução de FP | Validação cruzada em 10 dobras (subsets oficiais); IC por bootstrap de 1.000; não diz se usou o script oficial | 0,929 | 0,971 | Sete pontos: 0,773 / 0,888 / 0,945 / 0,971 / 0,977 / 0,977 / 0,977. Só o detector: CPM 0,911. Gera os próprios candidatos. |
| Zheng et al., Diagnostics 12:2660 | 2022 | Sistema completo: detector 3D multiescala tipo FPN com RPN (MSFD-Net) + classificador de candidatos em 32³ (CS-Net) | LUNA16, 888 exames, 1.186 nódulos; candidates V2 no treino da redução de FP | Validação cruzada em 10 dobras; métrica do desafio; não nomeia o script | 0,906 | 0,940 | Sete pontos: 0,771 / 0,849 / 0,905 / 0,940 / 0,957 / 0,960 / 0,960. Máscara de pulmão dilatada. |
| Setio et al., Med. Image Anal. 42:1–13 ([10.1016/j.media.2017.06.015](https://doi.org/10.1016/j.media.2017.06.015)) | 2017 | Desafio LUNA16, trilha de redução de falsos positivos: CNNs sobre a lista de candidatos fornecida | LUNA16, 888 exames, 1.186 nódulos; 754.975 candidatos (V2) | Validação cruzada em 10 dobras; script oficial; até 100 marcas por exame; IC por bootstrap de 1.000 | 0,908 (melhor, CUMedVis) a 0,758 (pior, ZNET) | 0,972 a 0,793 | **Mesma tarefa e mesmo avaliador que os nossos.** Fora de 2022–2025, mas é a referência de protocolo. |
| Ma et al. (TiCNet), J. Imaging Inform. Med. 37:196–208 ([10.1007/s10278-023-00904-y](https://doi.org/10.1007/s10278-023-00904-y)) | 2024 | Sistema completo: CNN 3D codificador-decodificador com transformer | LUNA16, 888 exames, 1.186 nódulos | 10 dobras; acerto por IoU 3D > 0,1, **não** pelo raio do nódulo do script oficial | 0,907 | 0,922 | Critério de acerto diferente do oficial. |
| Wang et al., Digital Health 12 ([10.1177/20552076261419238](https://doi.org/10.1177/20552076261419238)) | 2026 (submetido em 2025) | Sistema completo em dois estágios: detector 3D em U com atenção + rede multiescala de redução de FP | LUNA16, 888 exames, 1.186 nódulos | Validação cruzada em **6** dobras; não cita o script oficial | 0,878 | 0,939 | Mostra que mudar o número de dobras já muda a régua. |

### 1.3 O que é e o que não é comparável

Nosso protocolo: redução de falsos positivos sobre a lista **candidates_V2 fornecida pelo
desafio**, avaliador oficial do LUNA16 (`third_party/luna16`), até 100 marcas por exame, divisão
fixa por paciente com treino nos subsets 0–6, validação no 7 e **teste nos subsets 8 e 9, 176
exames e 223 nódulos**, uma única vez.

- Li (2025), Zheng (2022), TiCNet e Wang são **sistemas completos**: geram os próprios
  candidatos, usam redes muito maiores e reportam a média de 10 (ou 6) dobras sobre os 888
  exames. Nosso número é de um teste fixo de 176 exames. As duas coisas medem desempenho de
  forma diferente; os números servem de **referência do estado da arte, não de comparação
  direta**.
- A única referência com a **mesma tarefa** (classificar a lista V2) e o **mesmo avaliador** é a
  trilha de redução de falsos positivos do desafio (Setio et al., 2017). Mesmo nela o protocolo
  difere no ponto da divisão: 10 dobras contra o nosso teste fixo.
- TiCNet troca o critério de acerto e Wang usa 6 dobras. Os dois ficam só como contexto.

### 1.4 Quais usar nos slides

1. **Li et al. (2025)** como estado da arte recente (CPM 0,929), depois que o grupo confirmar
   que é esse o artigo.
2. **Zheng et al. (2022)** (CPM 0,906): confirmado, já citado no projeto, e aparece na tabela de
   Li, o que liga as duas referências.
3. **Setio et al. (2017), trilha de redução de FP** (CPM de 0,758 a 0,908): a referência de
   protocolo, a única que mede a mesma tarefa com o mesmo programa.

Em todos os casos, o slide diz que o protocolo é diferente do nosso.

## 2. Segundo uso do teste

Registrado na decisão 0009, antes de qualquer execução, com o texto para o relatório e os
hashes do Random Forest centro congelado, da divisão e da lista V2.

## 3. Random Forest centro no teste e FROC conjunta

### 3.1 Como a rede foi avaliada no teste (material reaproveitado)

`modelos/cnn_lr_3e-4_teste_final/` foi produzido por `scripts/20_avaliar_teste_cnn.py`, que
carrega o checkpoint congelado, confere os hashes da divisão e da lista V2, prevê os 150.412
candidatos V2 dos 176 exames de teste na CPU e chama `src/experimentos/avaliacao_oficial.py`.

| Arquivo | Conteúdo |
|---|---|
| `predicoes.csv` | Escore de cada candidato, entrada do avaliador oficial |
| `seriesuids.csv` | Os 176 exames de teste |
| `curva_froc_completa.csv` | Curva FROC inteira (`froc_predicoes.txt` convertido) |
| `froc_predicoes_bootstrapping.csv` | Média e IC95% da sensibilidade em 10.000 pontos de FP, das 1.000 reamostras por exame |
| `pontos_operacao.csv` | Os sete pontos oficiais com IC95% |
| `cpm_bootstrap.csv` | CPM de cada uma das 1.000 reamostras, para o IC do CPM e a diferença pareada |
| `CADAnalysis.txt` | 223 nódulos; 218 com candidato nos 100 maiores escores |
| `avaliacao.json`, `execucao.json` | Protocolo (semente 42, 1.000 reamostras, 95%) e métricas; run `d754f073...` |

**Método do intervalo de confiança.** O projeto já tem um: o bootstrap por exame do programa
oficial, 1.000 reamostras, percentis de 95%, semente 42. O IC dos sete pontos vem do próprio
programa, e o IC do CPM vem dos 1.000 CPMs das reamostras. Usamos o mesmo método para o Random
Forest. Com a mesma semente e a mesma lista de exames as reamostras são as mesmas para os dois
modelos (a função `generateBootstrapSet` sorteia exames, não candidatos), o que permite a
diferença pareada de CPM, como em `docs/11`.

### 3.2 Resultados

| Modelo | Conjunto | CPM | IC95% | Sens. 1 FP | Sens. 4 FP | Situação |
|---|---|---:|---|---:|---:|---|
| CNN `cnn_lr_3e-4` | Teste, 176 exames | 0,724 | 0,670 a 0,778 | 0,735 | 0,883 | **Medido** em 13/10/2026 |
| Random Forest centro | Teste, 176 exames | — | — | — | — | **Pendente**: precisa rodar o script 26 |
| CNN `cnn_lr_3e-4` | Validação, 89 exames | 0,752 | 0,681 a 0,819 | 0,784 | 0,883 | Medido |
| Random Forest centro | Validação, 89 exames | 0,409 | 0,323 a 0,505 | 0,405 | 0,559 | Medido |

Sensibilidade da CNN no teste nos sete pontos: 0,457 / 0,574 / 0,641 / 0,735 / 0,852 / 0,883 /
0,924.

**Bloqueio.** O Random Forest precisa das features dos 176 exames de teste, calculadas dos
recortes em `dados/processado/patches/*.npz`, e o avaliador precisa de `annotations.csv` e
`annotations_excluded.csv`. Nada disso está no repositório nem nesta máquina (os dados vivem no
HD externo indicado em `configuracao/config.yaml`). O script está pronto:

```bash
.venv/bin/python scripts/26_avaliar_teste_random_forest.py
.venv/bin/python scripts/27_comparar_froc.py \
    --avaliacao "CNN3D=modelos/cnn_lr_3e-4_teste_final" \
    --avaliacao "Random Forest centro=modelos/random_forest_centro_teste_final" \
    --titulo "Teste" --saida relatorios/experimentos/comparacao_teste
```

O script 26 recusa rodar se o hash do modelo, da divisão ou da lista V2 diferir do registrado na
decisão 0009, se a versão do scikit-learn não for a do treino (1.9.1), se os exames de teste
diferirem dos avaliados na CNN ou se a pasta de saída já existir.

O script 27 já foi conferido na validação, com as duas avaliações versionadas: reproduziu a
diferença pareada registrada em `docs/11` (0,342, IC95% 0,252 a 0,437). A figura está em
`relatorios/experimentos/comparacao_validacao_cnn_rf/froc.png`. A do teste só existirá depois do
script 26.

## 4. Três sementes da rede, na validação

Decisão 0010: sementes 43 e 44, fixadas antes de treinar; configuração idêntica; só validação.

| Seed | CPM | Sens@0,125 | Sens@0,25 | Sens@0,5 | Sens@1 | Sens@2 | Sens@4 | Sens@8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 (original, medido) | 0,752 | 0,505 | 0,631 | 0,730 | 0,784 | 0,838 | 0,883 | 0,892 |
| 43 | pendente | pendente | pendente | pendente | pendente | pendente | pendente | pendente |
| 44 | pendente | pendente | pendente | pendente | pendente | pendente | pendente | pendente |

Semente 42: checkpoint da época 40 (o protocolo não escolhe época), loss de treino final 0,066,
IC95% do CPM 0,681 a 0,819, run `2f8b0a3c57c84d9e8f51dea2eab94b7f`.

| Métrica | Resultado |
|---|---:|
| CPM médio | pendente |
| Desvio padrão | pendente |
| CPM mínimo | pendente |
| CPM máximo | pendente |

Com uma semente só, média e desvio não existem; não os calculamos.

**Bloqueio.** O treino precisa dos recortes de treino e validação (`dados/processado/patches`),
da lista V2 e das anotações, e de GPU CUDA. Esta máquina não tem os dados e tem GPU AMD, sem
CUDA; o PyTorch instalado é só CPU. Treinar na CPU também mudaria o modo de determinismo em
relação à semente 42 (decisão 0007). Comandos na decisão 0010.

Três análises que não se misturam:

- **IC95% do bootstrap por exame**: incerteza por termos poucos exames, com o modelo fixo.
- **Variação entre sementes**: quanto o resultado muda ao treinar de novo a mesma configuração.
- **Desvio padrão**: o resumo numérico dessa variação, com só três valores.

## 5. Baseline, resultados e metas

| Item | Número | Métrica e conjunto | Natureza |
|---|---:|---|---|
| Li et al. (2025) | CPM 0,929; 0,977 a 4 FP | 10 dobras, sistema completo | Literatura |
| Zheng et al. (2022) | CPM 0,906; 0,960 a 4 FP | 10 dobras, sistema completo | Literatura |
| Trilha de redução de FP do LUNA16 | CPM 0,758 a 0,908 | 10 dobras, lista V2, avaliador oficial | Literatura |
| Random Forest centro | CPM 0,409; 0,559 a 4 FP (IC95% 0,453 a 0,650) | Validação | Medido |
| Random Forest centro | — | Teste | Pendente |
| CNN `cnn_lr_3e-4` | CPM 0,724; 0,883 a 4 FP (IC95% 0,843 a 0,929) | Teste | Medido |
| Meta do baseline | sensibilidade ≥ 0,70 a 4 FP por exame | Um ponto da FROC | Meta, não resultado |
| Meta da CNN | sensibilidade ≥ 0,80 a 4 FP por exame | Um ponto da FROC | Meta, não resultado |

**As metas não estão na mesma métrica da literatura.** As metas do enunciado (bloco `metas` do
`config.yaml`) são a sensibilidade num único ponto, 4 FP por exame. A literatura reporta o CPM,
a média de sete pontos de 0,125 a 8 FP, que pesa muito os pontos de poucos falsos positivos. Uma
meta de 0,80 a 4 FP não equivale a CPM 0,80. Comparamos cada meta com a nossa sensibilidade a
4 FP, e o CPM com o da literatura, sempre lembrando a diferença de protocolo.

Situação das metas, sem transformar meta em resultado:

- **CNN:** no teste, 0,883 a 4 FP, com o IC95% inteiro acima de 0,80.
- **Random Forest centro:** na validação, 0,559 a 4 FP, com o IC95% inteiro abaixo de 0,70. No
  teste, pendente.
