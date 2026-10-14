# Preparação e treinamento da CNN

Integramos localmente o extrator, a CNN3D e o adaptador do avaliador oficial da
PR #5. A rede tem três blocos convolucionais com 16, 32 e 64 canais. Não é uma
ResNet. Mantivemos o primeiro Random Forest e seus artefatos.

## Ambiente

Usamos ambientes separados para a CNN, sem modificar o ambiente do Random Forest.
A extração dos recortes roda em `.venv-cnn`, com PyTorch para CPU:

```bash
python3 -m venv .venv-cnn
.venv-cnn/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cnn-cpu.lock.txt
.venv-cnn/bin/python -m pip check
```

O treino roda em `.venv-cnn-gpu`, com PyTorch para CUDA 13.0, desde 12/10/2026
(decisão 0007). As outras versões são as mesmas do lock de CPU.

```bash
python3 -m venv .venv-cnn-gpu
.venv-cnn-gpu/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements-cnn-gpu.lock.txt
.venv-cnn-gpu/bin/python -m pip check
```

Sem GPU, `cnn.device: cpu` no YAML faz o mesmo ambiente treinar na CPU.
`requirements-cnn.lock.txt` continua sendo o registro do ambiente original do Lucas.

## Recortes e procedência

O extrator lê os caminhos do YAML. Cada NPZ contém IDs, coordenadas, classes,
cubos e procedência: SHA-256 do volume e dos arquivos do extrator, além do tamanho.
A retomada confere essa procedência antes de reutilizar o arquivo. Recortes antigos
sem procedência são recriados; um volume ausente é uma falha, mesmo que exista NPZ.
O treino também recusa recortes cuja origem não corresponda aos volumes atuais.

```bash
.venv/bin/dvc repro patches
```

Esse comando gera o lote completo em `dados/processado/patches`, fora do Git.
Na conferência de 06/10/2026, o lote completo terminou com 888 arquivos válidos,
754.975 recortes e nenhuma falha. O resumo em `dados/intermediario/resumo_patches.json`
registra 39.690.009.514 bytes e 5.444,36 segundos de extração. O DVC atualizou
o lock.

## Treino de desenvolvimento

```bash
.venv/bin/dvc repro treinar_cnn
```

O DVC encadeia divisão, recortes e treinamento. A CNN usa todos os candidatos
positivos de treino e até cinco negativos por positivo, com amostragem ponderada.
A validação mantém todos os seus candidatos. Os parâmetros ficam em `cnn` no YAML.
O treino salva os pesos, as probabilidades e a FROC da validação no MLflow e na
pasta `caminhos.cnn_treino`. O teste não é lido nem avaliado por esse comando.

Uma nova configuração deve usar uma nova pasta `caminhos.cnn_treino`. O script
recusa sobrescrever execuções existentes. O histórico de cada execução fica
preservado nos artefatos do MLflow.

## Primeiro treino completo, 12/10/2026

Executamos `.venv/bin/dvc repro treinar_cnn` com a configuração do YAML: 15 épocas,
batch 32, learning rate 0,0001, dropout 0,3, weight decay 0,00001, cinco negativos
por positivo e semente 42, na RTX 4050 (decisão 0007). O DVC pulou inventário, base,
divisão e recortes. O run `4f52dc350af44d099574a26ac1567731` terminou como `FINISHED`,
com `teste_utilizado=false`. Saída em `modelos/cnn_v2`, log local em
`_local/cnn/treino-cnn-v2.log`.

O treino usou 6.588 candidatos V2, 1.098 positivos, de 623 exames. A loss de treino
caiu de 0,659 na época 1 para 0,404 na época 15 (`modelos/cnn_v2/historico.csv`).
O treino levou 565,9 segundos e o `dvc repro` inteiro, 13 min 17 s.

A validação tem 89 exames, 75.063 candidatos e 111 nódulos. O avaliador oficial,
com 1.000 reamostras por exame, mediu esta curva
(`modelos/cnn_v2/avaliacao_validacao/pontos_operacao.csv`):

| FP por exame | Sensibilidade | IC95% |
|---|---|---|
| 0,125 | 0,054 | 0,000 a 0,124 |
| 0,25 | 0,090 | 0,039 a 0,233 |
| 0,5 | 0,225 | 0,127 a 0,306 |
| 1 | 0,270 | 0,175 a 0,367 |
| 2 | 0,333 | 0,235 a 0,452 |
| 4 | 0,405 | 0,306 a 0,525 |
| 8 | 0,495 | 0,386 a 0,611 |

O CPM foi 0,268, com IC95% de 0,198 a 0,358. O avaliador considera no máximo 100
candidatos por exame. Dentro desse limite, 90 dos 111 nódulos receberam algum
candidato e 21 ficaram sem nenhum, o que põe a sensibilidade máxima da curva em
0,811 (`avaliacao_validacao/CADAnalysis.txt`). O teste não foi lido.

## Avaliação final, somente após congelar o modelo

```bash
.venv-cnn/bin/python scripts/20_avaliar_teste_cnn.py --modelo modelos/cnn_v2/modelo.pt --saida modelos/cnn_v2_teste_final
```

Esse comando não treina. Ele verifica os hashes da divisão e da lista V2 contra
os registrados no checkpoint. Em seguida, confere os recortes de teste e executa
a avaliação oficial, em uma execução separada do MLflow. Não foi executado na
base real nesta preparação.

O adaptador usa os arquivos oficiais e registra as adaptações para Python 3.
Preserva a contagem de nódulos perdidos e as marcações excluídas. Além dos intervalos
por ponto da FROC, calcula o intervalo do CPM nas mesmas reamostras por exame.
Não usa média dos limites dos intervalos por ponto para obter o intervalo do CPM.

## Verificação técnica em 05/10/2026

Executamos `scripts/19_preparar_reproducao_cnn.py --saida dados/processado/cnn_verificacao_4_exames`.
A amostra contém dois exames de treino e dois de validação, com pacientes obtidos
do mapa real. O script recria os recortes com a procedência atual e não seleciona
exames do teste. Essa amostra serve para conferir o código, não para escolher o modelo.

Executamos o script 18 com `--dados dados/processado/cnn_verificacao_4_exames`, uma
configuração técnica de uma época e dois threads, e saída em
`modelos/cnn_verificacao_4_exames`. O run
`9002b29b2f0d4c06a1388e8bd300baf1` produziu pesos, probabilidades e avaliação de
validação, com CPM 0. O resultado não estima o desempenho da CNN na divisão completa.
O YAML efetivamente usado foi incluído na procedência do MLflow.

Os relatórios em `relatorios/experimentos/cnn_reproducao` registram o ensaio
original do Lucas com seis exames, incluindo dois do subset 8. São históricos;
não correspondem ao novo comando de treino nem à avaliação final do projeto.
