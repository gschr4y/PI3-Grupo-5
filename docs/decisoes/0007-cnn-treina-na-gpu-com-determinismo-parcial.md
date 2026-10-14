# 0007: a CNN treina na GPU, com determinismo que avisa em vez de interromper

**Estado:** decidido e verificado
**Data:** 12/10/2026

## Contexto

A configuração da CNN previa CPU com quatro threads, porque em 05/10/2026 o driver NVIDIA não
respondia nesta máquina. Em 12/10/2026 o `nvidia-smi` passou a mostrar uma GeForce RTX 4050
Laptop com 6 GB e driver 580, compatível com CUDA 13.0. O PyTorch instalado em `.venv-cnn` era a
versão para CPU e não enxergava a placa.

O script de treino liga `torch.use_deterministic_algorithms(True)`, que garante o mesmo
resultado a cada execução com a mesma semente. Na CPU isso funciona. Na GPU, não sabíamos.

## O que medimos

Medimos em 12/10/2026, com a CNN3D do projeto, lotes de 32 cubos de 32³ com valores aleatórios,
50 lotes cronometrados depois de 5 de aquecimento na GPU e 20 depois de 3 na CPU, com
`scripts/21_medir_velocidade_cnn.py`.

| | CPU, 4 threads | RTX 4050 |
|---|---|---|
| Lote de treino | 480 a 530 ms | 21,8 a 23,1 ms |
| Lote de inferência | 172 a 197 ms | 7,2 a 7,5 ms |

As faixas vêm de três execuções no mesmo dia. A CPU variou mais, provavelmente por dividir os
núcleos com o resto do sistema; não medimos a carga da máquina.

Pela contagem de lotes, as 15 épocas sobre os 6.588 cubos de treino levariam cerca de 25 minutos
de cálculo na CPU e pouco mais de 1 minuto na GPU. Isso é estimativa a partir das medidas.

O treino completo na GPU, medido em 12/10/2026 (run `4f52dc350af44d099574a26ac1567731`), levou
565,9 segundos da leitura dos recortes até o fim da época 15, e 13 min 17 s no `dvc repro`
inteiro, incluindo hashes de procedência, previsão da validação e avaliação oficial. A maior parte
desse tempo é leitura de disco, que a GPU não acelera.

**O modo determinístico estrito quebra na GPU.** O primeiro `backward` interrompe com
`RuntimeError`, porque `max_pool3d_with_indices_backward_cuda` não tem implementação
determinística. Foi a única operação apontada; o `AdaptiveAvgPool3d` passou.

Com `warn_only=True`, o PyTorch só avisa sobre essa operação. Dois treinos curtos de 50 lotes,
com a mesma semente e os mesmos dados, terminaram com pesos idênticos bit a bit (diferença máxima
0,0). A explicação que adotamos, por inferência e não por teste: o `MaxPool3d(2)` usa janelas que
não se sobrepõem, então nenhum voxel recebe gradiente de duas saídas ao mesmo tempo.

## Decisão

**Na GPU, ligamos o modo determinístico com `warn_only=True`. Na CPU, ele continua estrito.**
`fixar_determinismo` recebe o device, e o treino registra no MLflow `determinismo` como
`estrito` ou `parcial_gpu`, além do nome da GPU.

Criamos um ambiente separado, `.venv-cnn-gpu`, com `torch==2.14.0+cu130` e as demais versões
iguais às de `requirements-cnn-cpu.lock.txt`. O lock fica em `requirements-cnn-gpu.lock.txt`. O
estágio `treinar_cnn` passou a usar esse ambiente. O estágio `patches` continua em `.venv-cnn` com
o lock de CPU, porque mudar a dependência dele faria o DVC refazer a extração dos 888 exames.

## O que descartamos

**Manter CPU.** Funciona e é determinístico de fato, mas cada rodada leva quase uma hora contra
poucos minutos. Comparar configurações e repetir sementes fica caro demais.

**Desligar o determinismo inteiro na GPU.** Perde a proteção nas outras operações, que hoje
continuam determinísticas, sem ganhar nada.

**Trocar `MaxPool3d` por convolução com passo 2.** Mudaria a arquitetura herdada da pesquisa,
que preservamos de propósito.

## Verificação

`testes/test_cnn.py::test_determinismo_estrito_na_cpu_e_so_aviso_na_gpu` confere os dois modos.
A suíte `.venv-cnn-gpu/bin/python -m pytest testes -q` passou em 137 testes em 12/10/2026, e o
`pip check` do ambiente novo não apontou conflito.

## Consequências

O PyTorch não garante bit a bit o mesmo resultado entre duas execuções na GPU, mesmo que o teste
curto tenha reproduzido. Resultado da GPU não deve ser comparado bit a bit com resultado da CPU.
