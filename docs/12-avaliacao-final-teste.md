# Avaliação final no teste, 13/10/2026

Avaliamos uma única vez no teste a CNN congelada pela decisão 0008, `cnn_lr_3e-4`. O teste tem
176 exames de 176 pacientes, nenhum deles usado em treino, validação ou escolha de configuração.

```bash
.venv-cnn-gpu/bin/python scripts/20_avaliar_teste_cnn.py --modelo modelos/cnn_lr_3e-4/modelo.pt --saida modelos/cnn_lr_3e-4_teste_final
```

O script não treina. Ele conferiu os hashes da divisão e da lista V2 contra os gravados no
checkpoint e a procedência dos recortes de teste. O run do MLflow
`d754f0738780451a8a734be0ac58a435` terminou como `FINISHED`, com `teste_utilizado=true` e hash
dos pesos `7e8a9a70...`, o mesmo registrado na decisão 0008 antes da execução. A avaliação levou
21 min 13 s, com a previsão na CPU.

## Resultado

Avaliador oficial do LUNA16, 223 nódulos, 1.000 reamostras por exame
(`modelos/cnn_lr_3e-4_teste_final/pontos_operacao.csv`):

| FP por exame | Sensibilidade | IC95% |
|---|---|---|
| 0,125 | 0,457 | 0,364 a 0,555 |
| 0,25 | 0,574 | 0,479 a 0,640 |
| 0,5 | 0,641 | 0,561 a 0,728 |
| 1 | 0,735 | 0,659 a 0,819 |
| 2 | 0,852 | 0,801 a 0,896 |
| 4 | 0,883 | 0,843 a 0,929 |
| 8 | 0,924 | 0,883 a 0,962 |

O CPM foi 0,724, com IC95% de 0,670 a 0,778. Dentro do limite de 100 candidatos por exame, 218
dos 223 nódulos receberam candidato (`CADAnalysis.txt`).

Na validação, a mesma configuração teve CPM 0,752 (IC95% de 0,681 a 0,819) e sensibilidade de
0,883 a 4 FP por exame. A meta do enunciado para a CNN, registrada em `metas` no YAML, é 0,80 a
4 FP por exame.

O Random Forest não foi avaliado no teste.

**Atualização de 26/10/2026.** Pela decisão 0009, o teste foi usado uma segunda vez, só para o
Random Forest centro congelado: CPM 0,420 (IC95% 0,349 a 0,490). O resultado da CNN acima não
mudou. Ver `docs/14-baseline-literatura-sementes-e-teste-rf.md`.
