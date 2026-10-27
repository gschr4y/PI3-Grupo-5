# Comparação: FROC própria × avaliador oficial do LUNA16
 
**Modelo:** `cnn_lr_3e-4` · **Conjunto:** validação (89 exames) · **Previsões:** `modelos/cnn_lr_3e-4/probabilidades_validacao.csv`
**Avaliador oficial:** `avaliacao_validacao/pontos_operacao.csv` (script oficial portado para Python 3)
**Própria:** `compute_froc` / `compute_cpm` de `notebooks/curvafroc_pesquisa.ipynb`, rodadas nas mesmas previsões.
 
## Resultado: NÃO dão o mesmo número
 
| FP/exame | Própria | Oficial | Diferença |
|---:|---:|---:|---:|
| 0,125 | 0,3344 | 0,5045 | −0,170 |
| 0,25 | 0,4500 | 0,6306 | −0,181 |
| 0,5 | 0,5250 | 0,7297 | −0,205 |
| 1 | 0,6250 | 0,7838 | −0,159 |
| 2 | 0,7500 | 0,8378 | −0,088 |
| 4 | 0,8333 | 0,8829 | −0,050 |
| 8 | 0,8917 | 0,8919 | −0,000 |
| **CPM** | **0,6299** | **0,7516** | **−0,122** |
 
A própria subestima o desempenho, principalmente com poucos falsos positivos (até 1 FP/exame). Com 8 FP/exame as duas coincidem.
 
## De onde vem a diferença
 
Contagens do `CADAnalysis.txt` do avaliador oficial: 111 nódulos, 547 candidatos ignorados em achados excluídos, 11 detecções duplas ignoradas, 100 marcas por exame (8.900 no total).
 
1. **Critério de acerto.** A própria usa o rótulo `class` do candidato: há 120 candidatos positivos. O oficial conta nódulos: 111, cada um acertado uma vez. Os 9 candidatos a mais são detecções duplas do mesmo nódulo, que a própria conta como acertos extras e o oficial ignora.
2. **Marcações excluídas.** O oficial não conta como falso positivo os 547 candidatos que caem em achados de `annotations_excluded.csv`. A própria não conhece essa lista e conta todos como FP. Isso infla o FP/exame e desloca a curva para a direita, e explica a maior parte da diferença nos pontos de poucos FP.
3. **Limite de 100 marcas por exame.** O oficial mantém só as 100 marcas de maior probabilidade por exame; a própria usa todas (média de 843 por exame).
 
## Quanto cada fator pesa (medido nas mesmas previsões)
 
| Variante da própria | CPM |
|---|---:|
| Como está no notebook | 0,6299 |
| + limite de 100 marcas por exame | 0,6406 |
| + denominador de 111 nódulos (como o oficial) | 0,6810 |
| Oficial | 0,7516 |
 
Limite de 100 marcas e denominador explicam cerca de 0,05 dos 0,12 de diferença. O restante (~0,07) é atribuído às marcações excluídas e às duplas, que não consegui isolar numericamente porque `annotations_excluded.csv` e `annotations.csv` estão só no HD. Essa parte é inferida das contagens do `CADAnalysis.txt`, não medida.
 
## Nota sobre a quarta casa decimal
 
Os valores da própria podem variar na quarta casa (ex.: 0,6298 ou 0,6299) conforme a versão do numpy, porque vários limiares empatam em FP/exame e a interpolação depende da ordem dos empates. O avaliador oficial é determinístico. Por isso a comparação é reportada com três casas.
 
## Como reproduzir
 
`python scripts/verificar_comparacao_froc.py` (baixa as previsões e o `pontos_operacao.csv` do repositório, refaz a conta e confere os números).
 
## Conclusão
 
Para qualquer número reportado como comparável ao LUNA16, vale o avaliador oficial (CPM 0,7516, IC95% 0,681–0,819). A implementação própria serve só para iterar rápido e não deve ser citada como resultado final.
