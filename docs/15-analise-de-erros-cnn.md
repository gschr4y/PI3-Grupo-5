# Onde a rede escolhida erra no teste, 26/10/2026

Análise dos nódulos perdidos e dos falsos positivos da `cnn_lr_3e-4` nos 176 exames de teste
(subsets 8 e 9, 223 nódulos). Usamos as previsões que o avaliador oficial já recebeu em 13/10
(`modelos/cnn_lr_3e-4_teste_final/predicoes.csv`): nada foi previsto nem avaliado de novo, e nada
foi escolhido a partir destes números.

Todos os números deste documento foram medidos em 26/10/2026 com:

```bash
.venv/bin/python scripts/29_analisar_erros_cnn.py
```

As saídas ficam em `relatorios/experimentos/analise_erros_cnn/`.

## Achados, em resumo

Cada item tem a seção com a tabela e o arquivo de onde veio.

1. A sensibilidade a 4 FP cai com o tamanho: 0,804 nos nódulos de 3 a 5 mm, 0,866 nos de 5 a
   10 mm e 0,969 nos acima de 10 mm (seção 2).
2. Dos 26 nódulos perdidos a 4 FP, 21 receberam marca e 18 delas tiveram escore de 0,80 ou mais;
   ficaram abaixo do limiar porque outras marcas do mesmo exame tiveram escore maior. Só 3 não têm
   candidato na lista V2 (seção 2).
3. Nódulos a 3 mm ou menos da borda do pulmão são 35% dos perdidos e 13% dos achados (razão de
   chances 3,6, p = 0,007). Os dois maiores perdidos, de 16,8 e 23,8 mm, foram cortados pela
   máscara de pulmão sem dilatação (seção 3).
4. Os falsos positivos de maior escore ficam perto de nódulos verdadeiros: 62% dos 24 primeiros
   estão a menos de 20 mm de um nódulo anotado, contra 24% de todos os 704 (seção 4).
5. Na nossa leitura visual dos 50 falsos positivos de maior escore, o tipo mais frequente é a
   marca ao lado de um nódulo (16), seguido de lesão grande ou irregular (11), opacidade
   arredondada isolada (10) e vaso (10) (seção 5).
6. Os falsos positivos se concentram: os 10 exames com mais deles somam 27% dos 704 (seção 4).

## 1. Como separamos acerto, perdido e falso positivo

O programa oficial só grava os totais, então repetimos o casamento dele em
`src/experimentos/erros.py`: até 100 marcas por exame (escore estritamente maior que o 101º),
acerto quando a marca está a menos de um raio do centro do nódulo, marca sobre achado excluído
fora da conta, diâmetro desconhecido das excluídas valendo 10 mm.

O script para antes de qualquer tabela se o casamento não reproduzir o oficial. Reproduziu: 218
nódulos com marca, 16.072 falsos positivos e 1.193 marcas ignoradas sobre excluídas, iguais ao
`CADAnalysis.txt`, e os sete pontos da FROC iguais aos de `pontos_operacao.csv` até a sexta casa.
O mesmo vale para o Random Forest centro no mesmo teste. Os testes estão em `testes/test_erros.py`.

"Achado a 4 FP" quer dizer escore maior ou igual ao limiar que deixa no máximo 704 falsos
positivos nos 176 exames (4 por exame). Esse limiar é 0,9706 e acha 197 dos 223 nódulos, o mesmo
0,883 da FROC oficial. A 1 FP o limiar é 0,9942 e acha 164.

## 2. Nódulos perdidos por tamanho

`por_tamanho.csv`:

| Diâmetro | Nódulos | Achados a 1 FP | Achados a 4 FP | Sens. 1 FP | Sens. 4 FP | Sem marca nas 100 |
|---|---:|---:|---:|---:|---:|---:|
| 3 a 5 mm | 46 | 24 | 37 | 0,522 | 0,804 | 4 |
| 5 a 10 mm | 112 | 83 | 97 | 0,741 | 0,866 | 1 |
| acima de 10 mm | 65 | 57 | 63 | 0,877 | 0,969 | 0 |
| total | 223 | 164 | 197 | 0,735 | 0,883 | 5 |

Os intervalos de confiança destas faixas não foram calculados: com 46 e 65 nódulos por faixa, a
comparação entre elas é descritiva.

Dos 26 perdidos a 4 FP (`nodulos.csv`):

- 5 não têm marca entre as 100 de maior escore do exame. Em 3 deles a lista V2 não tem nenhum
  candidato dentro do raio, o que nenhum classificador sobre a V2 recupera.
- 21 têm marca, e 18 dessas marcas têm escore de 0,80 ou mais. A posição mediana da marca no
  próprio exame é a 18ª: o nódulo recebeu escore alto, mas outras marcas do mesmo exame
  receberam escore ainda mais alto.
- O Random Forest centro, no limiar dele de 4 FP, acha 3 dos 26 que a rede perde.

## 3. Nódulos perto da parede do pulmão

Medimos a distância de cada nódulo à borda da máscara de pulmão do desafio (`seg-lungs-LUNA16`,
com o mesmo critério `> 0` do pré-processamento), em mm, até 30 mm. Valor negativo quer dizer
centro fora da máscara.

| | Nódulos | A 3 mm ou menos da borda | Mediana da distância |
|---|---:|---:|---:|
| Achados a 4 FP | 197 | 25 (13%) | 13,0 mm |
| Perdidos a 4 FP | 26 | 9 (35%) | 6,4 mm |

Teste exato de Fisher na tabela 2 × 2 (perto ou longe da borda, achado ou perdido): razão de
chances 3,6, p = 0,007. Mann-Whitney na distância: p = 0,03. Os dois estão em `resumo.json`, chave
`parede_e_vizinhanca`.

Os dois maiores perdidos, de 16,8 e 23,8 mm, têm o centro fora da máscara (−1,1 e −8,3 mm). Na
figura `perdidos_4.png` a maior parte do nódulo aparece preta: a máscara sem dilatação (decisão
0002) cortou o nódulo junto com a parede, e a rede viu só a borda dele.

## 4. Falsos positivos

A 4 FP são 704 falsos positivos em 150 dos 176 exames (`falsos_positivos_4fp.csv`,
`falsos_positivos_por_exame.csv`). A mediana é 3 por exame, e os 10 exames com mais falsos
positivos concentram 191 deles (27%). Dos 704, 82% estão em exames que têm nódulo anotado (119
dos 176 exames têm).

Distância de cada falso positivo ao nódulo anotado mais próximo do mesmo exame:

| Falsos positivos | A menos de 20 mm de um nódulo | A menos de 3 raios do nódulo | A 3 mm ou menos da borda |
|---|---:|---:|---:|
| Os 24 de maior escore | 62% | 58% | 4% |
| Os 100 de maior escore | 50% | 37% | 5% |
| Todos os 704 | 24% | 14% | 13% |

Quanto maior o escore do falso positivo, mais perto de um nódulo verdadeiro ele está.

## 5. Falsos positivos por tipo

Olhamos os 50 falsos positivos de maior escore nas figuras e demos um tipo a cada um. É leitura
visual nossa, sem radiologista, e entra como inferência. A classificação está em
`relatorios/anotacoes/falsos_positivos_cnn_teste.csv`, e o script confere que ela corresponde aos
mesmos 50 pontos antes de usar. Os tipos:

- **Ao lado de um nódulo**: a marca cai fora do nódulo, mas o nódulo anotado aparece dentro do
  recorte de 64 mm.
- **Sobre lesão grande ou irregular**: a marca cai em cima de uma lesão grande, alongada ou
  espiculada, longe do centro anotado ou sem nódulo anotado perto.
- **Opacidade arredondada isolada**: um foco pequeno e redondo, sem nódulo anotado perto. Pode ser
  vaso cortado de frente ou nódulo que o desafio não anota; não conferimos no LIDC-IDRI.
- **Vaso**: ramificação, cruzamento ou trecho de vaso.
- **Parede do pulmão**: marca encostada na pleura.
- **Outra**: nenhuma estrutura evidente no centro.

Não separamos cicatriz: no volume de 1 mm com a janela de −1000 a 400 HU não conseguimos
distinguir cicatriz de lesão irregular com segurança.

`falsos_positivos_por_tipo.csv`:

| Tipo | Falsos positivos | Fração dos 50 | Nódulo anotado mais próximo (mediana) | Distância à borda (mediana) |
|---|---:|---:|---:|---:|
| Ao lado de um nódulo | 16 | 0,32 | 10,1 mm | 16,8 mm |
| Sobre lesão grande ou irregular | 11 | 0,22 | 14,8 mm | 18,5 mm |
| Opacidade arredondada isolada | 10 | 0,20 | 92,9 mm | 15,8 mm |
| Vaso | 10 | 0,20 | 13,5 mm | 22,8 mm |
| Parede do pulmão | 2 | 0,04 | 57,1 mm | 1,7 mm |
| Outra | 1 | 0,02 | 15,1 mm | 16,0 mm |

Quatro dos 11 casos "sobre lesão grande ou irregular" são do mesmo exame (#13, #15, #23 e #32): a
mesma lesão alongada recebeu várias marcas.

## 6. Figuras

As figuras mostram o volume pré-processado que a rede recebe (1 mm, janela de −1000 a 400 HU,
fora do pulmão em preto), num recorte de 64 mm nos três planos:

- `perdidos_1.png` a `perdidos_4.png`: os 26 perdidos, do menor para o maior, com o diâmetro
  anotado em verde. Vários dos pequenos encostam na borda da máscara e aparecem cortados por ela
  (por exemplo os de 3,4, 3,9, 4,4 e 4,9 mm em `perdidos_1.png`; leitura visual nossa).
- `falsos_positivos_1.png` a `falsos_positivos_7.png`: os 50 falsos positivos de maior escore, na
  ordem do escore, com o centro da marca numa cruz.
- `tipo_*.png`: até quatro exemplos de cada tipo da seção 5.
