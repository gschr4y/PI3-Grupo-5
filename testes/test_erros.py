"""O casamento de marcas e nódulos repete as regras do avaliador oficial do LUNA16."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experimentos import erros


def achados(*linhas):
    return pd.DataFrame([{"seriesuid": u, "coordX": x, "coordY": 0.0, "coordZ": 0.0, "diameter_mm": d}
                         for u, x, d in linhas],
                        columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])


def marcas(*linhas):
    return pd.DataFrame([{"seriesuid": u, "coordX": x, "coordY": 0.0, "coordZ": 0.0, "probability": p}
                         for u, x, p in linhas])


def test_marca_dentro_do_raio_acerta_e_fora_e_falso_positivo():
    m, n = erros.casar(marcas(("a", 4.0, 0.9), ("a", 30.0, 0.8)), achados(("a", 0.0, 10.0)), achados())
    assert m.tipo.tolist() == ["acerto", "falso_positivo"]
    assert n.escore.tolist() == [0.9]


def test_a_borda_do_raio_nao_acerta():
    """O programa oficial compara com `<`: a marca exatamente no raio vira falso positivo."""
    m, n = erros.casar(marcas(("a", 5.0, 0.9)), achados(("a", 0.0, 10.0)), achados())
    assert m.tipo.tolist() == ["falso_positivo"]
    assert np.isnan(n.escore[0])


def test_marca_sobre_excluida_sai_da_conta_e_diametro_negativo_vale_10_mm():
    m, _ = erros.casar(marcas(("a", 4.0, 0.9), ("a", 6.0, 0.8)), achados(("a", 50.0, 10.0)),
                       achados(("a", 0.0, -1.0)))
    assert m.tipo.tolist() == ["excluida", "falso_positivo"]


def test_nodulo_guarda_a_maior_das_marcas_duplas():
    m, n = erros.casar(marcas(("a", 1.0, 0.3), ("a", -1.0, 0.7)), achados(("a", 0.0, 10.0)), achados())
    assert (m.tipo == "acerto").all()
    assert n.escore[0] == 0.7 and n.marcas[0] == 2


def test_marca_de_outro_exame_nao_acerta():
    m, n = erros.casar(marcas(("b", 0.0, 0.9)), achados(("a", 0.0, 10.0)), achados())
    assert m.tipo.tolist() == ["falso_positivo"]
    assert np.isnan(n.escore[0])


def test_limite_de_marcas_corta_os_empates_no_101():
    """O avaliador guarda só escores estritamente acima do 101º; o empate derruba os dois."""
    escores = [1.0 - i / 1000 for i in range(99)] + [0.5, 0.5, 0.5]
    previsoes = marcas(*[("a", float(i), p) for i, p in enumerate(escores)])
    assert len(erros.marcas_por_exame(previsoes)) == 99


def test_limiar_respeita_o_maximo_de_falsos_positivos():
    m = marcas(*[("a", 100.0 + i, p) for i, p in enumerate([0.9, 0.8, 0.7, 0.6])]).assign(tipo="falso_positivo")
    corte = erros.limiar(m, 1.0, 2)
    assert ((m.probability >= corte).sum()) == 2
