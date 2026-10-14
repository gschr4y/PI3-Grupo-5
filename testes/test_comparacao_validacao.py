"""Identidade dos candidatos do Random Forest e diagnóstico de ordenação da comparação."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def carregar(arquivo, nome):
    especificacao = importlib.util.spec_from_file_location(
        nome, Path(__file__).resolve().parents[1] / "scripts" / arquivo)
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


avaliar_rf = carregar("22_avaliar_random_forest.py", "avaliar_rf")
comparar = carregar("23_comparar_validacao.py", "comparar")


def candidatos():
    return pd.DataFrame({"seriesuid": ["a", "a", "b", "c"], "coordX": [1., 2., 3., 4.],
                         "coordY": [0., 0., 0., 0.], "coordZ": [0., 0., 0., 0.],
                         "class": [1, 0, 0, 1]})


def previsoes():
    return pd.DataFrame({"candidate_id": [2, 0, 1], "seriesuid": ["b", "a", "a"],
                         "coordX": [3., 1., 2.], "coordY": [0., 0., 0.], "coordZ": [0., 0., 0.],
                         "probability": [0.1, 0.9, 0.2]})


def test_aceita_os_mesmos_candidatos_em_outra_ordem():
    avaliar_rf.conferir_candidatos(previsoes(), candidatos(), ["a", "b"])


def test_recusa_candidato_faltando_ou_deslocado():
    with pytest.raises(ValueError, match="diferem"):
        avaliar_rf.conferir_candidatos(previsoes().iloc[:2], candidatos(), ["a", "b"])
    deslocado = previsoes()
    deslocado.loc[0, "coordX"] = 3.5
    with pytest.raises(ValueError, match="coordX"):
        avaliar_rf.conferir_candidatos(deslocado, candidatos(), ["a", "b"])


def test_diagnostico_distingue_ordem_perfeita_de_invertida():
    classes = np.array([1, 0, 0, 1, 0])
    exames = np.array(["a", "a", "a", "b", "b"])
    perfeita = comparar.diagnostico(np.array([0.9, 0.1, 0.2, 0.8, 0.3]), classes, exames)
    invertida = comparar.diagnostico(np.array([0.1, 0.9, 0.8, 0.2, 0.7]), classes, exames)
    assert perfeita["separacao_positivo_negativo"] == 1.0
    assert invertida["separacao_positivo_negativo"] == 0.0
    assert perfeita["positivos_entre_100_maiores_do_exame"] == 2


def avaliacao(exames, coordX=(1., 2.)):
    return {"exames": exames,
            "protocolo": {"seed": 42, "bootstrap": 1000, "confianca": 0.95, "max_marcas_por_exame": 100},
            "predicoes": pd.DataFrame({"seriesuid": ["a", "b"], "coordX": list(coordX),
                                       "coordY": [0., 0.], "coordZ": [0., 0.]})}


def test_pareamento_recusa_exames_ou_candidatos_diferentes():
    comparar.conferir_pareamento(avaliacao(["a", "b"]), avaliacao(["a", "b"]))
    with pytest.raises(ValueError, match="exames"):
        comparar.conferir_pareamento(avaliacao(["a", "b"]), avaliacao(["a", "c"]))
    with pytest.raises(ValueError, match="candidatos"):
        comparar.conferir_pareamento(avaliacao(["a", "b"]), avaliacao(["a", "b"], coordX=(1., 2.5)))
