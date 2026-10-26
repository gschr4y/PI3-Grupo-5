"""Segundo uso do teste só com o Random Forest congelado, FROC conjunta e resumo das sementes."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]


def carregar(arquivo, nome):
    spec = importlib.util.spec_from_file_location(nome, RAIZ / "scripts" / arquivo)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


teste_rf = carregar("26_avaliar_teste_random_forest.py", "teste_rf")
froc = carregar("27_comparar_froc.py", "froc")
sementes = carregar("28_resumir_sementes_cnn.py", "sementes")


def test_teste_do_rf_recusa_modelo_diferente_do_congelado(tmp_path):
    outro = tmp_path / "modelo.joblib"
    outro.write_bytes(b"outro modelo")
    with pytest.raises(ValueError, match="modelo difere"):
        teste_rf.conferir_congelado({"divisao": outro, "luna16_candidatos": outro}, outro)


def test_teste_do_rf_confere_a_divisao_depois_do_modelo_versionado(tmp_path):
    divisao = tmp_path / "divisao.csv"
    divisao.write_text("seriesuid,paciente,subset,particao\n")
    with pytest.raises(ValueError, match="divisao difere"):
        teste_rf.conferir_congelado({"divisao": divisao, "luna16_candidatos": divisao},
                                    RAIZ / "modelos/random_forest_centro/modelo.joblib")


def test_froc_conjunta_reproduz_a_comparacao_registrada_na_validacao():
    pastas = [RAIZ / "modelos/cnn_lr_3e-4/avaliacao_validacao",
              RAIZ / "modelos/random_forest_centro/avaliacao_validacao"]
    dados = [froc.carregar(p) for p in pastas]
    froc.comparacao_validacao().conferir_pareamento(*dados)
    modelos, par, tabela = froc.resumir(("CNN", "RF"), dados, 0.95)
    assert modelos["CNN"]["cpm"] == pytest.approx(0.7516, abs=1e-4)
    assert modelos["RF"]["cpm"] == pytest.approx(0.4093, abs=1e-4)
    # Mesmos números de relatorios/experimentos/comparacao_validacao/pares.csv.
    assert par["diferenca_cpm"] == pytest.approx(0.342, abs=1e-3)
    assert par["diferenca_ic"] == pytest.approx([0.252, 0.437], abs=1e-3)
    assert len(tabela) == 14 and [froc.nodulos(p) for p in pastas] == [111, 111]


def recibo(seed, **mudancas):
    parametros = {"learning_rate": 0.0003, "epochs": 40, "seed": seed, "gpu": "x", **mudancas}
    if seed != 42:
        parametros["seed_avaliacao"] = 42
    return {"parametros": parametros, "dados_sha256": "d", "teste_utilizado": False, "run_id": str(seed),
            "metricas": {"validacao_cpm": 0.7, "validacao_cpm_ic_inferior": 0.6, "validacao_cpm_ic_superior": 0.8,
                         **{f"validacao_sensibilidade_{str(fp).replace('.', '_')}fp": 0.5
                            for fp in sementes.PONTOS}}}


def test_sementes_aceitam_so_a_semente_diferente():
    sementes.conferir_configuracao([recibo(42), recibo(43, gpu="y"), recibo(44)])
    with pytest.raises(ValueError, match="learning_rate"):
        sementes.conferir_configuracao([recibo(42), recibo(43, learning_rate=0.001)])
    with pytest.raises(ValueError, match="repetidas"):
        sementes.conferir_configuracao([recibo(42), recibo(42)])


def test_resumo_das_sementes_usa_desvio_amostral():
    tabela = pd.DataFrame({"cpm": [0.70, 0.74, 0.78], "sens_4": [0.8, 0.9, 1.0]})
    resumo = sementes.estatisticas(tabela)
    assert resumo["cpm_medio"] == pytest.approx(0.74)
    assert resumo["cpm_desvio_padrao_amostral"] == pytest.approx(0.04)
    assert (resumo["cpm_minimo"], resumo["cpm_maximo"]) == (0.70, 0.78)
    assert resumo["cpm_amplitude"] == pytest.approx(0.08)


def test_linha_da_semente_registra_checkpoint_da_ultima_epoca():
    historico = pd.DataFrame({"epoch": [1, 2, 40], "train_loss": [0.5, 0.3, 0.07]})
    saida = sementes.linha(recibo(43), historico)
    assert saida["epoca_checkpoint"] == 40 and saida["loss_ultima_epoca"] == 0.07
    assert saida["sens_0.125"] == 0.5 and saida["sens_8"] == 0.5
