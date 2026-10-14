"""Características de intensidade e alinhamento com os candidatos."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detection import features


def test_calcula_as_sete_caracteristicas_sem_valor_invalido():
    patch = np.arange(27, dtype=np.float32).reshape(3, 3, 3)

    resultado = features.calcular(patch)

    assert tuple(resultado) == features.COLUNAS
    assert resultado["media"] == pytest.approx(13.0)
    assert resultado["minimo"] == 0.0
    assert resultado["maximo"] == 26.0
    assert np.isfinite(list(resultado.values())).all()


def test_patch_constante_recebe_curtose_zero():
    resultado = features.calcular(np.zeros((3, 3, 3), dtype=np.float32))

    assert resultado["desvio_padrao"] == 0.0
    assert resultado["curtose"] == 0.0


def test_amostra_tem_as_duas_classes_e_e_reproduzivel():
    candidatos = pd.DataFrame({
        "candidate_id": range(20),
        "class": [0] * 15 + [1] * 5,
    })

    primeira = features.selecionar_amostra(candidatos, positivos=3, negativos=4, semente=42)
    segunda = features.selecionar_amostra(candidatos, positivos=3, negativos=4, semente=42)

    assert primeira["class"].value_counts().to_dict() == {0: 4, 1: 3}
    assert primeira.candidate_id.tolist() == segunda.candidate_id.tolist()


def test_extracao_respeita_direcao_e_mantem_rotulo_alinhado():
    array = np.zeros((7, 8, 9), dtype=np.float32)
    array[3, 4, 5] = 1.0
    imagem = sitk.GetImageFromArray(array)
    imagem.SetOrigin((100.0, 50.0, -20.0))
    imagem.SetSpacing((1.0, 1.0, 1.0))
    imagem.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    mundo = imagem.TransformIndexToPhysicalPoint((5, 4, 3))
    candidatos = pd.DataFrame([{
        "candidate_id": 81,
        "seriesuid": "exame-lpi",
        "paciente": "paciente-1",
        "subset": 2,
        "particao": "treino",
        "coordX": mundo[0],
        "coordY": mundo[1],
        "coordZ": mundo[2],
        "class": 1,
    }])

    resultado, auditoria = features.extrair_da_imagem(imagem, candidatos, (3, 3, 3))

    assert resultado.candidate_id.tolist() == [81]
    assert resultado.classe_luna16.tolist() == [1]
    assert resultado.maximo.tolist() == [1.0]
    assert auditoria.status.tolist() == ["produzido"]


def test_candidato_fora_do_volume_fica_na_auditoria():
    imagem = sitk.GetImageFromArray(np.ones((5, 5, 5), dtype=np.float32))
    candidatos = pd.DataFrame([{
        "candidate_id": 9,
        "seriesuid": "exame",
        "paciente": "paciente",
        "subset": 0,
        "particao": "treino",
        "coordX": 100.0,
        "coordY": 100.0,
        "coordZ": 100.0,
        "class": 0,
    }])

    resultado, auditoria = features.extrair_da_imagem(imagem, candidatos, (3, 3, 3))

    assert resultado.empty
    assert auditoria.to_dict("records") == [{
        "candidate_id": 9,
        "seriesuid": "exame",
        "status": "descartado",
        "motivo": "centro_fora_do_volume",
    }]


def test_lote_reproduz_calcular_cubo_a_cubo():
    gerador = np.random.default_rng(3)
    cubos = gerador.random((5, 34, 34, 34), dtype=np.float32)
    cubos[2] = 0.25
    lote = features.calcular_lote(cubos)
    for i, cubo in enumerate(cubos):
        esperado = features.calcular(cubo)
        for coluna in features.COLUNAS:
            assert lote.loc[i, coluna] == pytest.approx(esperado[coluna], rel=1e-12, abs=1e-12)
    assert lote.loc[2, "curtose"] == 0.0


def test_multiescala_recorta_o_centro_de_cada_lado():
    cubos = np.zeros((1, 34, 34, 34), dtype=np.float32)
    cubos[0, 13:21, 13:21, 13:21] = 1
    medidas = features.calcular_multiescala(cubos, [8, 16, 34])
    assert medidas.loc[0, "media_8"] == 1.0
    assert medidas.loc[0, "media_16"] == pytest.approx(8 ** 3 / 16 ** 3)
    assert medidas.loc[0, "media_34"] == pytest.approx(8 ** 3 / 34 ** 3)
    with pytest.raises(ValueError, match="centra"):
        features.calcular_multiescala(cubos, [9])


def test_multiescala_do_exame_recusa_identidade_divergente(tmp_path):
    cubos = np.zeros((2, 34, 34, 34), dtype=np.float32)
    np.savez_compressed(tmp_path / "a.npz", patches=cubos, candidate_id=np.array([5, 6]),
                        classe_luna16=np.array([1, 0], dtype=np.int8))
    esperado = pd.DataFrame({"seriesuid": ["a", "a"], "class": [1, 0]}, index=[5, 6])
    medidas = features.multiescala_do_exame(tmp_path / "a.npz", esperado, [8, 34])
    assert medidas.candidate_id.tolist() == [5, 6] and medidas.classe_luna16.tolist() == [1, 0]
    with pytest.raises(ValueError, match="candidate_id"):
        features.multiescala_do_exame(tmp_path / "a.npz", esperado.set_axis([5, 7]), [8])
    with pytest.raises(ValueError, match="classes"):
        features.multiescala_do_exame(tmp_path / "a.npz", esperado.assign(**{"class": [0, 0]}), [8])
