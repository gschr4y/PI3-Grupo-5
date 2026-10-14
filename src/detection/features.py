"""Características de intensidade extraídas de patches 3D."""

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import kurtosis

from detection import patches


COLUNAS = (
    "media",
    "desvio_padrao",
    "curtose",
    "minimo",
    "maximo",
    "percentil_25",
    "percentil_75",
)


def calcular(patch) -> dict[str, float]:
    valores = np.asarray(patch, dtype=np.float64)
    if valores.ndim != 3 or valores.size == 0:
        raise ValueError("patch precisa ser um volume 3D nao vazio")
    if not np.isfinite(valores).all():
        raise ValueError("patch contem valor ausente ou infinito")

    desvio = float(valores.std())
    curtose = 0.0 if desvio == 0.0 else float(kurtosis(valores, axis=None))
    resultado = {
        "media": float(valores.mean()),
        "desvio_padrao": desvio,
        "curtose": curtose,
        "minimo": float(valores.min()),
        "maximo": float(valores.max()),
        "percentil_25": float(np.percentile(valores, 25)),
        "percentil_75": float(np.percentile(valores, 75)),
    }
    if not np.isfinite(list(resultado.values())).all():
        raise ValueError("caracteristica ausente ou infinita")
    return resultado


def calcular_lote(cubos) -> pd.DataFrame:
    """As mesmas estatísticas de `calcular`, para vários cubos de uma vez."""
    valores = np.asarray(cubos, dtype=np.float64).reshape(len(cubos), -1)
    if valores.shape[1] == 0 or not np.isfinite(valores).all():
        raise ValueError("cubos vazios ou com valor ausente ou infinito")
    desvio = valores.std(axis=1)
    curtose = np.zeros(len(valores))
    variam = desvio > 0
    if variam.any():
        curtose[variam] = kurtosis(valores[variam], axis=1)
    p25, p75 = np.percentile(valores, [25, 75], axis=1)
    return pd.DataFrame({"media": valores.mean(axis=1), "desvio_padrao": desvio, "curtose": curtose,
                         "minimo": valores.min(axis=1), "maximo": valores.max(axis=1),
                         "percentil_25": p25, "percentil_75": p75})


def calcular_multiescala(cubos, lados) -> pd.DataFrame:
    """Estatísticas de cubos concêntricos de cada lado, recortados do centro do recorte."""
    total = cubos.shape[1]
    partes = []
    for lado in lados:
        if lado > total or (total - lado) % 2:
            raise ValueError(f"lado {lado} nao centra num recorte de {total}")
        margem = (total - lado) // 2
        centro = cubos[:, margem:margem + lado, margem:margem + lado, margem:margem + lado]
        partes.append(calcular_lote(centro).add_suffix(f"_{lado}"))
    return pd.concat(partes, axis=1)


def multiescala_do_exame(arquivo, esperado: pd.DataFrame, lados) -> pd.DataFrame:
    """Lê os recortes de um exame, confere identidade e classe contra a lista V2 e calcula a multiescala."""
    with np.load(arquivo, allow_pickle=False) as dados:
        if not np.array_equal(dados["candidate_id"], esperado.index.to_numpy()):
            raise ValueError(f"candidate_id diverge da lista V2: {arquivo}")
        if not np.array_equal(dados["classe_luna16"], esperado["class"].to_numpy()):
            raise ValueError(f"classes divergem da lista V2: {arquivo}")
        medidas = calcular_multiescala(dados["patches"], lados)
    medidas.insert(0, "candidate_id", esperado.index.to_numpy())
    medidas.insert(1, "seriesuid", esperado.seriesuid.to_numpy())
    medidas.insert(2, "classe_luna16", esperado["class"].to_numpy())
    return medidas


def selecionar_amostra(candidatos: pd.DataFrame, positivos: int, negativos: int, semente: int):
    if positivos < 0 or negativos < 0:
        raise ValueError("quantidades da amostra precisam ser nao negativas")
    classes = set(candidatos["class"].unique())
    if not classes <= {0, 1}:
        raise ValueError(f"classes inesperadas: {sorted(classes)}")

    partes = []
    for classe, quantidade in ((0, negativos), (1, positivos)):
        grupo = candidatos[candidatos["class"] == classe]
        if len(grupo) < quantidade:
            raise ValueError(
                f"amostra pediu {quantidade} candidatos da classe {classe}, mas existem {len(grupo)}"
            )
        partes.append(grupo.sample(n=quantidade, random_state=semente))

    return pd.concat(partes).sort_values("candidate_id").reset_index(drop=True)


def extrair_da_imagem(imagem, candidatos: pd.DataFrame, tamanho_patch):
    volume = np.asarray(sitk.GetArrayFromImage(imagem))
    dimensao = np.asarray(imagem.GetSize(), dtype=int)
    produzidas = []
    auditoria = []

    for candidato in candidatos.to_dict("records"):
        identificacao = {
            "candidate_id": int(candidato["candidate_id"]),
            "seriesuid": candidato["seriesuid"],
        }
        mundo = (candidato["coordX"], candidato["coordY"], candidato["coordZ"])
        if not np.isfinite(mundo).all():
            auditoria.append({**identificacao, "status": "descartado", "motivo": "coordenada_invalida"})
            continue

        try:
            centro = np.asarray(imagem.TransformPhysicalPointToIndex(mundo), dtype=int)
        except (RuntimeError, ValueError, TypeError):
            auditoria.append({**identificacao, "status": "descartado", "motivo": "conversao_coordenada"})
            continue

        if ((centro < 0) | (centro >= dimensao)).any():
            auditoria.append({**identificacao, "status": "descartado", "motivo": "centro_fora_do_volume"})
            continue

        patch = patches.extrair_patch(volume, centro, tamanho_patch)
        try:
            valores = calcular(patch)
        except ValueError as erro:
            auditoria.append({**identificacao, "status": "descartado", "motivo": str(erro)})
            continue

        produzidas.append({
            **identificacao,
            "paciente": candidato["paciente"],
            "subset": int(candidato["subset"]),
            "particao": candidato["particao"],
            "coordX": float(candidato["coordX"]),
            "coordY": float(candidato["coordY"]),
            "coordZ": float(candidato["coordZ"]),
            "classe_luna16": int(candidato["class"]),
            **valores,
        })
        auditoria.append({**identificacao, "status": "produzido", "motivo": ""})

    return pd.DataFrame(produzidas), pd.DataFrame(auditoria)
