"""Testa as hipóteses do professor para o Random Forest fraco: posição das features, decoração do treino e sinal de cada feature."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

import config
from detection.features import COLUNAS, calcular


def separacao(escores, classes):
    postos = pd.Series(escores).rank().to_numpy()
    positivos, negativos = (classes == 1).sum(), (classes == 0).sum()
    return float((postos[classes == 1].sum() - positivos * (positivos + 1) / 2) / (positivos * negativos))


def diferenca_contra_recortes(tabela, pasta, exames):
    tabela = tabela.set_index("candidate_id")
    maior, total = dict.fromkeys(COLUNAS, 0.0), 0
    for uid in exames:
        with np.load(Path(pasta) / f"{uid}.npz", allow_pickle=False) as dados:
            for identificador, cubo in zip(dados["candidate_id"], dados["patches"]):
                recalculado = calcular(cubo)
                for coluna in COLUNAS:
                    maior[coluna] = max(maior[coluna], abs(recalculado[coluna] - tabela.loc[identificador, coluna]))
                total += 1
    return total, maior


def main():
    cfg = config.carregar()
    config.fixar_semente(cfg["seed"])
    caminhos = cfg["caminhos"]
    treino = pd.read_csv(caminhos["caracteristicas_treino"])
    validacao = pd.read_csv(caminhos["caracteristicas_validacao"])
    modelo = joblib.load(caminhos["random_forest"] / "modelo.joblib")["modelo"]
    positiva = int(np.flatnonzero(modelo.classes_ == 1)[0])

    exames = sorted(validacao.seriesuid.unique())[:3]
    total, maior = diferenca_contra_recortes(validacao, caminhos["processado"] / "patches", exames)
    classes = validacao.classe_luna16.to_numpy()
    resultado = {
        "features_recalculadas_dos_recortes": {"exames": len(exames), "candidatos": total,
                                               "diferenca_maxima": maior},
        "separacao_rf_treino": separacao(
            modelo.predict_proba(treino[list(COLUNAS)])[:, positiva], treino.classe_luna16.to_numpy()),
        "separacao_rf_validacao": separacao(
            modelo.predict_proba(validacao[list(COLUNAS)])[:, positiva], classes),
        "separacao_por_feature_validacao": {c: separacao(validacao[c].to_numpy(), classes) for c in COLUNAS},
        "teste_utilizado": False,
    }
    destino = caminhos["diagnostico_random_forest"]
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "diagnostico.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in resultado.items() if k.startswith("separacao_rf")}, indent=2))


if __name__ == "__main__":
    main()
