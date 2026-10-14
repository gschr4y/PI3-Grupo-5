"""Treina o Random Forest com estatísticas de cubos concêntricos e avalia na validação com o programa oficial."""

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

import config
from detection.features import multiescala_do_exame
from experimentos import avaliacao_oficial, rastreamento


def caracteristicas(divisao, candidatos, pasta, particao, lados, processos=1):
    uids = sorted(divisao.loc[divisao.particao.eq(particao), "seriesuid"])
    grupos = [candidatos.loc[candidatos.seriesuid.eq(uid)] for uid in uids]
    with ProcessPoolExecutor(max_workers=processos) as executor:
        arquivos = [Path(pasta) / f"{uid}.npz" for uid in uids]
        blocos = list(executor.map(multiescala_do_exame, arquivos, grupos, repeat(lados)))
    return pd.concat(blocos, ignore_index=True)


def main():
    cfg = config.carregar()
    config.fixar_semente(cfg["seed"])
    caminhos = cfg["caminhos"]
    destino = caminhos["random_forest_centro"]
    if destino.exists():
        raise ValueError(f"preserve a execução existente e escolha uma pasta nova: {destino}")
    lados = cfg["random_forest_centro"]["lados"]
    processos = cfg["random_forest_centro"]["processos"]
    divisao = pd.read_csv(caminhos["divisao"])
    candidatos = pd.read_csv(caminhos["luna16_candidatos"])
    pasta = caminhos["processado"] / "patches"
    tabelas = caminhos["caracteristicas_centro"]
    tabelas.mkdir(parents=True, exist_ok=True)

    inicio = perf_counter()
    treino = caracteristicas(divisao, candidatos, pasta, "treino", lados, processos)
    validacao = caracteristicas(divisao, candidatos, pasta, "validacao", lados, processos)
    segundos_features = perf_counter() - inicio
    if set(treino.seriesuid) & set(validacao.seriesuid):
        raise ValueError("exame compartilhado entre treino e validacao")
    treino.to_csv(tabelas / "treino.csv", index=False)
    validacao.to_csv(tabelas / "validacao.csv", index=False)
    colunas = [c for c in treino.columns if c not in ("candidate_id", "seriesuid", "classe_luna16")]

    parametros = dict(cfg["random_forest"])
    avaliacao = cfg["avaliacao"]
    registro = {**parametros, "seed": cfg["seed"], "lados": ",".join(map(str, lados)),
                "features": len(colunas), "lista": "candidates_V2", "sklearn": sklearn.__version__,
                "origem_features": "recortes_npz", "bootstrap": avaliacao["bootstrap_reamostras"]}
    entradas = {"divisao": caminhos["divisao"], "candidatos": caminhos["luna16_candidatos"],
                "anotacoes": caminhos["luna16_anotacoes"], "excluidas": caminhos["luna16_anotacoes_excluidas"],
                "caracteristicas_treino": tabelas / "treino.csv",
                "caracteristicas_validacao": tabelas / "validacao.csv"}
    destino.mkdir(parents=True)
    with rastreamento.iniciar(cfg, "Random Forest centro", "treinamento", registro,
                             entradas) as (cliente, run_id):
        cliente.set_tag(run_id, "teste_utilizado", "false")
        print(f"MLflow: {run_id}; {len(treino)} candidatos de treino, {len(colunas)} features", flush=True)
        inicio = perf_counter()
        modelo = RandomForestClassifier(random_state=cfg["seed"], **parametros)
        modelo.fit(treino[colunas], treino.classe_luna16)
        segundos_treino = perf_counter() - inicio
        positiva = int(np.flatnonzero(modelo.classes_ == 1)[0])
        previsoes = validacao[["candidate_id", "seriesuid"]].copy()
        posicao = candidatos.loc[validacao.candidate_id, ["coordX", "coordY", "coordZ", "class"]]
        previsoes[["coordX", "coordY", "coordZ", "class"]] = posicao.to_numpy()
        previsoes["probability"] = modelo.predict_proba(validacao[colunas])[:, positiva]
        joblib.dump({"modelo": modelo, "features": colunas, "run_id": run_id, "parametros": registro},
                    destino / "modelo.joblib", compress=3)
        previsoes.drop(columns="class").to_csv(destino / "probabilidades_validacao.csv", index=False)

        exames = sorted(divisao.loc[divisao.particao.eq("validacao"), "seriesuid"])
        metricas = {"segundos_features": segundos_features, "treino_segundos": segundos_treino,
                    "treino_candidatos": len(treino), "treino_positivos": int(treino.classe_luna16.sum()),
                    "validacao_candidatos": len(previsoes)}
        medidas = avaliacao_oficial.avaliar(
            previsoes, exames, caminhos["luna16_anotacoes"], caminhos["luna16_anotacoes_excluidas"],
            destino / "avaliacao_validacao", cfg["seed"], avaliacao["bootstrap_reamostras"],
            avaliacao["intervalo_confianca"])
        metricas.update({f"validacao_{k}": float(v) for k, v in medidas.items()})
        for nome, valor in metricas.items():
            cliente.log_metric(run_id, nome, float(valor))
        recibo = {"run_id": run_id, "parametros": registro, "metricas": metricas,
                  "teste_utilizado": False, "avaliacao_froc": "oficial"}
        (destino / "execucao.json").write_text(json.dumps(recibo, indent=2), encoding="utf-8")
        cliente.log_artifacts(run_id, str(destino), artifact_path="random_forest_centro")
    print(json.dumps({"run_id": run_id, "validacao_cpm": metricas["validacao_cpm"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
