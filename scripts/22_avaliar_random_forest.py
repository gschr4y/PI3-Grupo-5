"""Avalia na validação, com o programa oficial LUNA16, os escores já salvos do Random Forest."""

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

import config
from experimentos import avaliacao_oficial, rastreamento


def conferir_candidatos(previsoes, candidatos, exames):
    esperado = candidatos.loc[candidatos.seriesuid.isin(exames)]
    if previsoes.candidate_id.duplicated().any() or set(previsoes.candidate_id) != set(esperado.index):
        raise ValueError("candidatos do RF diferem da lista V2 da validacao")
    alinhado = esperado.loc[previsoes.candidate_id]
    if not np.array_equal(alinhado.seriesuid.to_numpy(), previsoes.seriesuid.to_numpy()):
        raise ValueError("seriesuid do RF diverge da lista V2")
    for eixo in ("coordX", "coordY", "coordZ"):
        if not np.allclose(alinhado[eixo].to_numpy(), previsoes[eixo].to_numpy(), rtol=0, atol=1e-4):
            raise ValueError(f"{eixo} do RF diverge da lista V2")


def main():
    cfg = config.carregar()
    config.fixar_semente(cfg["seed"])
    caminhos = cfg["caminhos"]
    origem = caminhos["random_forest"]
    destino = caminhos["random_forest_validacao"]
    if destino.exists():
        raise ValueError(f"preserve a avaliação existente e escolha uma pasta nova: {destino}")
    treino = json.loads((origem / "execucao.json").read_text(encoding="utf-8"))
    previsoes = pd.read_csv(origem / "probabilidades_validacao.csv")
    divisao = pd.read_csv(caminhos["divisao"])
    exames = sorted(divisao.loc[divisao.particao.eq("validacao"), "seriesuid"])
    candidatos = pd.read_csv(caminhos["luna16_candidatos"])
    conferir_candidatos(previsoes, candidatos, exames)
    previsoes["class"] = candidatos.loc[previsoes.candidate_id, "class"].to_numpy()

    entradas = {"probabilidades": origem / "probabilidades_validacao.csv",
                "execucao_treino": origem / "execucao.json", "divisao": caminhos["divisao"],
                "candidatos": caminhos["luna16_candidatos"], "anotacoes": caminhos["luna16_anotacoes"],
                "excluidas": caminhos["luna16_anotacoes_excluidas"]}
    for nome in avaliacao_oficial.FONTES:
        entradas[f"avaliador/{nome}"] = avaliacao_oficial.ORIGEM / nome
    avaliacao = cfg["avaliacao"]
    registro = {"treino_run_id": treino["run_id"], "seed": cfg["seed"], "escopo": "validacao",
                "bootstrap": avaliacao["bootstrap_reamostras"],
                "confianca": avaliacao["intervalo_confianca"], "max_marcas_oficial": 100}
    destino.mkdir(parents=True)
    with rastreamento.iniciar(cfg, "Random Forest V2 validacao oficial", "avaliacao_validacao",
                             registro, entradas) as (cliente, run_id):
        cliente.set_tag(run_id, "teste_utilizado", "false")
        cliente.set_tag(run_id, "avaliacao", "LUNA16_oficial_python3")
        cliente.set_tag(run_id, "treino_run_id", treino["run_id"])
        print(f"MLflow: {run_id}; {len(previsoes)} candidatos de {len(exames)} exames", flush=True)
        metricas = {"validacao_exames": len(exames), "validacao_candidatos": len(previsoes),
                    "validacao_positivos": int(previsoes["class"].sum())}
        medidas = avaliacao_oficial.avaliar(
            previsoes, exames, caminhos["luna16_anotacoes"], caminhos["luna16_anotacoes_excluidas"],
            destino / "avaliacao_validacao", cfg["seed"], avaliacao["bootstrap_reamostras"],
            avaliacao["intervalo_confianca"])
        metricas.update({f"validacao_{k}": float(v) for k, v in medidas.items()})
        for nome, valor in metricas.items():
            cliente.log_metric(run_id, nome, float(valor))
        recibo = {"run_id": run_id, "treino_run_id": treino["run_id"], "parametros": registro,
                  "metricas": metricas, "teste_utilizado": False, "avaliacao_froc": "oficial"}
        (destino / "execucao.json").write_text(json.dumps(recibo, indent=2), encoding="utf-8")
        cliente.log_artifacts(run_id, str(destino), artifact_path="avaliacao_rf")
    print(json.dumps({"run_id": run_id, "validacao_cpm": metricas["validacao_cpm"],
                      "saida": str(destino)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
