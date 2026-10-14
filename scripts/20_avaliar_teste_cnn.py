"""Avalia um checkpoint congelado no teste, sem ajustar pesos."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from experimentos import avaliacao_oficial, cnn, rastreamento


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    args = parser.parse_args()
    if args.saida.exists():
        raise ValueError("escolha uma pasta nova para preservar avaliacoes anteriores")
    cfg = config.carregar()
    caminhos = cfg["caminhos"]
    pacote = torch.load(args.modelo, map_location="cpu", weights_only=True)
    for chave, arquivo in (("divisao_sha256", "divisao"), ("candidatos_sha256", "luna16_candidatos")):
        if pacote.get(chave) != rastreamento.sha256(caminhos[arquivo]):
            raise ValueError(f"{arquivo} difere da entrada congelada no treinamento")
    divisao = pd.read_csv(caminhos["divisao"])
    cnn.conferir_divisao(divisao)
    uids = divisao.loc[divisao.particao.eq("teste"), "seriesuid"].tolist()
    if not uids:
        raise ValueError("divisao sem pacientes de teste")
    pasta = caminhos["processado"] / "patches"
    cnn.conferir_origem(pasta, caminhos["volumes"], uids)
    parametros = pacote["parametros"]
    cnn.fixar_determinismo(pacote["seed"], parametros["threads"])
    modelo = cnn.CNN3D(parametros["dropout_rate"])
    modelo.load_state_dict(pacote["state_dict"])
    candidatos = pd.read_csv(caminhos["luna16_candidatos"])
    entradas = {"modelo": args.modelo, "divisao": caminhos["divisao"],
                "candidatos": caminhos["luna16_candidatos"],
                "anotacoes": caminhos["luna16_anotacoes"],
                "excluidas": caminhos["luna16_anotacoes_excluidas"]}
    entradas.update({f"patches/{u}": pasta / f"{u}.npz" for u in uids})
    with rastreamento.iniciar(cfg, "CNN teste final", "avaliacao_final", {
        "treino_run_id": pacote["run_id"], "seed": pacote["seed"],
        "pesos_sha256": cnn.hash_pesos(modelo),
    }, entradas) as (cliente, run_id):
        cliente.set_tag(run_id, "teste_utilizado", "true")
        previsoes = cnn.prever(modelo, candidatos, divisao, "teste", pasta, parametros, "cpu")
        metricas = avaliacao_oficial.avaliar(
            previsoes, uids, entradas["anotacoes"], entradas["excluidas"], args.saida,
            pacote["seed"], cfg["avaliacao"]["bootstrap_reamostras"],
            cfg["avaliacao"]["intervalo_confianca"],
        )
        for nome, valor in metricas.items():
            cliente.log_metric(run_id, nome, valor)
        (args.saida / "execucao.json").write_text(json.dumps({
            "run_id": run_id, "treino_run_id": pacote["run_id"],
            "teste_utilizado": True, "metricas": metricas,
        }, indent=2))
        cliente.log_artifacts(run_id, str(args.saida), artifact_path="teste_final")


if __name__ == "__main__":
    main()
