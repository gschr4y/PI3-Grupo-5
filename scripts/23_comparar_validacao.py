"""Compara modelos na validação: FROC oficial, diferença pareada de CPM contra a referência e diagnóstico de ordenação."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def carregar(pasta):
    avaliacao = pasta / "avaliacao_validacao"
    return {"protocolo": json.loads((avaliacao / "avaliacao.json").read_text(encoding="utf-8")),
            "exames": (avaliacao / "seriesuids.csv").read_text(encoding="utf-8").split(),
            "pontos": pd.read_csv(avaliacao / "pontos_operacao.csv"),
            "curva": pd.read_csv(avaliacao / "curva_froc_completa.csv"),
            "replicas": np.loadtxt(avaliacao / "cpm_bootstrap.csv", delimiter=","),
            "predicoes": pd.read_csv(avaliacao / "predicoes.csv")}


def conferir_pareamento(a, b):
    if a["exames"] != b["exames"]:
        raise ValueError("as avaliacoes usam exames diferentes")
    for chave in ("seed", "bootstrap", "confianca", "max_marcas_por_exame"):
        if a["protocolo"][chave] != b["protocolo"][chave]:
            raise ValueError(f"protocolo diverge em {chave}")
    colunas = ["seriesuid", "coordX", "coordY", "coordZ"]
    candidatos_a = a["predicoes"][colunas].round(4).sort_values(colunas).to_numpy()
    candidatos_b = b["predicoes"][colunas].round(4).sort_values(colunas).to_numpy()
    if candidatos_a.shape != candidatos_b.shape or not (candidatos_a == candidatos_b).all():
        raise ValueError("as avaliacoes usam candidatos diferentes")


def diagnostico(probabilidades, classes, exames):
    positivos, negativos = probabilidades[classes == 1], probabilidades[classes == 0]
    # Chance de um candidato positivo sorteado ter escore maior que um negativo sorteado.
    postos = pd.Series(probabilidades).rank().to_numpy()
    separacao = (postos[classes == 1].sum() - len(positivos) * (len(positivos) + 1) / 2) / (
        len(positivos) * len(negativos))
    ordem = pd.DataFrame({"exame": exames, "p": probabilidades, "classe": classes})
    ordem["posto"] = ordem.groupby("exame").p.rank(ascending=False, method="first")
    return {"separacao_positivo_negativo": float(separacao),
            "mediana_positivos": float(np.median(positivos)),
            "mediana_negativos": float(np.median(negativos)),
            "positivos": int(len(positivos)),
            "positivos_entre_100_maiores_do_exame": int(ordem.loc[ordem.classe.eq(1), "posto"].le(100).sum())}


CORES = ("#1f5f8b", "#b5651d", "#3b8b4f", "#8b3b7a", "#6b6b6b", "#c2a33b", "#2f8b8b")


def intervalo(valores, confianca):
    cauda = (1 - confianca) / 2
    return [float(np.quantile(valores, cauda)), float(np.quantile(valores, 1 - cauda))]


def main():
    cfg = config.carregar()
    config.fixar_semente(cfg["seed"])
    caminhos = cfg["caminhos"]
    referencia = cfg["comparacao"]["referencia"]
    modelos = cfg["comparacao"]["modelos"]
    if referencia not in modelos:
        raise ValueError(f"referencia ausente da lista de modelos: {referencia}")
    dados = {nome: carregar(config.RAIZ / m["avaliacao"]) for nome, m in modelos.items()}
    for nome in modelos:
        conferir_pareamento(dados[referencia], dados[nome])
    destino = caminhos["comparacao_validacao"]
    destino.mkdir(parents=True, exist_ok=True)

    candidatos = pd.read_csv(caminhos["luna16_candidatos"])
    confianca = cfg["avaliacao"]["intervalo_confianca"]
    resumo = {"referencia": referencia, "exames": len(dados[referencia]["exames"]),
              "reamostras": len(dados[referencia]["replicas"]), "modelos": {}, "teste_utilizado": False}
    linhas, pontos = [], []
    for nome, modelo in modelos.items():
        d = dados[nome]
        tabela = pd.read_csv(config.RAIZ / modelo["probabilidades"])
        classes = candidatos.loc[tabela.candidate_id, "class"].to_numpy()
        diferenca = d["replicas"] - dados[referencia]["replicas"]
        cpm = float(d["pontos"].sensibilidade.mean())
        sensibilidade = d["pontos"].set_index("fp_por_exame").sensibilidade
        resumo["modelos"][nome] = {
            "rotulo": modelo["rotulo"], "cpm": cpm, "cpm_ic": intervalo(d["replicas"], confianca),
            "diferenca_cpm_contra_referencia": cpm - float(dados[referencia]["pontos"].sensibilidade.mean()),
            "diferenca_cpm_ic": intervalo(diferenca, confianca),
            "fracao_reamostras_acima_da_referencia": float((diferenca > 0).mean()),
            "diagnostico": diagnostico(tabela.probability.to_numpy(), classes, tabela.seriesuid.to_numpy()),
        }
        linhas.append({"modelo": nome, "cpm": cpm, "cpm_ic_inferior": resumo["modelos"][nome]["cpm_ic"][0],
                       "cpm_ic_superior": resumo["modelos"][nome]["cpm_ic"][1],
                       "sensibilidade_1fp": float(sensibilidade[1.0]), "sensibilidade_4fp": float(sensibilidade[4.0]),
                       "diferenca_cpm": resumo["modelos"][nome]["diferenca_cpm_contra_referencia"],
                       "diferenca_ic_inferior": resumo["modelos"][nome]["diferenca_cpm_ic"][0],
                       "diferenca_ic_superior": resumo["modelos"][nome]["diferenca_cpm_ic"][1]})
        pontos.append(d["pontos"][["fp_por_exame", "sensibilidade", "ic_inferior", "ic_superior"]].assign(modelo=nome))
    pares = []
    for a, b in cfg["comparacao"].get("pares", []):
        diferenca = dados[a]["replicas"] - dados[b]["replicas"]
        pares.append({"modelo": a, "contra": b,
                      "diferenca_cpm": resumo["modelos"][a]["cpm"] - resumo["modelos"][b]["cpm"],
                      "ic_inferior": intervalo(diferenca, confianca)[0],
                      "ic_superior": intervalo(diferenca, confianca)[1],
                      "fracao_reamostras_maior": float((diferenca > 0).mean())})
    resumo["pares"] = pares
    pd.DataFrame(pares).to_csv(destino / "pares.csv", index=False)
    (destino / "comparacao.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")
    pd.DataFrame(linhas).to_csv(destino / "resumo.csv", index=False)
    pd.concat(pontos)[["modelo", "fp_por_exame", "sensibilidade", "ic_inferior", "ic_superior"]].to_csv(
        destino / "pontos_operacao.csv", index=False)

    figura, eixo = plt.subplots(figsize=(11, 4.8))
    for (nome, modelo), cor in zip(modelos.items(), CORES):
        d = dados[nome]
        eixo.plot(d["curva"].fp_por_exame, d["curva"].sensibilidade, color=cor,
                  label=f"{modelo['rotulo']} (CPM {resumo['modelos'][nome]['cpm']:.3f})".replace(".", ","))
        eixo.fill_between(d["pontos"].fp_por_exame, d["pontos"].ic_inferior, d["pontos"].ic_superior,
                          color=cor, alpha=0.12, linewidth=0)
    eixo.set_xscale("log", base=2)
    eixo.set_xlim(0.125, 8)
    eixo.set_ylim(0, 1)
    eixo.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8], ["0,125", "0,25", "0,5", "1", "2", "4", "8"])
    eixo.set_xlabel("Falsos positivos por exame")
    eixo.set_ylabel("Sensibilidade")
    eixo.set_title(f"Validação, {resumo['exames']} exames, avaliador oficial LUNA16")
    eixo.grid(alpha=0.3)
    eixo.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False)
    figura.tight_layout()
    figura.savefig(destino / "froc_validacao.png", dpi=150)
    print(pd.DataFrame(linhas).round(3).to_string(index=False), flush=True)
    print(pd.DataFrame(pares).round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
