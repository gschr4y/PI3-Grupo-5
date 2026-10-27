"""Refaz a comparação FROC própria x avaliador oficial e confere os números.
Uso:  python verificar_comparacao.py        (precisa de pandas, numpy e internet)
"""
import numpy as np, pandas as pd
 
B = "https://raw.githubusercontent.com/kauanLDD/PI3-Grupo-5/main/modelos/cnn_lr_3e-4"
PTS = (0.125, 0.25, 0.5, 1, 2, 4, 8)
 
prob = pd.read_csv(f"{B}/probabilidades_validacao.csv")
oficial = pd.read_csv(f"{B}/avaliacao_validacao/pontos_operacao.csv")
n_exames = prob.seriesuid.nunique()
 
def froc(df, n_nodulos=None):
    """Mesma lógica do compute_froc/compute_cpm do notebook (limiar = cada score)."""
    y, s = df["class"].values, df.probability.values
    den = n_nodulos or int(y.sum())
    ths = np.concatenate([np.unique(s), [s.max() + 1e-6]])
    sens, fp = [], []
    for t in ths:
        p = s >= t
        sens.append((p & (y == 1)).sum() / den)
        fp.append((p & (y == 0)).sum() / n_exames)
    o = np.argsort(fp)
    fp, sens = np.array(fp)[o], np.array(sens)[o]
    v = [float(np.interp(x, fp, sens)) for x in PTS]
    return v, float(np.mean(v))
 
propria, cpm_p = froc(prob)
top100 = prob.sort_values("probability", ascending=False).groupby("seriesuid").head(100)
p100, cpm_100 = froc(top100)
p111, cpm_111 = froc(top100, 111)
of = oficial.sensibilidade.values
 
tab = pd.DataFrame({"FP/exame": PTS, "propria": propria, "+top100": p100,
                    "+den111": p111, "oficial": of})
tab.loc[len(tab)] = ["CPM", cpm_p, cpm_100, cpm_111, of.mean()]
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(tab.to_string(index=False))
 
# ---- conferências: cada linha deve imprimir OK ----
def ok(nome, cond): print(("OK   " if cond else "FALHA"), nome)
ok("89 exames na validação", n_exames == 89)
ok("120 candidatos com class=1", int(prob["class"].sum()) == 120)
ok("oficial: CPM = média dos 7 pontos = 0,7516", abs(of.mean() - 0.7516) < 1e-3)
ok("oficial: 0,125 FP -> 56 de 111 nódulos (0,5045)", abs(56 / 111 - of[0]) < 1e-6)
ok("própria CPM = 0,6299", abs(cpm_p - 0.6299) < 1e-3)
ok("própria difere do oficial (não são iguais)", abs(cpm_p - of.mean()) > 0.05)
tab.to_csv("comparacao_propria_vs_oficial.csv", index=False)
print("\nSalvo: comparacao_propria_vs_oficial.csv")
