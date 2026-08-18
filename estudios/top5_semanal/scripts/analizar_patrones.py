"""¿Hay patrón en las que lo hacen bien? Con las cautelas estadísticas del caso.

Tres análisis, del más fiable al más frágil:

1. **Retorno por puesto dentro del ranking.** Si el score ordena, debería verse
   pendiente. Es lo más directo: no hay que elegir ninguna métrica.
2. **Information coefficient por métrica.** Correlación de rangos entre el
   percentil de cada métrica y el retorno posterior, calculada DENTRO de cada
   fecha y promediada. Calcularla sobre todo mezclado saldría inflada por los
   efectos de calendario (si en 2025 subió el oro, cualquier métrica que
   correlacione con mineras parecería predictiva).
3. **¿Sobrevive dentro del sector?** Si el patrón desaparece al comparar solo
   valores del mismo sector, lo que se había medido era el sector.

Cautelas aplicadas:
- Las 136 fechas semanales con horizontes de 3-6 meses **no son independientes**.
  El error estándar se calcula sobre bloques trimestrales, no sobre las 136.
- 20 métricas × 4 horizontes = 80 contrastes: se marca cuáles sobrevivirían a la
  corrección de Bonferroni, y se exige consistencia entre horizontes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from screener.metrics import REGISTRY

AQUI = Path(__file__).resolve().parent.parent
HOR = ["fwd_21", "fwd_63", "fwd_126", "fwd_hoy"]
REGION = sys.argv[1] if len(sys.argv) > 1 else "us"
SUFIJO = "" if REGION == "us" else f"_{REGION}"


def ic_por_fecha(d, metrica, col, minimo=30):
    """Correlación de Spearman dentro de cada fecha.

    Se calcula como Pearson sobre los rangos, que es la definición de Spearman:
    evita depender de scipy y da exactamente lo mismo.
    """
    out = {}
    for fecha, g in d.groupby("fecha"):
        g = g[[metrica, col]].dropna()
        if len(g) < minimo or g[metrica].nunique() < 5:
            continue
        out[fecha] = g[metrica].rank().corr(g[col].rank())
    return pd.Series(out).dropna()


def resumen_ic(ics):
    """Media, % de fechas positivas y t sobre bloques trimestrales."""
    if len(ics) < 8:
        return None
    por_trim = ics.groupby(pd.Series(ics.index).dt.to_period("Q").values).mean()
    n = len(por_trim)
    sd = por_trim.std(ddof=1)
    se = sd / np.sqrt(n) if n > 1 and sd > 0 else np.nan
    # `ic` es la media sobre fechas y `t` sale de la media por trimestres: los
    # trimestres no tienen el mismo número de fechas, así que ambas medias
    # difieren. Por eso se guarda `se` explícitamente — deducirlo como ic/t da
    # disparates cuando el IC ronda cero (ROIC en EEUU: 0,0002 en vez de 0,0146).
    return {"ic": ics.mean(), "ic_trim": por_trim.mean(), "se": se,
            "pct_pos": (ics > 0).mean() * 100,
            "n_fechas": len(ics), "n_trim": n,
            "t": por_trim.mean() / se if se and np.isfinite(se) else np.nan}


def main():
    d = pd.read_parquet(AQUI / f"panel_transversal{SUFIJO}.parquet")
    d["fecha"] = pd.to_datetime(d["fecha"])
    metricas = [n for n in REGISTRY if n in d.columns]

    print("=" * 100)
    print(f"PANEL: {len(d):,} filas · {d.fecha.nunique()} fechas · "
          f"{d.symbol.nunique()} valores · {len(metricas)} métricas")
    print("=" * 100)

    # ---------- 1. retorno por puesto ----------
    print("\n1) RETORNO MEDIO POR PUESTO EN EL RANKING (los 20 primeros)\n")
    top = d[d.puesto <= 20].copy()
    tabla = top.groupby("puesto")[HOR].mean() * 100
    tabla["n"] = top.groupby("puesto").size()
    print(tabla.to_string(float_format=lambda v: f"{v:+7.2f}"))

    print("\n   por tramos:")
    d["tramo"] = pd.cut(d.puesto, [0, 3, 5, 10, 20, 50, 100, 10**6],
                        labels=["1-3", "4-5", "6-10", "11-20", "21-50", "51-100", ">100"])
    tr = d.groupby("tramo", observed=True)[HOR].mean() * 100
    tr["n"] = d.groupby("tramo", observed=True).size()
    print(tr.to_string(float_format=lambda v: f"{v:+7.2f}"))

    # ¿la pendiente dentro del top 20 es real? correlación puesto-retorno por fecha
    print("\n   ¿ordena el puesto dentro del top 20? (correlación puesto vs retorno, por fecha)")
    for col in HOR:
        ics = ic_por_fecha(top, "puesto", col, minimo=15)
        r = resumen_ic(ics)
        if r:
            print(f"     {col:9s} rho medio {r['ic']:+.3f}   fechas con signo esperado "
                  f"{100 - r['pct_pos']:.0f}%   t={r['t']:+.2f}")

    # ---------- 2. IC por métrica ----------
    print("\n\n2) INFORMATION COEFFICIENT POR MÉTRICA (Spearman dentro de cada fecha)")
    print("   IC>0 = percentil alto de la métrica -> mejor retorno posterior\n")
    filas = []
    for m in metricas:
        fila = {"metrica": m, "panel": REGISTRY[m].panel[:3], "etiqueta": REGISTRY[m].label}
        for col in HOR:
            r = resumen_ic(ic_por_fecha(d, m, col))
            fila[f"ic_{col}"] = r["ic"] if r else np.nan
            fila[f"ic_trim_{col}"] = r["ic_trim"] if r else np.nan
            fila[f"se_{col}"] = r["se"] if r else np.nan
            fila[f"t_{col}"] = r["t"] if r else np.nan
            fila[f"pos_{col}"] = r["pct_pos"] if r else np.nan
        filas.append(fila)
    ic = pd.DataFrame(filas).set_index("etiqueta")

    vista = ic[["panel"] + [f"ic_{c}" for c in HOR] + ["t_fwd_63", "pos_fwd_63"]]
    print(vista.sort_values("ic_fwd_63", ascending=False).to_string(
        float_format=lambda v: f"{v:+.3f}"))

    n_contrastes = len(metricas) * len(HOR)
    umbral = 2.9  # ~Bonferroni al 5% con 80 contrastes
    print(f"\n   {n_contrastes} contrastes. Con |t| > {umbral} sobrevivirían a Bonferroni:")
    sobreviven = []
    for col in HOR:
        for et, row in ic.iterrows():
            if abs(row.get(f"t_{col}", np.nan)) > umbral:
                sobreviven.append(f"{et} ({col}, t={row[f't_{col}']:+.1f})")
    print("     " + ("; ".join(sobreviven) if sobreviven else
                     "NINGUNO. Nada supera el listón de las comparaciones múltiples."))

    # ---------- 3. ¿es el sector? ----------
    print("\n\n3) ¿ES EL SECTOR?\n")
    top20 = d[d.puesto <= 20]
    sec = top20.groupby("sector").agg(n=("symbol", "size"), fwd_63=("fwd_63", "mean"),
                                      fwd_hoy=("fwd_hoy", "mean"))
    sec[["fwd_63", "fwd_hoy"]] *= 100
    print("   retorno de los top 20 por sector:")
    print(sec.sort_values("fwd_63", ascending=False).to_string(
        float_format=lambda v: f"{v:+7.2f}"))

    print("\n   IC de las 5 métricas más fuertes, recalculado DENTRO de cada sector:")
    mejores = ic.sort_values("ic_fwd_63", ascending=False).head(5)
    for et, row in mejores.iterrows():
        m = row["metrica"]
        dentro = []
        for sector, g in d.groupby("sector"):
            if len(g) < 400:
                continue
            r = resumen_ic(ic_por_fecha(g, m, "fwd_63"))
            if r:
                dentro.append(r["ic"])
        if dentro:
            print(f"     {et:28s} global {row['ic_fwd_63']:+.3f}   "
                  f"media intra-sector {np.mean(dentro):+.3f}   "
                  f"({len(dentro)} sectores)")

    ic.to_csv(AQUI / f"information_coefficient{SUFIJO}.csv")
    tabla.to_csv(AQUI / f"retorno_por_puesto{SUFIJO}.csv")
    print("\nGuardado: information_coefficient.csv y retorno_por_puesto.csv")


main()
