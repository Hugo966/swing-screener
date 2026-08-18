"""¿El patrón de métricas de EEUU se repite fuera? Comparación región a región.

El cambio de pesos (A3 arriba, A4 abajo) se dedujo del panel americano. Si es
una regularidad y no un ajuste a esos datos, el IC de cada métrica debería
ordenarse parecido en las demás regiones. Este script mide justo eso:

1. Tabla del IC a 63 días de cada métrica en cada región, lado a lado.
2. Correlación de rangos entre los vectores de IC de cada par de regiones. Es
   el contraste que importa: si sale ~0, el orden de las métricas no viaja y
   cualquier peso afinado en una región es ruido en las demás.
3. Detalle de las dos métricas sobre las que se movieron los pesos.

Cautela: cada región tiene su propio tamaño de sección transversal, así que sus
IC no son igual de fiables. Se imprime el número de filas para no comparar un
panel de 90.000 con uno de 13.000 como si pesaran lo mismo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent.parent
REGIONES = sys.argv[1:] or ["us", "europe_dev", "emerging", "korea"]
FOCO = ["A3 RS vs sector", "A4 momentum del sector"]


def carga(region):
    sufijo = "" if region == "us" else f"_{region}"
    ic = AQUI / f"information_coefficient{sufijo}.csv"
    panel = AQUI / f"panel_transversal{sufijo}.parquet"
    if not ic.exists():
        return None, None
    d = pd.read_csv(ic).set_index("etiqueta")
    filas = len(pd.read_parquet(panel, columns=["symbol"])) if panel.exists() else None
    return d, filas


def main():
    ics, tam = {}, {}
    for r in REGIONES:
        d, filas = carga(r)
        if d is None:
            print(f"(sin datos para {r} todavía)")
            continue
        ics[r], tam[r] = d, filas

    if len(ics) < 2:
        print("Hacen falta al menos dos regiones para comparar.")
        return

    print("=" * 90)
    print("TAMAÑO DE CADA PANEL")
    for r, n in tam.items():
        print(f"   {r:12s} {n:>8,} filas" if n else f"   {r:12s}   (parquet no encontrado)")
    print("=" * 90)

    # ---------- 1. IC a 63 días lado a lado ----------
    tabla = pd.DataFrame({r: d["ic_fwd_63"] for r, d in ics.items()})
    tabla["panel"] = next(iter(ics.values()))["panel"]
    orden = tabla.sort_values(REGIONES[0] if REGIONES[0] in tabla else tabla.columns[0],
                              ascending=False)
    print("\n1) IC A 63 DÍAS POR MÉTRICA Y REGIÓN (ordenado por la primera región)\n")
    print(orden.to_string(float_format=lambda v: f"{v:+.3f}"))

    # ---------- 2. ¿viaja el orden de las métricas? ----------
    print("\n\n2) ¿VIAJA EL ORDEN DE LAS MÉTRICAS ENTRE REGIONES?")
    print("   Correlación de rangos entre los vectores de IC. ~+1 = mismo orden;")
    print("   ~0 = el orden no se repite; negativo = se invierte.\n")
    rs = list(ics)
    for i, a in enumerate(rs):
        for b in rs[i + 1:]:
            comun = tabla[[a, b]].dropna()
            if len(comun) < 5:
                continue
            rho = comun[a].rank().corr(comun[b].rank())
            pearson = comun[a].corr(comun[b])
            print(f"   {a:12s} vs {b:12s}   rho={rho:+.3f}   pearson={pearson:+.3f}"
                  f"   ({len(comun)} métricas)")

    # ---------- 3. las dos métricas del cambio de pesos ----------
    print("\n\n3) LAS DOS MÉTRICAS SOBRE LAS QUE SE MOVIERON LOS PESOS\n")
    for met in FOCO:
        print(f"   {met}")
        for r, d in ics.items():
            if met not in d.index:
                print(f"      {r:12s} (no está en la tabla)")
                continue
            row = d.loc[met]
            n = len(d)
            puesto = int((d["ic_fwd_63"] > row["ic_fwd_63"]).sum()) + 1
            print(f"      {r:12s} IC63={row['ic_fwd_63']:+.3f}  t={row['t_fwd_63']:+.2f}"
                  f"  puesto {puesto}/{n}")
        print()

    # ---------- 4. ¿qué sobrevive en todas las regiones? ----------
    print("\n\n4) ¿QUÉ AGUANTA EN TODAS LAS REGIONES?")
    print("   Una métrica solo cuenta si mantiene el signo en todas. Pero ojo:")
    print("   coincidir por azar es fácil. Si cada región fuese una moneda, la")
    print("   probabilidad de que k regiones coincidan es 2*(1/2)^k, y eso da un")
    print("   número esperado de coincidencias que hay que superar para creerse nada.\n")

    solo_ic = tabla.drop(columns=["panel"], errors="ignore")
    completas = solo_ic.dropna()
    k = completas.shape[1]
    if k >= 2 and len(completas):
        mismo = completas.apply(lambda r: bool(r.gt(0).all() or r.lt(0).all()), axis=1)
        esperado = len(completas) * 2 * (0.5 ** k)
        print(f"   {len(completas)} métricas con dato en las {k} regiones")
        print(f"   coinciden en signo: {int(mismo.sum())}")
        print(f"   esperadas por azar: {esperado:.1f}")
        veredicto = ("por encima del azar — merece mirarse"
                     if mismo.sum() > esperado * 1.5 else
                     "INDISTINGUIBLE DEL AZAR — no hay señal por métrica")
        print(f"   veredicto: {veredicto}\n")

        if mismo.sum():
            print("   Las que coinciden, con su rango entre regiones:")
            cons = completas[mismo].copy()
            cons["media"] = cons.mean(axis=1)
            cons["rango"] = cons.max(axis=1) - cons.min(axis=1)
            for et, row in cons.sort_values("media", ascending=False).iterrows():
                vals = "  ".join(f"{r}={row[r]:+.3f}" for r in solo_ic.columns)
                print(f"     {et:28s} media {row['media']:+.3f}  rango {row['rango']:.3f}   {vals}")

        print("\n   Las más inestables (mayor rango entre regiones):")
        rango = (completas.max(axis=1) - completas.min(axis=1)).sort_values(ascending=False)
        for et, r in rango.head(5).items():
            print(f"     {et:28s} rango {r:.3f}   "
                  + "  ".join(f"{c}={completas.loc[et, c]:+.3f}" for c in completas.columns))

    tabla.to_csv(AQUI / "comparativa_regiones_ic63.csv")
    print("\nGuardado: comparativa_regiones_ic63.csv")


main()
