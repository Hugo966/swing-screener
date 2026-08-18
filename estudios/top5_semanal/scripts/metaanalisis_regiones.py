"""¿Qué métricas replican en TODAS las regiones? Metaanálisis, no recuento.

Contar cuántas métricas coinciden en signo entre regiones es un test pobre:
tira las magnitudes y las precisiones, y con 3-4 regiones el número esperado por
azar (2*(1/2)^k por métrica) se parece tanto al observado que nunca concluye
nada. Aquí se hace lo que corresponde:

1. **Combinación por varianza inversa.** El IC de cada región se pondera por
   1/se², con el `se` que `analizar_patrones` guarda explícitamente. NO se deduce
   como ic/t: `ic` es la media sobre fechas y `t` sale de la media por
   trimestres, que llevan distinto número de fechas, así que el cociente da
   disparates cuando el IC ronda cero (ROIC en EEUU salía 0,0002 en vez de
   0,0146). Se usa el par coherente `ic_trim` + `se`. Una región con sección
   transversal grande pesa más que una pequeña, que es lo que debe pasar.

2. **Q de Cochran.** Mide si las regiones se contradicen. Es el contraste que de
   verdad importa: una métrica puede dar un t agregado alto promediando -0,06 en
   EEUU con +0,09 en Europa, y ese promedio no significa nada. Q por encima del
   crítico (7,81 con 4 regiones, 5,99 con 3) = las regiones miden cosas
   distintas y el agregado no se puede interpretar.

3. **Robustez quitando Corea.** Su universo es de ~86 valores y el 60% de sus
   fechas no llega al mínimo de 30 para calcular el IC; las que sobreviven son
   todas posteriores a 2025Q3. Se informa con y sin ella.

CAUTELA QUE NO SE PUEDE CORREGIR AQUÍ: las regiones comparten el mismo periodo
(ene-2024 en adelante), así que sus errores están correlacionados por los
movimientos globales. El metaanálisis supone independencia entre estudios, luego
el t agregado sale INFLADO. Léase cualquier |t| cercano a 2 como menos de 2.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent.parent
REGIONES = sys.argv[1:] or ["us", "europe_dev", "emerging", "korea"]
SIN = "korea"          # región que se excluye en el contraste de robustez
COL = "ic_fwd_63"


def carga(regiones):
    ic, se = {}, {}
    for r in regiones:
        f = AQUI / f"information_coefficient{'' if r == 'us' else f'_{r}'}.csv"
        if not f.exists():
            print(f"(falta {f.name}, se omite {r})")
            continue
        d = pd.read_csv(f).set_index("etiqueta")
        h = COL.split("_", 1)[1]
        # Se usa la media POR TRIMESTRES junto con su propio error estándar: son
        # el par coherente. La media sobre fechas (`ic_fwd_63`) tiene otro
        # numerador —los trimestres no llevan el mismo número de fechas— y
        # combinarla con este `se` desajusta las ponderaciones.
        ic[r] = d[f"ic_trim_{h}"]
        se[r] = d[f"se_{h}"]
    return pd.DataFrame(ic), pd.DataFrame(se).replace(0, np.nan)


def meta(IC, SE, cols):
    """Devuelve ic combinado, t y Q de Cochran para el subconjunto de regiones."""
    I, S = IC[cols], SE[cols]
    W = 1 / S**2
    pool = (I * W).sum(axis=1) / W.sum(axis=1)
    t = pool * np.sqrt(W.sum(axis=1))
    Q = (W * (I.sub(pool, axis=0)) ** 2).sum(axis=1)
    return pd.DataFrame({"ic": pool, "t": t, "Q": Q, "n": I.notna().sum(axis=1)})


def qcrit(k):
    """Valor crítico de chi2 al 5% con k-1 grados de libertad."""
    return {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07}.get(k - 1, np.nan)


def main():
    IC, SE = carga(REGIONES)
    todas = [r for r in REGIONES if r in IC.columns]
    if len(todas) < 3:
        print("Hacen falta al menos 3 regiones.")
        return

    a = meta(IC, SE, todas)
    qa = qcrit(len(todas))

    resto = [r for r in todas if r != SIN]
    hay_resto = len(resto) >= 3 and SIN in todas
    b, qb = (meta(IC, SE, resto), qcrit(len(resto))) if hay_resto else (None, None)

    print("=" * 100)
    print(f"METAANÁLISIS · IC a 63 días · regiones: {', '.join(todas)}")
    print(f"Q crítico al 5%: {qa} ({len(todas)} regiones)"
          + (f"  ·  {qb} sin {SIN}" if hay_resto else ""))
    print("=" * 100)

    out = pd.DataFrame({"ic": a.ic, "t": a.t, "Q": a.Q})
    if hay_resto:
        out[f"t_sin_{SIN}"] = b.t
        out[f"Q_sin_{SIN}"] = b.Q
    out["homogenea"] = (a.Q < qa) & ((b.Q < qb) if hay_resto else True)
    out = out.dropna(subset=["t"]).sort_values("t", ascending=False)
    print(out.to_string(float_format=lambda v: f"{v:+.3f}"))

    print("\n" + "-" * 100)
    print("SÓLIDAS — |t| > 2 en ambos cortes y sin heterogeneidad:")
    cond = out.homogenea & (out.t.abs() > 2)
    if hay_resto:
        cond &= out[f"t_sin_{SIN}"].abs() > 2
    solidas = out[cond]
    for e, r in solidas.iterrows():
        print(f"   {e:28s} IC={r.ic:+.3f}  t={r.t:+.2f}  Q={r.Q:.2f}")
    if solidas.empty:
        print("   NINGUNA")

    print("\nDESCARTADAS POR HETEROGENEIDAD (las regiones se contradicen):")
    for e, r in out[~out.homogenea].iterrows():
        vals = "  ".join(f"{c}={IC.loc[e, c]:+.3f}" for c in todas if pd.notna(IC.loc[e, c]))
        print(f"   {e:28s} Q={r.Q:5.2f}   {vals}")

    out.to_csv(AQUI / "metaanalisis_regiones.csv")
    print("\nGuardado: metaanalisis_regiones.csv")
    print("\nRecordatorio: las regiones comparten periodo, así que el t agregado")
    print("está inflado. Un |t| de 2 es menos de 2 en realidad.")


main()
