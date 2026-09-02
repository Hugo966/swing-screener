"""¿Qué cambia al pasar de 4,5 a 13,6 años de panel?

Compara los dos backtests de la misma región y periodo nominal: uno con
fundamentales de Yahoo, que solo alcanza 2022, y otro con los de la SEC, que
llega a 2013.

La pregunta que importa no es si el exceso medio es mayor —eso se mide en el
propio backtest— sino **si ahora se puede afirmar algo con significación**. Con
2,6 años el estudio anterior concluyó que no; el listón es t >= 2.

Dos decisiones metodológicas:

- **Se agrega por fecha antes de calcular el estadístico.** Las alertas de una
  misma fecha comparten el mismo shock de mercado: tratarlas como observaciones
  independientes multiplica artificialmente la n y regala significación.
- **Se usan horizontes no solapados.** Un horizonte de 126 sesiones muestreado
  cada mes solapa cinco veces; la n efectiva es el número de periodos
  independientes, no el de fechas.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent.parent
SALIDA = Path("out")


def cargar(nombre: str) -> pd.DataFrame | None:
    ruta = SALIDA / nombre
    if not ruta.exists():
        return None
    d = pd.read_csv(ruta, parse_dates=["date"])
    return d


def exceso_por_fecha(alertas: pd.DataFrame, universo: pd.DataFrame,
                     horizonte: str) -> pd.Series:
    """Exceso de la cesta de alertas sobre el universo, fecha a fecha.

    Contrastar el retorno bruto no dice nada: en un mercado alcista sale
    significativo aunque la estrategia no aporte. Lo que hay que medir es la
    diferencia contra el listón del propio universo puntuado ese día.
    """
    col = f"fwd_{horizonte}"
    cesta = alertas[alertas[col].notna()].groupby("date")[col].mean()
    listón = universo.set_index("date")[col]
    par = pd.concat([cesta.rename("cesta"), listón.rename("listón")],
                    axis=1, join="inner").dropna()
    return par["cesta"] - par["listón"]


def estadistico(serie: pd.Series, sesiones: int) -> dict:
    """t de la media, con la n corregida por solapamiento.

    Muestrear cada mes un horizonte de `sesiones` días hace que cada
    observación comparta la mayor parte de su ventana con las vecinas. La n
    efectiva se aproxima dividiendo por el factor de solape.
    """
    if len(serie) < 3:
        return {}
    solape = max(1.0, sesiones / 21.0)      # rebalanceo mensual ~21 sesiones
    n_efectiva = len(serie) / solape
    media = serie.mean()
    error = serie.std(ddof=1) / np.sqrt(n_efectiva)
    return {
        "fechas": len(serie),
        "n_efectiva": n_efectiva,
        "media_pct": media * 100 if abs(media) < 1 else media,
        "t": media / error if error else np.nan,
    }


def main() -> None:
    fuentes = {}
    for etiqueta, base in (("SEC (2013-)", "backtest_us_sec"),
                           ("Yahoo (2022-)", "backtest_us_yfinance")):
        alertas, universo = cargar(f"{base}.csv"), cargar(f"{base}_universo.csv")
        if alertas is not None and universo is not None:
            fuentes[etiqueta] = (alertas, universo)
    if not fuentes:
        print("No hay CSV de backtest todavía.")
        return

    print("=" * 74)
    print("SIGNIFICACIÓN DEL EXCESO · agregado por fecha, n corregida por solape")
    print("=" * 74)

    for etiqueta, (d, universo) in fuentes.items():
        print(f"\n{etiqueta}   {d.date.min().date()} → {d.date.max().date()}"
              f"   {d.date.nunique()} fechas · {len(d)} alertas")
        for h, sesiones in (("21", 21), ("63", 63), ("126", 126)):
            serie = exceso_por_fecha(d, universo, h)
            est = estadistico(serie, sesiones)
            if not est:
                continue
            veredicto = "SIGNIFICATIVO" if abs(est["t"]) >= 2 else "no significativo"
            print(f"   {h:>4}d  exceso {est['media_pct']:+7.2f}%   "
                  f"n_ef {est['n_efectiva']:5.1f}   t {est['t']:+5.2f}   {veredicto}")

    # Estabilidad del barrido: si el mejor umbral de un panel no es el mejor del
    # otro, la elección es ruido y no debe tocarse. Es la misma conclusión que
    # el metaanálisis de pesos, comprobada ahora sobre una muestra 3 veces mayor.
    if len(fuentes) == 2:
        print("\n" + "=" * 74)
        print("¿COINCIDEN LOS PANELES EN QUÉ ALERTAS SON BUENAS?")
        print("=" * 74)
        sec, yah = fuentes["SEC (2013-)"][0], fuentes["Yahoo (2022-)"][0]
        comun = pd.merge(
            sec[["date", "symbol", "score", "fwd_63"]],
            yah[["date", "symbol", "score", "fwd_63"]],
            on=["date", "symbol"], suffixes=("_sec", "_yah"))
        print(f"\n  alertas emitidas por ambos en la misma fecha: {len(comun)}")
        if len(comun) > 30:
            r = comun["score_sec"].corr(comun["score_yah"])
            print(f"  correlación de los scores: {r:.3f}")
            if r >= 0.7:
                print("  Alta: los dos paneles puntúan casi igual. Que sus barridos")
                print("  de umbrales señalen ganadores distintos es entonces prueba")
                print("  de que el barrido ajusta ruido, no de que discrepen.")
            else:
                print("  Baja: los paneles puntúan distinto, así que comparar sus")
                print("  barridos de umbrales no tiene sentido.")

        solo_sec = sec[sec.date < yah.date.min()]
        print(f"\n  alertas que SOLO existen gracias a la SEC "
              f"(antes de {yah.date.min().date()}): {len(solo_sec)}")
        if len(solo_sec):
            print(f"  su retorno medio a 63d: {solo_sec.fwd_63.mean() * 100:+.2f}%")


main()
