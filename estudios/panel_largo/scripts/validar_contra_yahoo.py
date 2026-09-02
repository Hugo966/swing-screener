"""¿Dicen lo mismo la SEC y Yahoo donde se solapan?

Antes de fiarse de un panel que llega a 2012 hay que comprobar el mapeo de
etiquetas XBRL contra una fuente independiente. Yahoo cubre 2024-2026, así que
ahí se puede contrastar dato a dato.

Qué se compara y por qué así:

- **Por periodo, no por posición.** Los ejercicios fiscales no coinciden con el
  año natural, y emparejar "el último" de cada fuente cruzaría periodos
  distintos. Se cruza por fecha de cierre exacta.
- **Diferencia relativa, no absoluta.** Comparar dólares mezclaría una petrolera
  con una biotech; lo que interesa es si el valor es el mismo, no su tamaño.
- **Se informa la mediana y la cola.** Una mediana buena con un 10% de disparates
  significa que el mapeo funciona salvo en un sector concreto, y eso hay que
  verlo antes de construir nada encima.

Un desacuerdo no siempre es un error del mapeo: Yahoo normaliza algunas partidas
y la SEC publica lo declarado. Por eso el veredicto se da por métrica, no global.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

from screener.config import load_config
from screener.data.provider import build_provider
from screener.data.sec_provider import SecProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
log = logging.getLogger("validar")

AQUI = Path(__file__).resolve().parent.parent
CUANTOS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TOLERANCIA = 0.01          # 1%: por debajo se considera el mismo número

# Filas presentes en ambas fuentes con el mismo significado.
COMPARABLES = [
    ("Total Revenue", "income"),
    ("Net Income", "income"),
    ("Gross Profit", "income"),
    # El GAAP declarado, no el operativo normalizado de Yahoo: son filas
    # distintas y para Intel difieren en dos órdenes de magnitud.
    ("Total Operating Income As Reported", "income"),
    ("Operating Cash Flow", "cashflow"),
    ("Capital Expenditure", "cashflow"),
    ("Stock Based Compensation", "cashflow"),
    ("Total Assets", "balance"),
    ("Stockholders Equity", "balance"),
]


def fila(marco: pd.DataFrame | None, etiqueta: str) -> pd.Series | None:
    if marco is None or marco.empty or etiqueta not in marco.index:
        return None
    serie = marco.loc[etiqueta].dropna()
    serie.index = pd.to_datetime(serie.index).tz_localize(None)
    return serie if len(serie) else None


def comparar(symbol: str, sec: SecProvider, yahoo) -> list[dict]:
    try:
        est_sec = sec.statements(symbol)
        est_yah = yahoo.fundamentals.statements(symbol)
    except Exception as err:  # noqa: BLE001 — red o símbolo raro
        log.debug("%s: %s", symbol, err)
        return []

    marcos = {
        "income": (est_sec.income_a, est_yah.income_a),
        "cashflow": (est_sec.cashflow_a, est_yah.cashflow_a),
        "balance": (est_sec.balance_q, est_yah.balance_q),
    }

    salida = []
    for etiqueta, cual in COMPARABLES:
        m_sec, m_yah = marcos[cual]
        s, y = fila(m_sec, etiqueta), fila(m_yah, etiqueta)
        if s is None or y is None:
            continue
        # Cruce por fecha de cierre: emparejar por posición mezclaría periodos.
        for fecha in s.index.intersection(y.index):
            v_sec, v_yah = float(s[fecha]), float(y[fecha])
            if abs(v_yah) < 1:          # evita dividir por cero
                continue
            salida.append({
                "symbol": symbol, "etiqueta": etiqueta, "fecha": fecha,
                "sec": v_sec, "yahoo": v_yah,
                "dif_rel": abs(v_sec - v_yah) / abs(v_yah),
            })
    return salida


def main() -> None:
    cfg = load_config()
    region = cfg.region("us")
    yahoo = build_provider(region, cfg)
    sec = SecProvider("./.cache/sec")

    log.info("cargando el consolidado de la SEC…")
    universo = sec.datos["cik"].nunique()
    log.info("%d empresas en el panel de la SEC", universo)

    # Símbolos grandes y conocidos: si el mapeo falla ahí, falla en todo.
    candidatos = [s for s in (
        "AAPL MSFT NVDA GOOGL AMZN META TSLA AVGO LLY JPM XOM UNH V MA COST HD "
        "PG JNJ ABBV WMT MRK KO PEP ADBE CRM AMD NFLX TMO CSCO ACN LIN ABT MCD "
        "DHR INTC VZ TXN NEE PM CAT WFC BMY RTX AMGN UNP SPGI LOW HON COP GS "
        "BLK PFE PLD ETN AXP SYK BKNG MDT ADI TJX VRTX GILD C MMC LRCX SCHW"
    ).split() if sec.cik(s)][:CUANTOS]
    log.info("comparando %d símbolos", len(candidatos))

    filas = []
    for i, symbol in enumerate(candidatos, 1):
        filas.extend(comparar(symbol, sec, yahoo))
        if i % 10 == 0:
            log.info("%d/%d", i, len(candidatos))

    if not filas:
        print("\nSin solapamiento: ¿está el consolidado construido?")
        return

    d = pd.DataFrame(filas)
    AQUI.mkdir(parents=True, exist_ok=True)
    d.to_csv(AQUI / "validacion_sec_yahoo.csv", index=False)

    print("\n" + "=" * 78)
    print(f"SEC vs YAHOO · {d.symbol.nunique()} empresas · {len(d)} comparaciones")
    print("=" * 78)
    resumen = d.groupby("etiqueta").agg(
        n=("dif_rel", "size"),
        coinciden_pct=("dif_rel", lambda s: (s <= TOLERANCIA).mean() * 100),
        mediana_dif=("dif_rel", "median"),
        p90_dif=("dif_rel", lambda s: s.quantile(0.90)),
    ).sort_values("coinciden_pct", ascending=False)
    print(resumen.to_string(float_format=lambda v: f"{v:8.3f}"))

    print(f"\nGlobal: {(d.dif_rel <= TOLERANCIA).mean() * 100:.1f}% coinciden "
          f"dentro del {TOLERANCIA:.0%}")

    print("\nLas 10 discrepancias mayores (para depurar el mapeo):")
    peores = d.nlargest(10, "dif_rel")
    for _, f in peores.iterrows():
        print(f"  {f.symbol:6s} {f.etiqueta:26s} {str(f.fecha)[:10]}  "
              f"sec={f.sec:>16,.0f}  yahoo={f.yahoo:>16,.0f}  "
              f"dif={f.dif_rel * 100:6.1f}%")

    print("\nGuardado: validacion_sec_yahoo.csv")


main()
