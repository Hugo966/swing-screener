"""Top 5 semanal comparando cuatro reglas de salida.

El scoring es idéntico para las cuatro, así que se puntúa UNA vez y luego se
simula cada regla sobre las mismas señales. Las decisiones de compra sí difieren:
al cerrarse una posición, ese valor vuelve a ser comprable.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dataclasses import dataclass

import pandas as pd

from screener import universe as universe_mod
from screener.backtest import FxHistory
from screener.config import load_config
from screener.data.provider import build_provider
from screener.engine import build_sector_indices, fetch_fundamentals, score_universe
from screener.metrics._indicators import atr_pct
from screener.models import Candidate, TickerData
from screener.pointintime import as_of
from screener.regime import compute_regime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
log = logging.getLogger("sim")

SALIDA = Path(__file__).resolve().parent.parent / "posiciones"
SALIDA.mkdir(exist_ok=True)
REGION, TICKET_EUR, TOP_N = "us", 100.0, 5
MESES = 31          # hasta donde el panel B tiene cobertura decente
PERIOD = "4y"       # ya cacheado; cubre los 13 meses previos que exige el gate
ATR_K = 3.0
HARD_STOP = 0.08    # variante híbrida
ARM_AT = 0.10


@dataclass
class Pos:
    symbol: str
    entrada: pd.Timestamp
    precio_entrada: float
    fx_entrada: float
    acciones: float
    salida: pd.Timestamp | None = None
    precio_salida: float | None = None
    fx_salida: float | None = None
    motivo: str = ""


def cierres(prices):
    idx = pd.DatetimeIndex(prices.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(prices["Close"].to_numpy(), index=idx)
    return s[~s.index.duplicated()]


# --- reglas de salida: (serie, atr, entrada, precio) -> (fecha, precio, motivo)|None
def sin_stop(serie, atr, entrada, precio):
    return None


def take_profit_20(serie, atr, entrada, precio):
    tramo = serie[serie.index > entrada]
    objetivo = precio * 1.20
    hit = tramo[tramo >= objetivo]
    return (hit.index[0], float(hit.iloc[0]), "+20%") if len(hit) else None


def trailing_pct(serie, atr, entrada, precio, banda=0.15):
    tramo = serie[serie.index > entrada]
    pico = precio
    for fecha, valor in tramo.items():
        if valor <= pico * (1 - banda):
            return fecha, float(valor), f"trailing {banda:.0%}"
        pico = max(pico, float(valor))
    return None


def trailing_atr(serie, atr, entrada, precio, k=ATR_K):
    tramo = serie[serie.index > entrada]
    pico = precio
    for fecha, valor in tramo.items():
        banda = float(atr.get(fecha, atr.iloc[-1] if len(atr) else 0.05)) * k
        if valor <= pico * (1 - banda):
            return fecha, float(valor), f"ATR×{k:g}"
        pico = max(pico, float(valor))
    return None


def hibrido(serie, atr, entrada, precio):
    tramo = serie[serie.index > entrada]
    pico, armado = precio, False
    for fecha, valor in tramo.items():
        pico = max(pico, float(valor))
        if not armado and pico >= precio * (1 + ARM_AT):
            armado = True
        if not armado:
            if valor <= precio * (1 - HARD_STOP):
                return fecha, float(valor), f"stop duro -{HARD_STOP:.0%}"
        else:
            banda = float(atr.get(fecha, 0.05)) * ATR_K
            if valor <= pico * (1 - banda):
                return fecha, float(valor), f"ATR×{ATR_K:g}"
    return None


REGLAS = {
    "sin stop": sin_stop,
    "+20% fijo": take_profit_20,
    "trailing 15%": trailing_pct,
    f"trailing ATR k={ATR_K:g}": trailing_atr,
    f"híbrido -{HARD_STOP:.0%} / ATR": hibrido,
}


def main():
    cfg = load_config()
    region = cfg.region(REGION)
    provider = build_provider(region, cfg)

    cands = universe_mod.build_candidates(region, provider, cfg)
    prices = provider.prices.history([c.symbol for c in cands], period=PERIOD)
    cands = [c for c in cands if c.symbol in prices]
    fundamentales = fetch_fundamentals(provider, [c.symbol for c in cands],
                                       int(cfg.data["max_workers"]))
    benchmark = provider.prices.close_series(region.benchmark, period=PERIOD)
    indices = build_sector_indices(cands, prices)
    fx = FxHistory(provider, PERIOD)

    base, por_symbol = {}, {}
    for c in cands:
        paq = fundamentales.get(c.symbol)
        if paq is None:
            continue
        perfil, est, esti = paq
        sector = perfil.get("sector") or c.sector
        if not universe_mod.passes_industry(perfil.get("industry") or c.industry, cfg):
            continue
        if not universe_mod.passes_country(perfil.get("country"), region):
            continue
        c.sector = sector
        por_symbol[c.symbol] = c
        base[c.symbol] = TickerData(
            symbol=c.symbol, asof=pd.Timestamp.today().date(), sector=sector,
            industry=perfil.get("industry"), prices=prices.get(c.symbol),
            benchmark=benchmark, sector_index=indices.get(sector) if sector else None,
            profile=perfil, income_q=est.income_q, income_a=est.income_a,
            cashflow_q=est.cashflow_q, cashflow_a=est.cashflow_a,
            balance_q=est.balance_q, balance_a=est.balance_a,
            earnings_dates=esti.earnings_dates, eps_revisions=esti.eps_revisions,
            eps_trend=esti.eps_trend, shares_full=esti.shares_full)
    log.info("universo con datos: %d", len(base))

    series = {s: cierres(d.prices) for s, d in base.items()}
    atrs = {}
    for s, d in base.items():
        a = atr_pct(d.prices, 14)
        idx = pd.DatetimeIndex(a.index)
        a.index = idx.tz_localize(None) if idx.tz is not None else idx
        atrs[s] = a.dropna()

    sesiones = sorted({t for x in series.values() for t in x.index})
    calendario = pd.DatetimeIndex(sesiones)
    hoy = calendario.max()
    ventana = calendario[calendario >= hoy - pd.DateOffset(months=MESES)]
    lunes = sorted(g.min() for _, g in pd.Series(ventana, index=ventana)
                   .groupby(pd.Grouper(freq="W-SUN")) if len(g))
    log.info("%d lunes: %s .. %s", len(lunes), lunes[0].date(), lunes[-1].date())

    # ---------- 1. puntuar una sola vez ----------
    señales, descartes = [], []
    for i, fecha in enumerate(lunes, 1):
        pasado = {s: as_of(d, fecha, fallback_lag_days=60) for s, d in base.items()}
        pasado = {s: d for s, d in pasado.items() if d.prices is not None}
        precios = {s: d.prices for s, d in pasado.items()}
        lote = [Candidate(symbol=s, name=por_symbol[s].name, sector=d.sector,
                          industry=d.industry, market_cap=d.profile.get("marketCap"),
                          avg_volume=float(d.prices["Volume"].iloc[-63:].mean()),
                          price=float(d.prices["Close"].iloc[-1]),
                          currency=por_symbol[s].currency, exchange=por_symbol[s].exchange)
                for s, d in pasado.items()]
        vivos, _ = universe_mod.apply_cheap_gates(
            lote, cfg, provider=None, rates=lambda cur: fx.rates(cur, fecha))
        vivos, _ = universe_mod.apply_trend_gate(vivos, precios, cfg)
        if not vivos:
            continue
        reg = compute_regime(pasado[vivos[0].symbol].benchmark, precios, cfg)
        sc = score_universe({c.symbol: pasado[c.symbol] for c in vivos}, vivos, region,
                            cfg, regime_multiplier=reg.multiplier)
        top = sorted(sc.results, key=lambda r: r.score_final, reverse=True)[:TOP_N]
        señales.append({"fecha": fecha.isoformat(),
                        "top": [r.symbol for r in top],
                        "regimen": reg.multiplier,
                        "puntuados": len(sc.results),
                        "descartados": len(sc.dropped_low_coverage)})
        descartes.append(len(sc.dropped_low_coverage))
        if i % 10 == 0 or i == len(lunes):
            log.info("%d/%d  %s  top=%s", i, len(lunes), fecha.date(),
                     ",".join(r.symbol for r in top))
    json.dump(señales, open(SALIDA / "senales.json", "w"), indent=1)
    log.info("señales guardadas: %d fechas", len(señales))

    # ---------- 2. simular cada regla ----------
    fx_hoy = fx.rate("EUR", hoy)
    resumen, detalles = [], {}
    for nombre, regla in REGLAS.items():
        abiertas, cerradas = {}, []
        for s in señales:
            fecha = pd.Timestamp(s["fecha"])
            for sym, pos in list(abiertas.items()):
                r = regla(series[sym], atrs[sym], pos.entrada, pos.precio_entrada)
                if r and r[0] <= fecha:
                    pos.salida, pos.precio_salida, pos.motivo = r
                    pos.fx_salida = fx.rate("EUR", r[0])
                    cerradas.append(pos); del abiertas[sym]
            fxd = fx.rate("EUR", fecha)
            for sym in s["top"]:
                if sym in abiertas or sym not in series:
                    continue
                serie = series[sym]
                previos = serie[serie.index <= fecha]
                if previos.empty:
                    continue
                precio = float(previos.iloc[-1])
                abiertas[sym] = Pos(sym, fecha, precio, fxd, TICKET_EUR * fxd / precio)
        for sym, pos in list(abiertas.items()):
            r = regla(series[sym], atrs[sym], pos.entrada, pos.precio_entrada)
            if r:
                pos.salida, pos.precio_salida, pos.motivo = r
                pos.fx_salida = fx.rate("EUR", r[0])
                cerradas.append(pos); del abiertas[sym]

        filas = []
        for pos in cerradas + list(abiertas.values()):
            if pos.salida is not None:
                valor = pos.acciones * pos.precio_salida / pos.fx_salida
                estado = f"{pos.motivo} {pos.salida.date()}"
            else:
                valor = pos.acciones * float(series[pos.symbol].iloc[-1]) / fx_hoy
                estado = "abierta"
            filas.append({"symbol": pos.symbol, "entrada": pos.entrada.date(),
                          "invertido": TICKET_EUR, "valor": valor,
                          "ret_%": (valor / TICKET_EUR - 1) * 100, "estado": estado})
        d = pd.DataFrame(filas)
        detalles[nombre] = d
        inv, val = d["invertido"].sum(), d["valor"].sum()
        resumen.append({"regla": nombre, "posiciones": len(d), "invertido": inv,
                        "valor": val, "pnl": val - inv, "pnl_%": (val / inv - 1) * 100,
                        "ganadoras_%": (d["ret_%"] > 0).mean() * 100,
                        "cerradas": int((d.estado != "abierta").sum()),
                        "mejor_%": d["ret_%"].max(), "peor_%": d["ret_%"].min()})
        d.to_csv(SALIDA / f"sim2_{nombre.replace(' ', '_').replace('/', '-')}.csv", index=False)

    r = pd.DataFrame(resumen)
    r.to_csv(SALIDA / "sim2_resumen.csv", index=False)
    print("\n" + "=" * 96)
    print(f"TOP {TOP_N} SEMANAL · {REGION} · {lunes[0].date()} → {hoy.date()} · "
          f"{len(señales)} semanas · ticket {TICKET_EUR:.0f} €")
    print(f"descartados por cobertura, media por fecha: {sum(descartes)/max(len(descartes),1):.0f}")
    print("=" * 96)
    print(r.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


main()
