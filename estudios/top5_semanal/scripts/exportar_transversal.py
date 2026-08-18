"""Exporta la sección transversal completa de cada lunes, con desglose de métricas.

Guarda, por cada fecha y cada valor puntuado: puesto, score, A_pct, B_pct, sector
y el percentil de las 20 métricas, más los retornos forward a 21/63/126 sesiones
y hasta hoy.

Con esto se puede estudiar qué distingue a las que lo hacen bien sin volver a
puntuar. El fichero sale a `panel_transversal.parquet`.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

from screener import universe as universe_mod
from screener.backtest import FxHistory
from screener.config import load_config
from screener.data.provider import build_provider
from screener.engine import build_sector_indices, fetch_fundamentals, score_universe
from screener.models import Candidate, TickerData
from screener.pointintime import as_of
from screener.regime import compute_regime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
log = logging.getLogger("export")

AQUI = Path(__file__).resolve().parent.parent
REGION = sys.argv[1] if len(sys.argv) > 1 else "us"
MESES, PERIOD = 31, "4y"
HORIZONTES = (21, 63, 126)


def cierres(prices):
    idx = pd.DatetimeIndex(prices.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(prices["Close"].to_numpy(), index=idx, dtype="float64")
    return s[~s.index.duplicated()]


def forward(serie, fecha, n):
    pos = serie.index.searchsorted(fecha, side="right") - 1
    if pos < 0 or pos + n >= len(serie):
        return None
    a, b = float(serie.iloc[pos]), float(serie.iloc[pos + n])
    return b / a - 1.0 if a > 0 else None


def hasta_hoy(serie, fecha):
    pos = serie.index.searchsorted(fecha, side="right") - 1
    if pos < 0:
        return None
    a, b = float(serie.iloc[pos]), float(serie.iloc[-1])
    return b / a - 1.0 if a > 0 else None


def main():
    cfg = load_config()
    region = cfg.region(REGION)
    provider = build_provider(region, cfg)

    cands = universe_mod.build_candidates(region, provider, cfg)
    prices = provider.prices.history([c.symbol for c in cands], period=PERIOD)
    cands = [c for c in cands if c.symbol in prices]
    fund = fetch_fundamentals(provider, [c.symbol for c in cands], int(cfg.data["max_workers"]))
    benchmark = provider.prices.close_series(region.benchmark, period=PERIOD)
    indices = build_sector_indices(cands, prices)
    fx = FxHistory(provider, PERIOD)

    base, por_symbol = {}, {}
    for c in cands:
        paq = fund.get(c.symbol)
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
    calendario = pd.DatetimeIndex(sorted({t for x in series.values() for t in x.index}))
    hoy = calendario.max()
    ventana = calendario[calendario >= hoy - pd.DateOffset(months=MESES)]
    lunes = sorted(g.min() for _, g in pd.Series(ventana, index=ventana)
                   .groupby(pd.Grouper(freq="W-SUN")) if len(g))
    log.info("%d lunes", len(lunes))

    filas = []
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
        sc = score_universe({c.symbol: pasado[c.symbol] for c in vivos}, vivos,
                            region, cfg, regime_multiplier=reg.multiplier)
        orden = sorted(sc.results, key=lambda r: r.score_final, reverse=True)

        for puesto, r in enumerate(orden, 1):
            serie = series[r.symbol]
            fila = {"fecha": fecha, "symbol": r.symbol, "sector": r.sector,
                    "puesto": puesto, "score": r.score_final,
                    "a_pct": r.a_pct, "b_pct": r.b_pct, "regimen": reg.multiplier,
                    "n_puntuados": len(orden)}
            for panel in (r.momentum, r.quality):
                if panel is None:
                    continue
                for m in panel.metrics:
                    fila[m.name] = m.percentile
                    fila[f"{m.name}__imp"] = m.imputed
            for h in HORIZONTES:
                fila[f"fwd_{h}"] = forward(serie, fecha, h)
            fila["fwd_hoy"] = hasta_hoy(serie, fecha)
            filas.append(fila)

        if i % 10 == 0 or i == len(lunes):
            log.info("%d/%d %s (%d valores)", i, len(lunes), fecha.date(), len(orden))

    d = pd.DataFrame(filas)
    sufijo = "" if REGION == "us" else f"_{REGION}"
    destino = AQUI / f"panel_transversal{sufijo}.parquet"
    d.to_parquet(destino, index=False)
    log.info("guardado %s: %d filas x %d columnas", destino.name, len(d), d.shape[1])

    # Las señales salen del propio panel: así `comparar_top_n.py` simula las
    # carteras sin volver a puntuar, que es lo caro.
    import json
    señales = [
        {"fecha": fecha.isoformat(),
         "regimen": float(g.regimen.iloc[0]),
         "puntuados": int(g.n_puntuados.iloc[0]),
         "top": g.nsmallest(20, "puesto").symbol.tolist(),
         "scores": [round(x, 2) for x in g.nsmallest(20, "puesto").score.tolist()]}
        for fecha, g in d.groupby("fecha")
    ]
    (AQUI / f"senales_top20_{REGION}.json").write_text(json.dumps(señales, indent=1))
    log.info("señales guardadas: %d fechas", len(señales))

    print(f"\nfilas: {len(d)}   fechas: {d.fecha.nunique()}   "
          f"valores únicos: {d.symbol.nunique()}")


main()
