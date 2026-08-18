"""Compara top 3 / top 5 / top 10 semanales, sin stop, contra el índice.

Se invoca con la región como argumento (`comparar_top_n.py emerging`); por
defecto, `us`.

Puntúa una sola vez guardando el top 20, así cualquier N <= 20 futuro sale de
ese fichero sin recalcular. Mide:

- TWR anualizada: encadena los retornos diarios netos de aportación. Es la que se
  puede comparar con un índice, porque no depende de cuándo metiste el dinero.
- TIR (money-weighted): lo que ha rendido tu dinero de verdad, dado que las
  aportaciones están escalonadas.
- Comparación contra meter EXACTAMENTE los mismos importes los mismos días en un
  índice. Sin esto, un +40% no se puede juzgar.

**Dos referencias, y las dos hacen falta.** El índice de la propia región mide
habilidad (¿bate el screener a lo que tenía a mano?) y el SPY mide si valía la
pena salir de EEUU. Pueden dar respuestas opuestas: en Europa la estrategia bate
al STOXX y aun así rinde menos que comprar SPY y no hacer nada. Ojo con la
divisa: STOXX cotiza en EUR y KOSPI en KRW, no en USD.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
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
log = logging.getLogger("topn")

AQUI = Path(__file__).resolve().parent.parent
REGION = sys.argv[1] if len(sys.argv) > 1 else "us"
SUFIJO = "" if REGION == "us" else f"_{REGION}"
TICKET_EUR = 100.0
MESES, PERIOD = 31, "4y"
GUARDAR_TOP = 20
NS = (3, 5, 10)


def cierres(prices):
    idx = pd.DatetimeIndex(prices.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(prices["Close"].to_numpy(), index=idx, dtype="float64")
    return s[~s.index.duplicated()]


def tir(flujos: list[tuple[pd.Timestamp, float]]) -> float | None:
    """TIR anualizada por bisección. Flujos: negativos al aportar."""
    if len(flujos) < 2:
        return None
    t0 = flujos[0][0]
    años = [(f - t0).days / 365.25 for f, _ in flujos]

    def npv(r):
        return sum(c / (1 + r) ** a for (_, c), a in zip(flujos, años))

    lo, hi = -0.95, 10.0
    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def metricas(curva: pd.DataFrame) -> dict:
    """curva: índice fecha, columnas `valor` (cartera) y `aporte` (del día)."""
    v, a = curva["valor"], curva["aporte"]
    previo = v.shift(1).fillna(0.0)
    # retorno diario neto de la aportación de ese día
    base = previo + a
    diario = (v / base.where(base > 0)).fillna(1.0) - 1.0
    twr = float((1 + diario).prod() - 1)
    años = (curva.index[-1] - curva.index[0]).days / 365.25
    twr_anual = (1 + twr) ** (1 / años) - 1 if años > 0 else float("nan")

    acumulada = (1 + diario).cumprod()
    dd = float((acumulada / acumulada.cummax() - 1).min())

    vol = float(diario.std() * np.sqrt(252))
    sharpe = (twr_anual / vol) if vol > 0 else float("nan")
    return {"twr": twr, "twr_anual": twr_anual, "max_dd": dd, "vol": vol, "sharpe": sharpe}


def simular(señales, n, series, fx, calendario):
    """Compra TICKET_EUR de cada nuevo valor del top N. Sin stop: nunca vende."""
    compras = []          # (fecha, symbol, acciones)
    en_cartera = set()
    for s in señales:
        fecha = pd.Timestamp(s["fecha"])
        fxd = fx.rate("EUR", fecha)
        for sym in s["top"][:n]:
            if sym in en_cartera or sym not in series:
                continue
            previos = series[sym][series[sym].index <= fecha]
            if previos.empty:
                continue
            precio = float(previos.iloc[-1])
            compras.append((fecha, sym, TICKET_EUR * fxd / precio))
            en_cartera.add(sym)

    dias = calendario[calendario >= compras[0][0]]
    valores, aportes = [], []
    posiciones: dict[str, float] = {}
    por_fecha: dict[pd.Timestamp, list] = {}
    for f, sym, acc in compras:
        por_fecha.setdefault(f, []).append((sym, acc))

    for d in dias:
        aporte = 0.0
        for sym, acc in por_fecha.get(d, []):
            posiciones[sym] = posiciones.get(sym, 0.0) + acc
            aporte += TICKET_EUR
        fxd = fx.rate("EUR", d)
        total = 0.0
        for sym, acc in posiciones.items():
            serie = series[sym]
            hasta = serie[serie.index <= d]
            if len(hasta):
                total += acc * float(hasta.iloc[-1]) / fxd
        valores.append(total)
        aportes.append(aporte)

    curva = pd.DataFrame({"valor": valores, "aporte": aportes}, index=dias)
    invertido = float(curva["aporte"].sum())
    final = float(curva["valor"].iloc[-1])

    flujos = [(f, -TICKET_EUR) for f, _, _ in compras] + [(dias[-1], final)]
    flujos.sort(key=lambda x: x[0])

    m = metricas(curva)
    return {
        "posiciones": len(compras), "invertido": invertido, "valor": final,
        "pnl": final - invertido, "pnl_%": (final / invertido - 1) * 100,
        "twr_total_%": m["twr"] * 100, "twr_anual_%": m["twr_anual"] * 100,
        "tir_anual_%": (tir(flujos) or float("nan")) * 100,
        "max_dd_%": m["max_dd"] * 100, "vol_anual_%": m["vol"] * 100,
        "sharpe": m["sharpe"],
    }, curva, compras


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
    log.info("%d lunes: %s .. %s", len(lunes), lunes[0].date(), lunes[-1].date())

    destino = AQUI / f"senales_top{GUARDAR_TOP}_{REGION}.json"
    if destino.exists():
        señales = json.loads(destino.read_text())
        log.info("señales reutilizadas de %s", destino.name)
    else:
        señales = []
        for i, fecha in enumerate(lunes, 1):
            pasado = {s: as_of(d, fecha, fallback_lag_days=60) for s, d in base.items()}
            pasado = {s: d for s, d in pasado.items() if d.prices is not None}
            precios = {s: d.prices for s, d in pasado.items()}
            lote = [Candidate(symbol=s, name=por_symbol[s].name, sector=d.sector,
                              industry=d.industry, market_cap=d.profile.get("marketCap"),
                              avg_volume=float(d.prices["Volume"].iloc[-63:].mean()),
                              price=float(d.prices["Close"].iloc[-1]),
                              currency=por_symbol[s].currency,
                              exchange=por_symbol[s].exchange)
                    for s, d in pasado.items()]
            vivos, _ = universe_mod.apply_cheap_gates(
                lote, cfg, provider=None, rates=lambda cur: fx.rates(cur, fecha))
            vivos, _ = universe_mod.apply_trend_gate(vivos, precios, cfg)
            if not vivos:
                continue
            reg = compute_regime(pasado[vivos[0].symbol].benchmark, precios, cfg)
            sc = score_universe({c.symbol: pasado[c.symbol] for c in vivos}, vivos,
                                region, cfg, regime_multiplier=reg.multiplier)
            top = sorted(sc.results, key=lambda r: r.score_final, reverse=True)[:GUARDAR_TOP]
            señales.append({"fecha": fecha.isoformat(), "regimen": reg.multiplier,
                            "puntuados": len(sc.results),
                            "top": [r.symbol for r in top],
                            "scores": [round(r.score_final, 2) for r in top]})
            if i % 10 == 0 or i == len(lunes):
                log.info("%d/%d %s", i, len(lunes), fecha.date())
        destino.write_text(json.dumps(señales, indent=1))
        log.info("señales guardadas en %s", destino.name)

    # --- referencias: el mismo dinero, los mismos días, en un índice -----
    # Dos varas distintas y las dos hacen falta:
    #  - el índice de la propia región mide habilidad (¿bate el screener a lo
    #    que tenía a mano?),
    #  - el SPY mide si valía la pena salir de EEUU para empezar.
    refs = [(region.benchmark, region.currency)]
    if region.benchmark != "SPY":
        refs.append(("SPY", "USD"))
    idx = {t: cierres(provider.prices.history([t], period=PERIOD)[t]) for t, _ in refs}

    filas, curvas = [], {}
    for n in NS:
        r, curva, compras = simular(señales, n, series, fx, calendario)
        r["cartera"] = f"top {n}"
        filas.append(r)
        curvas[f"top {n}"] = curva

        # cada compra del top N es una aportación: se replica una a una
        falsas = [{"fecha": f.isoformat(), "top": ["__SPY__"]} for f, _, _ in compras]
        for ticker, divisa in refs:
            rs, curva_ref, _ = simular_spy(falsas, idx[ticker], fx, calendario, divisa)
            rs["cartera"] = f"{ticker} (flujos de top {n})"
            filas.append(rs)
            curvas[f"{ticker} top {n}"] = curva_ref

    d = pd.DataFrame(filas)[["cartera", "posiciones", "invertido", "valor", "pnl", "pnl_%",
                             "twr_anual_%", "tir_anual_%", "max_dd_%", "vol_anual_%", "sharpe"]]
    d.to_csv(AQUI / f"comparativa_top_n{SUFIJO}.csv", index=False)

    print("\n" + "=" * 112)
    print(f"TOP N SEMANAL SIN STOP · {REGION} · {lunes[0].date()} → {hoy.date()} · "
          f"{len(señales)} semanas · ticket {TICKET_EUR:.0f} €")
    print("=" * 112)
    print(d.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # --- rentabilidad por año natural (TWR, comparable con el índice) ----
    print("\nTWR por año natural (%):")
    años = sorted({d_.year for d_ in curvas["top 5"].index})
    tabla = {}
    for nombre, curva in curvas.items():
        v, a = curva["valor"], curva["aporte"]
        base_ = v.shift(1).fillna(0.0) + a
        diario = (v / base_.where(base_ > 0)).fillna(1.0) - 1.0
        tabla[nombre] = {y: ((1 + diario[diario.index.year == y]).prod() - 1) * 100
                         for y in años}
    print(pd.DataFrame(tabla).T.to_string(float_format=lambda v: f"{v:+.2f}"))

    # Se guarda TAMBIÉN el aporte diario: sin él, `pct_change` sobre el valor
    # cuenta cada ingreso de ticket como si fuera rentabilidad, y cualquier
    # comparación posterior entre curvas sale inflada.
    salida = {}
    for k, c in curvas.items():
        salida[k] = c["valor"]
        salida[f"{k} __aporte"] = c["aporte"]
    pd.DataFrame(salida).to_csv(AQUI / f"curvas_valor{SUFIJO}.csv")


def simular_spy(falsas, spy, fx, calendario, divisa="USD"):
    """Mismos importes, mismos días, todo al índice de referencia.

    `divisa` es la moneda en que cotiza el índice: el SPY y el EEM en USD, pero
    el STOXX en EUR y el KOSPI en KRW. Convertir todo como si fuera USD inflaría
    o hundiría la referencia según el par, que es justo el error que haría
    parecer que la estrategia bate a su índice cuando solo bate al cambio.
    """
    compras = []
    for s in falsas:
        fecha = pd.Timestamp(s["fecha"])
        # EUR -> USD -> divisa del índice
        fxd = fx.rate("EUR", fecha) / fx.rate(divisa, fecha)
        previos = spy[spy.index <= fecha]
        if previos.empty:
            continue
        compras.append((fecha, "__SPY__", TICKET_EUR * fxd / float(previos.iloc[-1])))

    dias = calendario[calendario >= compras[0][0]]
    por_fecha = {}
    for f, _, acc in compras:
        por_fecha[f] = por_fecha.get(f, 0.0) + acc

    conteo = {}
    for f, _, _ in compras:
        conteo[f] = conteo.get(f, 0) + 1

    acciones, valores = 0.0, []
    for d in dias:
        acciones += por_fecha.get(d, 0.0)
        hasta = spy[spy.index <= d]
        fxv = fx.rate("EUR", d) / fx.rate(divisa, d)
        valores.append(acciones * float(hasta.iloc[-1]) / fxv if len(hasta) else 0.0)
    aportes = [TICKET_EUR * conteo.get(d, 0) for d in dias]

    curva = pd.DataFrame({"valor": valores, "aporte": aportes}, index=dias)
    invertido = float(curva["aporte"].sum())
    final = float(curva["valor"].iloc[-1])
    flujos = [(f, -TICKET_EUR) for f, _, _ in compras] + [(dias[-1], final)]
    flujos.sort(key=lambda x: x[0])
    m = metricas(curva)
    return {"posiciones": len(compras), "invertido": invertido, "valor": final,
            "pnl": final - invertido, "pnl_%": (final / invertido - 1) * 100,
            "twr_total_%": m["twr"] * 100, "twr_anual_%": m["twr_anual"] * 100,
            "tir_anual_%": (tir(flujos) or float("nan")) * 100,
            "max_dd_%": m["max_dd"] * 100, "vol_anual_%": m["vol"] * 100,
            "sharpe": m["sharpe"]}, curva, compras


main()
