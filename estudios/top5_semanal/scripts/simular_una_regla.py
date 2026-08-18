"""Simulación: 100 € en cada valor del top 5, cada lunes, últimos 3 meses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SALIDA = Path(__file__).resolve().parent.parent / "posiciones"

import logging
from dataclasses import dataclass, field

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
log = logging.getLogger("sim")

REGION = "us"
TICKET_EUR = 100.0
TAKE_PROFIT = 0.20
MESES = 18
TOP_N = 5


@dataclass
class Posicion:
    symbol: str
    entrada: pd.Timestamp
    precio_entrada: float
    fx_entrada: float          # USD por 1 EUR
    acciones: float
    invertido_eur: float
    salida: pd.Timestamp | None = None
    precio_salida: float | None = None
    fx_salida: float | None = None
    motivo: str = ""

    def valor_eur(self, precio: float, fx: float) -> float:
        return self.acciones * precio / fx

    @property
    def cerrada(self) -> bool:
        return self.salida is not None


def precio_en(prices: pd.DataFrame, fecha: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    idx = pd.DatetimeIndex(prices.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    mask = idx <= fecha
    if not mask.any():
        return None
    pos = int(mask.sum()) - 1
    return idx[pos], float(prices["Close"].iloc[pos])


def main() -> None:
    cfg = load_config()
    region = cfg.region(REGION)
    provider = build_provider(region, cfg)
    # 18 meses de simulación + los ~13 meses de histórico que exige el gate
    # (A1 mira 252 sesiones terminando 21 atrás) = 31 meses mínimos.
    period = "4y"

    candidatos = universe_mod.build_candidates(region, provider, cfg)
    prices = provider.prices.history([c.symbol for c in candidatos], period=period)
    candidatos = [c for c in candidatos if c.symbol in prices]
    log.info("universo con precios: %d", len(candidatos))

    fundamentales = fetch_fundamentals(
        provider, [c.symbol for c in candidatos], int(cfg.data["max_workers"])
    )
    benchmark = provider.prices.close_series(region.benchmark, period=period)
    indices = build_sector_indices(candidatos, prices)
    fx = FxHistory(provider, period)

    base: dict[str, TickerData] = {}
    por_symbol: dict[str, Candidate] = {}
    for c in candidatos:
        paquete = fundamentales.get(c.symbol)
        if paquete is None:
            continue
        perfil, estados, estimaciones = paquete
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
            profile=perfil,
            income_q=estados.income_q, income_a=estados.income_a,
            cashflow_q=estados.cashflow_q, cashflow_a=estados.cashflow_a,
            balance_q=estados.balance_q, balance_a=estados.balance_a,
            earnings_dates=estimaciones.earnings_dates,
            eps_revisions=estimaciones.eps_revisions, eps_trend=estimaciones.eps_trend,
            shares_full=estimaciones.shares_full,
        )
    log.info("con fundamentales: %d", len(base))

    # --- lunes de los últimos 3 meses -----------------------------------
    sesiones = set()
    for f in prices.values():
        idx = pd.DatetimeIndex(f.index)
        sesiones.update(idx.tz_localize(None) if idx.tz is not None else idx)
    calendario = pd.DatetimeIndex(sorted(sesiones))
    hoy = calendario.max()
    desde = hoy - pd.DateOffset(months=MESES)

    # Semanas de lunes a domingo: `W-SUN` termina en domingo, así que la primera
    # sesión de cada grupo es el lunes (o el martes si el lunes fue festivo).
    # Con `W-MON` los grupos van de martes a lunes y salía el martes.
    ventana = calendario[(calendario >= desde) & (calendario <= hoy)]
    lunes = sorted(
        grupo.min()
        for _, grupo in pd.Series(ventana, index=ventana).groupby(
            pd.Grouper(freq="W-SUN")
        )
        if len(grupo)
    )
    for d in lunes:
        assert d.dayofweek <= 2, f"{d.date()} no es principio de semana"
    log.info("%d fechas de compra: %s .. %s", len(lunes), lunes[0].date(), lunes[-1].date())

    # --- simulación ------------------------------------------------------
    posiciones: list[Posicion] = []
    abiertas: dict[str, Posicion] = {}
    historial = []

    for fecha in lunes:
        # 1. cerrar por +20% lo que toque ANTES de comprar
        for symbol, pos in list(abiertas.items()):
            serie = base[symbol].prices
            idx = pd.DatetimeIndex(serie.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            tramo = serie[(idx > pos.entrada) & (idx <= fecha)]
            if tramo.empty:
                continue
            objetivo = pos.precio_entrada * (1 + TAKE_PROFIT)
            alcanza = tramo[tramo["Close"] >= objetivo]
            if not alcanza.empty:
                dia = pd.Timestamp(alcanza.index[0])
                if dia.tz is not None:
                    dia = dia.tz_localize(None)
                pos.salida, pos.precio_salida = dia, float(alcanza["Close"].iloc[0])
                pos.fx_salida = fx.rate("EUR", dia)
                pos.motivo = f"+20% el {dia.date()}"
                del abiertas[symbol]

        # 2. puntuar el universo tal como se veía ese lunes
        pasado = {s: as_of(d, fecha, fallback_lag_days=60) for s, d in base.items()}
        pasado = {s: d for s, d in pasado.items() if d.prices is not None}
        precios_hoy = {s: d.prices for s, d in pasado.items()}
        cands = []
        for s, d in pasado.items():
            c0 = por_symbol[s]
            p = float(d.prices["Close"].iloc[-1])
            cands.append(Candidate(
                symbol=s, name=c0.name, sector=d.sector, industry=d.industry,
                market_cap=d.profile.get("marketCap"),
                avg_volume=float(d.prices["Volume"].iloc[-63:].mean()),
                price=p, currency=c0.currency, exchange=c0.exchange,
            ))
        vivos, _ = universe_mod.apply_cheap_gates(
            cands, cfg, provider=None, rates=lambda cur: fx.rates(cur, fecha)
        )
        vivos, _ = universe_mod.apply_trend_gate(vivos, precios_hoy, cfg)
        if not vivos:
            continue
        regimen = compute_regime(pasado[vivos[0].symbol].benchmark, precios_hoy, cfg)
        scoring = score_universe(
            {c.symbol: pasado[c.symbol] for c in vivos}, vivos, region, cfg,
            regime_multiplier=regimen.multiplier,
        )
        ranking = sorted(scoring.results, key=lambda r: r.score_final, reverse=True)[:TOP_N]

        # 3. comprar los que no estén ya en cartera
        fx_dia = fx.rate("EUR", fecha)
        comprados = []
        for r in ranking:
            if r.symbol in abiertas:
                comprados.append(f"{r.symbol}(ya)")
                continue
            dato = precio_en(base[r.symbol].prices, fecha)
            if dato is None:
                continue
            _, precio = dato
            pos = Posicion(
                symbol=r.symbol, entrada=fecha, precio_entrada=precio, fx_entrada=fx_dia,
                acciones=(TICKET_EUR * fx_dia) / precio, invertido_eur=TICKET_EUR,
            )
            posiciones.append(pos)
            abiertas[r.symbol] = pos
            comprados.append(r.symbol)
        historial.append({
            "fecha": fecha, "regimen": regimen.multiplier,
            "top5": ", ".join(r.symbol for r in ranking), "comprados": ", ".join(comprados),
        })
        log.info("%s  top5=%s", fecha.date(), ", ".join(r.symbol for r in ranking))

    # 4. cierre final de las que llegaron a +20% después del último lunes
    for symbol, pos in list(abiertas.items()):
        serie = base[symbol].prices
        idx = pd.DatetimeIndex(serie.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        tramo = serie[idx > pos.entrada]
        objetivo = pos.precio_entrada * (1 + TAKE_PROFIT)
        alcanza = tramo[tramo["Close"] >= objetivo]
        if not alcanza.empty:
            dia = pd.Timestamp(alcanza.index[0])
            if dia.tz is not None:
                dia = dia.tz_localize(None)
            pos.salida, pos.precio_salida = dia, float(alcanza["Close"].iloc[0])
            pos.fx_salida = fx.rate("EUR", dia)
            pos.motivo = f"+20% el {dia.date()}"
            del abiertas[symbol]

    pd.DataFrame(historial).to_csv(SALIDA / "sim_semanas.csv", index=False)

    # --- resultados -------------------------------------------------------
    fx_hoy = fx.rate("EUR", hoy)
    filas = []
    for pos in posiciones:
        ultimo = precio_en(base[pos.symbol].prices, hoy)
        precio_hoy = ultimo[1] if ultimo else pos.precio_entrada

        # escenario A: con la regla del +20%
        if pos.cerrada:
            valor_a = pos.valor_eur(pos.precio_salida, pos.fx_salida)
            estado = f"cerrada {pos.motivo}"
        else:
            valor_a = pos.valor_eur(precio_hoy, fx_hoy)
            estado = "abierta"
        # escenario B: sin cerrar nunca
        valor_b = pos.valor_eur(precio_hoy, fx_hoy)

        filas.append({
            "symbol": pos.symbol, "entrada": pos.entrada.date(),
            "precio_ent": pos.precio_entrada, "precio_hoy": precio_hoy,
            "ret_bruto_%": (precio_hoy / pos.precio_entrada - 1) * 100,
            "invertido": pos.invertido_eur,
            "valor_con_stop": valor_a, "valor_sin_stop": valor_b,
            "estado": estado,
        })

    d = pd.DataFrame(filas)
    d.to_csv(SALIDA / "sim_posiciones.csv", index=False)

    invertido = d["invertido"].sum()
    print("\n" + "=" * 78)
    print(f"SIMULACIÓN  ·  {REGION}  ·  {lunes[0].date()} → {hoy.date()}")
    print("=" * 78)
    print(f"semanas: {len(lunes)}   posiciones abiertas: {len(d)}   "
          f"tickets de {TICKET_EUR:.0f} €")
    print(f"INVERTIDO: {invertido:,.2f} €".replace(",", "."))
    print()
    for etiqueta, col in (("CON regla de +20%", "valor_con_stop"),
                          ("SIN cerrar ninguna", "valor_sin_stop")):
        valor = d[col].sum()
        pnl = valor - invertido
        print(f"{etiqueta:22s} valor hoy {valor:9,.2f} €   "
              f"P&L {pnl:+9,.2f} €   ({pnl / invertido * 100:+.2f}%)")
    print()
    cerradas = d[d["estado"].str.startswith("cerrada")]
    print(f"posiciones cerradas por +20%: {len(cerradas)} de {len(d)}")
    print(f"ganadoras (a precio de hoy):  {(d['ret_bruto_%'] > 0).sum()} de {len(d)}")
    print("\nDetalle:")
    print(d.sort_values("ret_bruto_%", ascending=False).to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))


main()
