"""Backtest point-in-time (§11).

    python -m screener.backtest --region us

En cada fecha de rebalanceo reconstruye el universo tal como se veía ese día
(`pointintime.as_of`) y lo puntúa con **la misma** `engine.score_universe` que la
corrida real, para que el backtest no mida una reimplementación que se
desincroniza del motor.

Las cuatro trampas del §11, y qué se hace con cada una:

- **Look-ahead / point-in-time.** Resuelto: cada estado financiero se retrasa a
  su fecha de report real (`earnings_dates`), no a su fecha de periodo.
- **Percentiles sin look-ahead.** Resuelto por construcción: el pool de percentil
  de la fecha t se calcula solo con los datos recortados a t.
- **FX histórico.** Resuelto: los gates de tamaño y liquidez usan el tipo de
  cambio de la fecha, inyectado en `apply_cheap_gates`.
- **Survivorship bias.** **NO resuelto, y no se puede con Yahoo**: el universo
  parte del screener de hoy, así que no contiene deslistadas ni quebradas. El
  resultado está sesgado al alza y el informe lo dice en cada corrida.

Otra limitación honesta: `eps_revisions` es una foto de hoy sin vintages, así que
B8 no es backtesteable y se desactiva. Sus 10 puntos de peso se reparten solos.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from screener import universe as universe_mod
from screener.config import Config, load_config
from screener.data.provider import DataProvider, build_provider
from screener.engine import build_sector_indices, fetch_fundamentals, score_universe
from screener.models import Candidate, Region, TickerData
from screener.pointintime import as_of
from screener.regime import compute_regime

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FX histórico
# ---------------------------------------------------------------------------
class FxHistory:
    """Tipos de cambio por fecha. Convertir con el de hoy falsea los gates."""

    def __init__(self, provider: DataProvider, period: str) -> None:
        self.provider = provider
        self.period = period
        self._series: dict[str, pd.Series | None] = {}

    def _load(self, code: str) -> pd.Series | None:
        if code not in self._series:
            series = None
            if code != "USD":
                series = self.provider.prices.close_series(f"{code}USD=X", period=self.period)
                if series is not None and len(series):
                    index = pd.DatetimeIndex(series.index)
                    series.index = index.tz_localize(None) if index.tz is not None else index
                    series = series.sort_index()
            self._series[code] = series
        return self._series[code]

    def rate(self, currency: str, when: pd.Timestamp) -> float:
        """USD por unidad mayor de la divisa en esa fecha (el último dato previo)."""
        code = (currency or "USD").upper()
        if code == "USD":
            return 1.0
        series = self._load(code)
        if series is None or series.empty:
            return 1.0
        past = series.loc[series.index <= when]
        if past.empty:
            return float(series.iloc[0])
        return float(past.iloc[-1])

    def rates(self, currency: str, when: pd.Timestamp) -> tuple[float, float]:
        """(unidad cotizada, unidad de capitalización) — difieren en GBp."""
        major = self.rate(currency, when)
        quoted = major / 100.0 if universe_mod.is_subunit(currency or "") else major
        return quoted, major


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    """Una fecha de rebalanceo evaluada."""

    date: pd.Timestamp
    regime: float
    scored: int
    alerts: list[dict] = field(default_factory=list)
    universe_forward: dict[int, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    region: str
    snapshots: list[Snapshot] = field(default_factory=list)
    horizons: list[int] = field(default_factory=list)
    universe_size: int = 0
    disabled_metrics: list[str] = field(default_factory=list)

    def alerts_frame(self) -> pd.DataFrame:
        rows = [alert for snapshot in self.snapshots for alert in snapshot.alerts]
        return pd.DataFrame(rows)

    def universe_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"date": s.date, "regime": s.regime, "scored": s.scored,
                 **{f"fwd_{h}": s.universe_forward.get(h) for h in self.horizons}}
                for s in self.snapshots
            ]
        )


# ---------------------------------------------------------------------------
# Retornos forward
# ---------------------------------------------------------------------------
def forward_return(prices: pd.DataFrame | None, when: pd.Timestamp, horizon: int) -> float | None:
    """Retorno desde el cierre de `when` hasta `horizon` sesiones después.

    None si la ventana no ha terminado todavía: contarla como 0 sesgaría el
    resultado hacia la nada y las últimas fechas son justo las más frescas.
    """
    if prices is None or prices.empty:
        return None
    index = pd.DatetimeIndex(prices.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close = prices["Close"]

    mask = index <= when
    if not mask.any():
        return None
    start = int(mask.sum()) - 1
    end = start + horizon
    if end >= len(close):
        return None

    entry, exit_ = float(close.iloc[start]), float(close.iloc[end])
    if entry <= 0:
        return None
    return exit_ / entry - 1.0


# ---------------------------------------------------------------------------
# Motor del backtest
# ---------------------------------------------------------------------------
def rebalance_dates(
    prices: dict[str, pd.DataFrame], start: pd.Timestamp | None, end: pd.Timestamp | None, freq: str
) -> list[pd.Timestamp]:
    """Fechas de sesión reales, no de calendario: evita caer en festivos."""
    sessions = set()
    for frame in prices.values():
        index = pd.DatetimeIndex(frame.index)
        sessions.update(index.tz_localize(None) if index.tz is not None else index)
    if not sessions:
        return []

    calendar = pd.DatetimeIndex(sorted(sessions))
    if start is not None:
        calendar = calendar[calendar >= start]
    if end is not None:
        calendar = calendar[calendar <= end]
    if calendar.empty:
        return []

    rule = {"monthly": "ME", "weekly": "W", "quarterly": "QE"}.get(freq, "ME")
    grouped = pd.Series(calendar, index=calendar).resample(rule).last().dropna()
    return [pd.Timestamp(d) for d in grouped.tolist()]


def _candidate_as_of(
    base: Candidate, data: TickerData, when: pd.Timestamp, fx: FxHistory
) -> Candidate:
    """Candidato con precio, volumen y capitalización de la fecha."""
    price = avg_volume = None
    if data.prices is not None and len(data.prices):
        price = float(data.prices["Close"].iloc[-1])
        avg_volume = float(data.prices["Volume"].iloc[-63:].mean())

    return Candidate(
        symbol=base.symbol,
        name=base.name,
        sector=data.sector or base.sector,
        industry=data.industry or base.industry,
        market_cap=data.profile.get("marketCap"),
        avg_volume=avg_volume,
        price=price,
        currency=base.currency,
        exchange=base.exchange,
        is_watchlist=base.is_watchlist,
    )


def run_backtest(
    region: Region,
    cfg: Config,
    provider: DataProvider | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
) -> BacktestResult:
    provider = provider or build_provider(region, cfg)
    params = cfg.raw.get("backtest") or {}
    period = str(params.get("price_history_period", "10y"))
    lag = int(params.get("publication_lag_days", 60))
    horizons = [int(h) for h in params.get("horizons_days", [21, 63, 126])]
    freq = str(params.get("rebalance", "monthly"))

    result = BacktestResult(region=region.key, horizons=horizons)

    # 1. Universo de hoy. Es el sesgo de supervivencia que Yahoo no permite evitar.
    candidates = universe_mod.build_candidates(region, provider, cfg)
    result.universe_size = len(candidates)
    log.info("universo (sesgado a supervivientes): %d", len(candidates))

    # 2. Historia completa de precios y fundamentales, una sola vez.
    prices = provider.prices.history([c.symbol for c in candidates], period=period)
    candidates = [c for c in candidates if c.symbol in prices]
    log.info("con historia de precios: %d", len(candidates))
    if not candidates:
        return result

    fundamentals = fetch_fundamentals(
        provider, [c.symbol for c in candidates], int(cfg.data["max_workers"])
    )
    benchmark = provider.prices.close_series(region.benchmark, period=period)
    sector_indices = build_sector_indices(candidates, prices)
    fx = FxHistory(provider, period)

    base: dict[str, TickerData] = {}
    by_symbol: dict[str, Candidate] = {}
    for candidate in candidates:
        bundle = fundamentals.get(candidate.symbol)
        if bundle is None:
            continue
        profile, statements, estimates = bundle
        sector = profile.get("sector") or candidate.sector
        if not universe_mod.passes_industry(profile.get("industry") or candidate.industry, cfg):
            continue
        if not universe_mod.passes_country(profile.get("country"), region):
            continue
        candidate.sector = sector
        by_symbol[candidate.symbol] = candidate
        base[candidate.symbol] = TickerData(
            symbol=candidate.symbol,
            asof=pd.Timestamp.today().date(),
            sector=sector,
            industry=profile.get("industry"),
            prices=prices.get(candidate.symbol),
            benchmark=benchmark,
            sector_index=sector_indices.get(sector) if sector else None,
            profile=profile,
            income_q=statements.income_q,
            income_a=statements.income_a,
            cashflow_q=statements.cashflow_q,
            cashflow_a=statements.cashflow_a,
            balance_q=statements.balance_q,
            balance_a=statements.balance_a,
            earnings_dates=estimates.earnings_dates,
            eps_revisions=estimates.eps_revisions,
            eps_trend=estimates.eps_trend,
            shares_full=estimates.shares_full,
        )

    dates = rebalance_dates(
        prices,
        pd.Timestamp(start) if start else None,
        pd.Timestamp(end) if end else None,
        freq,
    )
    log.info("%d fechas de rebalanceo (%s)", len(dates), freq)

    # 3. Una pasada por fecha, todo recortado a lo conocido entonces.
    for when in dates:
        snapshot = _evaluate_date(when, base, by_symbol, region, cfg, fx, lag, horizons)
        if snapshot is not None:
            result.snapshots.append(snapshot)
            log.info(
                "%s: %d puntuados, %d alertas, régimen %.3f",
                when.date(), snapshot.scored, len(snapshot.alerts), snapshot.regime,
            )

    if result.snapshots:
        result.disabled_metrics = ["estimate_revisions"]
    return result


def _evaluate_date(
    when: pd.Timestamp,
    base: dict[str, TickerData],
    by_symbol: dict[str, Candidate],
    region: Region,
    cfg: Config,
    fx: FxHistory,
    lag: int,
    horizons: list[int],
) -> Snapshot | None:
    past = {s: as_of(d, when, fallback_lag_days=lag) for s, d in base.items()}
    past = {s: d for s, d in past.items() if d.prices is not None}
    if not past:
        return None

    prices_then = {s: d.prices for s, d in past.items()}
    candidates_then = [_candidate_as_of(by_symbol[s], past[s], when, fx) for s in past]

    survivors, _ = universe_mod.apply_cheap_gates(
        candidates_then, cfg, provider=None, rates=lambda currency: fx.rates(currency, when)
    )
    survivors, _ = universe_mod.apply_trend_gate(survivors, prices_then, cfg)
    if not survivors:
        return None

    reading = compute_regime(past[survivors[0].symbol].benchmark, prices_then, cfg)
    tickers = {c.symbol: past[c.symbol] for c in survivors}
    scoring = score_universe(
        tickers, survivors, region, cfg, regime_multiplier=reading.multiplier
    )
    if not scoring.results:
        return None

    snapshot = Snapshot(date=when, regime=reading.multiplier, scored=len(scoring.results))

    # Retornos forward con la serie COMPLETA: es lo único que puede mirar al futuro.
    universe_returns: dict[int, list[float]] = {h: [] for h in horizons}
    for entry in scoring.results:
        full = base[entry.symbol].prices
        returns = {h: forward_return(full, when, h) for h in horizons}
        for h, value in returns.items():
            if value is not None:
                universe_returns[h].append(value)
        if entry.alert:
            snapshot.alerts.append(
                {
                    "date": when,
                    "symbol": entry.symbol,
                    "sector": entry.sector,
                    "a_pct": entry.a_pct,
                    "b_pct": entry.b_pct,
                    "regime": entry.regime,
                    "score": entry.score_final,
                    **{f"fwd_{h}": returns[h] for h in horizons},
                }
            )

    snapshot.universe_forward = {
        h: (float(pd.Series(v).mean()) if v else None) for h, v in universe_returns.items()
    }
    return snapshot


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------
def summarize(result: BacktestResult) -> str:
    lines = [
        "",
        "=" * 74,
        f"BACKTEST {result.region}",
        "=" * 74,
        "⚠  SESGO DE SUPERVIVENCIA NO CORREGIDO: el universo son los "
        f"{result.universe_size} valores",
        "   que cotizan HOY. Yahoo no sirve constituyentes históricos, así que no hay",
        "   deslistadas ni quebradas. Los retornos están sesgados al alza.",
    ]
    if result.disabled_metrics:
        lines.append(
            f"⚠  Métricas desactivadas por no ser backtesteables: "
            f"{', '.join(result.disabled_metrics)} (eps_revisions es una foto de hoy)."
        )

    alerts = result.alerts_frame()
    universe = result.universe_frame()
    lines += [
        "",
        f"fechas evaluadas: {len(result.snapshots)}"
        + (f"  ({universe['date'].min().date()} → {universe['date'].max().date()})"
           if not universe.empty else ""),
        f"alertas totales:  {len(alerts)}"
        + (f"  ({len(alerts) / len(result.snapshots):.1f} por fecha)" if result.snapshots else ""),
    ]

    if alerts.empty:
        lines += ["", "Ninguna alerta en el periodo: no hay nada que medir."]
        return "\n".join(lines)

    lines += [
        "",
        f"{'horizonte':>10} {'alertas':>8} {'media':>9} {'mediana':>9} {'universo':>9} "
        f"{'exceso':>8} {'aciertos':>9}",
        "-" * 74,
    ]
    for horizon in result.horizons:
        column = f"fwd_{horizon}"
        values = alerts[column].dropna()
        if values.empty:
            continue
        universe_mean = universe[column].dropna().mean()
        excess = values.mean() - universe_mean if pd.notna(universe_mean) else float("nan")
        lines.append(
            f"{horizon:>7}d {len(values):>8} {values.mean() * 100:>8.2f}% "
            f"{values.median() * 100:>8.2f}% {universe_mean * 100:>8.2f}% "
            f"{excess * 100:>7.2f}% {(values > 0).mean() * 100:>8.1f}%"
        )

    lines += [
        "",
        "'universo' es la media de TODOS los valores puntuados en las mismas fechas:",
        "es el listón que hay que batir, no el 0%.",
    ]
    return "\n".join(lines)


def sweep_thresholds(result: BacktestResult, grid: list[float], horizon: int) -> pd.DataFrame:
    """Rejilla de umbrales A/B sobre las alertas ya calculadas.

    Los percentiles de cada fecha no dependen del umbral, así que se puede
    reevaluar el corte sin repetir el backtest. Ojo: esto solo re-filtra los que
    YA alertaron con el umbral de config, así que la rejilla debe ir de ese
    umbral hacia arriba.
    """
    alerts = result.alerts_frame()
    column = f"fwd_{horizon}"
    if alerts.empty or column not in alerts:
        return pd.DataFrame()

    rows = []
    for a_threshold in grid:
        for b_threshold in grid:
            subset = alerts[(alerts["a_pct"] >= a_threshold) & (alerts["b_pct"] >= b_threshold)]
            values = subset[column].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "a_threshold": a_threshold,
                    "b_threshold": b_threshold,
                    "alertas": len(subset),
                    "por_fecha": len(subset) / max(len(result.snapshots), 1),
                    "media_pct": values.mean() * 100,
                    "mediana_pct": values.median() * 100,
                    "aciertos_pct": (values > 0).mean() * 100,
                }
            )
    return pd.DataFrame(rows).sort_values("media_pct", ascending=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="screener.backtest", description=__doc__)
    parser.add_argument("--region", default="us")
    parser.add_argument("--config", default=None)
    parser.add_argument("--start", default=None, help="AAAA-MM-DD")
    parser.add_argument("--end", default=None, help="AAAA-MM-DD")
    parser.add_argument("--sweep-horizon", type=int, default=None,
                        help="horizonte en sesiones para el barrido de umbrales")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    cfg = load_config(args.config)
    params = cfg.raw.get("backtest") or {}
    region = cfg.region(args.region)

    result = run_backtest(
        region, cfg, start=args.start or params.get("start"), end=args.end or params.get("end")
    )
    print(summarize(result))

    grid = [float(g) for g in params.get("threshold_grid") or []]
    horizon = args.sweep_horizon or (result.horizons[1] if len(result.horizons) > 1 else None)
    if grid and horizon:
        sweep = sweep_thresholds(result, grid, horizon)
        if not sweep.empty:
            print(f"\nBarrido de umbrales a {horizon} sesiones "
                  f"(arranque {cfg.alerting['a_threshold']}/{cfg.alerting['b_threshold']}):\n")
            print(sweep.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    output = Path(cfg.run["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    alerts = result.alerts_frame()
    if not alerts.empty:
        path = output / f"backtest_{region.key}.csv"
        alerts.to_csv(path, index=False)
        print(f"\nAlertas del backtest: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
