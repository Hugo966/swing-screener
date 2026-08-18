"""Motor: batch por región (§2).

Los percentiles son cross-sectional, así que **no se puede puntuar un ticker
aislado**: primero se calcula todo el universo de la región y luego se rankea.
Este módulo orquesta el universo entero, no un símbolo.

    universo -> gates -> métricas -> percentiles -> paneles -> régimen -> decisión
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from screener import universe as universe_mod
from screener.data.provider import DataProvider, Estimates, Statements, build_provider
from screener.models import Candidate, GateResult, Region, TickerData, TickerResult
from screener.normalize import percentile_rank
from screener.panels import finalize_panel, prepare_panel
from screener.regime import RegimeReading, compute_regime

log = logging.getLogger(__name__)


@dataclass
class RegionRun:
    region: str
    asof: date
    results: list[TickerResult] = field(default_factory=list)
    regime: RegimeReading | None = None
    gate_log: list[GateResult] = field(default_factory=list)
    universe_size: int = 0
    scored: int = 0
    dropped_low_coverage: list[str] = field(default_factory=list)
    metric_coverage: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def alerts(self) -> list[TickerResult]:
        return [r for r in self.results if r.alert]

    def ranked(self) -> list[TickerResult]:
        return sorted(self.results, key=lambda r: r.score_final, reverse=True)


# ---------------------------------------------------------------------------
# Índice sectorial sintético
# ---------------------------------------------------------------------------
def build_sector_indices(
    candidates: list[Candidate], prices: dict[str, pd.DataFrame]
) -> dict[str, pd.Series]:
    """Índice equiponderado por sector desde los propios constituyentes.

    Evita depender de ETFs sectoriales, que no tienen equivalente limpio en
    Europa ni en la Fase 2. Se construye sobre **todo** el universo que pasó los
    gates baratos, no solo sobre los que pasaron el de tendencia: si no, el
    índice sectorial mediría solo valores en tendencia y A3/A4 estarían sesgados.
    """
    by_sector: dict[str, list[pd.Series]] = {}
    for candidate in candidates:
        frame = prices.get(candidate.symbol)
        if not candidate.sector or frame is None or frame.empty:
            continue
        close = frame["Close"].dropna()
        if len(close) < 60:
            continue
        base = float(close.iloc[0])
        if base <= 0:
            continue
        by_sector.setdefault(candidate.sector, []).append(close / base)

    indices: dict[str, pd.Series] = {}
    for sector, series_list in by_sector.items():
        if len(series_list) < 3:
            log.debug("sector %s con solo %d series: sin índice", sector, len(series_list))
            continue
        combined = pd.concat(series_list, axis=1).sort_index()
        indices[sector] = combined.mean(axis=1, skipna=True).dropna()
    return indices


# ---------------------------------------------------------------------------
# Descarga de fundamentales
# ---------------------------------------------------------------------------
def _fetch_one(provider: DataProvider, symbol: str) -> tuple[str, dict, Statements, Estimates]:
    profile = provider.fundamentals.profile(symbol)
    statements = provider.fundamentals.statements(symbol)
    estimates = provider.fundamentals.estimates(symbol)
    return symbol, profile, statements, estimates


def fetch_fundamentals(
    provider: DataProvider, symbols: list[str], max_workers: int
) -> dict[str, tuple[dict, Statements, Estimates]]:
    out: dict[str, tuple[dict, Statements, Estimates]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, provider, s): s for s in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                symbol, profile, statements, estimates = future.result()
                out[symbol] = (profile, statements, estimates)
            except Exception as exc:  # noqa: BLE001 — un ticker no tumba la corrida
                log.debug("fundamentales de %s fallaron: %s", symbol, exc)
            done += 1
            if done % 50 == 0:
                log.info("fundamentales: %d/%d", done, len(symbols))
    return out


# ---------------------------------------------------------------------------
# Decisión (§7)
# ---------------------------------------------------------------------------
def decide(result: TickerResult, cfg) -> None:
    """AND de dos umbrales altos por percentil, más el corte tras régimen.

    No se promedian los paneles (llenaría las alertas de empresas mediocres pero
    uniformes) ni se usa max (premia excelencia en una sola dimensión):
    B decide *si* la empresa merece la pena, A decide *si ahora*.
    """
    base = cfg.alerting
    thresholds = base["watchlist"] if result.is_watchlist else base

    a_threshold = float(thresholds["a_threshold"])
    b_threshold = float(thresholds["b_threshold"])
    final_cut = float(thresholds.get("final_cut", base["final_cut"]))

    result.score_final = result.combined

    if result.a_pct < a_threshold:
        result.alert_reason = f"A_pct {result.a_pct:.0f} < {a_threshold:.0f}"
    elif result.b_pct < b_threshold:
        result.alert_reason = f"B_pct {result.b_pct:.0f} < {b_threshold:.0f}"
    elif result.score_final < final_cut:
        result.alert_reason = f"score {result.score_final:.1f} < corte {final_cut:.0f}"
    else:
        result.alert = True
        result.alert_reason = (
            f"A {result.a_pct:.0f} / B {result.b_pct:.0f} / score {result.score_final:.1f}"
        )


# ---------------------------------------------------------------------------
# Corrida de una región
# ---------------------------------------------------------------------------
def run_region(region: Region, cfg, provider: DataProvider | None = None) -> RegionRun:
    provider = provider or build_provider(region, cfg)
    run = RegionRun(region=region.key, asof=date.today())
    period = str(cfg.data["price_history_period"])

    # 1-2. Universo y gates baratos (screener: tamaño, liquidez, sector)
    candidates = universe_mod.build_candidates(region, provider, cfg)
    run.universe_size = len(candidates)
    log.info("región %s: %d candidatos", region.key, len(candidates))

    candidates, cheap_log = universe_mod.apply_cheap_gates(candidates, cfg, provider)
    run.gate_log.extend(cheap_log)
    log.info("tras gates de tamaño/liquidez/sector: %d (%s)", len(candidates),
             universe_mod.summarize(cheap_log))
    if not candidates:
        return run

    # 3. Precios de toda la región: sirven al gate de tendencia, a la amplitud
    #    del régimen y a los índices sectoriales.
    prices = provider.prices.history([c.symbol for c in candidates], period=period)
    log.info("precios disponibles para %d/%d", len(prices), len(candidates))

    sector_indices = build_sector_indices(candidates, prices)
    benchmark = provider.prices.close_series(region.benchmark, period=period)
    run.regime = compute_regime(benchmark, prices, cfg)
    log.info("régimen %s: %.3f (%s)", region.key, run.regime.multiplier, run.regime.detail)

    # 4. Gate de tendencia
    survivors, trend_log = universe_mod.apply_trend_gate(candidates, prices, cfg)
    run.gate_log.extend(trend_log)
    log.info("tras gate de tendencia: %d (%s)", len(survivors), universe_mod.summarize(trend_log))
    if not survivors:
        return run

    # 5. Fundamentales solo de los supervivientes
    fundamentals = fetch_fundamentals(
        provider, [c.symbol for c in survivors], int(cfg.data["max_workers"])
    )

    tickers: dict[str, TickerData] = {}
    kept: list[Candidate] = []
    for candidate in survivors:
        bundle = fundamentals.get(candidate.symbol)
        if bundle is None:
            continue
        profile, statements, estimates = bundle

        sector = profile.get("sector") or candidate.sector
        industry = profile.get("industry") or candidate.industry
        if not universe_mod.passes_industry(industry, cfg):
            run.gate_log.append(GateResult(candidate.symbol, False, "industria", industry or ""))
            continue
        country = profile.get("country")
        if not universe_mod.passes_country(country, region):
            run.gate_log.append(GateResult(candidate.symbol, False, "domicilio", country or "s/d"))
            continue

        candidate.sector = sector
        candidate.industry = industry
        kept.append(candidate)
        tickers[candidate.symbol] = TickerData(
            symbol=candidate.symbol,
            asof=run.asof,
            sector=sector,
            industry=industry,
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

    if not tickers:
        return run

    scoring = score_universe(tickers, kept, region, cfg, regime_multiplier=(
        run.regime.multiplier if run.regime else 1.0
    ))
    run.results = scoring.results
    run.metric_coverage = scoring.metric_coverage
    run.dropped_low_coverage = scoring.dropped_low_coverage
    run.scored = len(run.results)

    log.info("región %s: %d puntuados, %d alertas", region.key, run.scored, len(run.alerts))
    return run


@dataclass
class Scoring:
    """Resultado de puntuar un universo ya construido."""

    results: list[TickerResult] = field(default_factory=list)
    metric_coverage: dict[str, dict[str, float]] = field(default_factory=dict)
    dropped_low_coverage: list[str] = field(default_factory=list)


def score_universe(
    tickers: dict[str, TickerData],
    candidates: list[Candidate],
    region: Region,
    cfg,
    *,
    regime_multiplier: float = 1.0,
) -> Scoring:
    """Pasos 5-7 de la spec: métricas -> percentiles -> paneles -> decisión.

    Es la parte del motor que no hace I/O, así que el backtest la reutiliza tal
    cual con datos recortados a una fecha pasada. Compartir esta función es lo
    que garantiza que el backtest mide el mismo scoring que la corrida real, en
    vez de una reimplementación que se desincroniza.
    """
    scoring = Scoring()

    momentum_inputs = prepare_panel("momentum", tickers, region.weights["momentum"], cfg)
    quality_inputs = prepare_panel("quality", tickers, region.weights["quality"], cfg)
    scoring.metric_coverage = {
        "momentum": momentum_inputs.metric_coverage,
        "quality": quality_inputs.metric_coverage,
    }
    if momentum_inputs.raw.empty or quality_inputs.raw.empty:
        return scoring

    # Suelo de cobertura por ticker, ANTES de percentilar: un valor con datos
    # basura no puede desplazar el percentil de los demás.
    min_coverage = float(cfg.coverage["min_panel_coverage"])
    coverage = pd.concat(
        [momentum_inputs.ticker_coverage(), quality_inputs.ticker_coverage()], axis=1
    ).min(axis=1)
    retained = coverage[coverage >= min_coverage].index
    scoring.dropped_low_coverage = sorted(set(coverage.index) - set(retained))
    if scoring.dropped_low_coverage:
        log.info(
            "descartados por cobertura < %.0f%%: %d",
            min_coverage * 100,
            len(scoring.dropped_low_coverage),
        )
    if len(retained) == 0:
        return scoring

    momentum_inputs.raw = momentum_inputs.raw.loc[retained]
    quality_inputs.raw = quality_inputs.raw.loc[retained]
    kept = [c for c in candidates if c.symbol in set(retained)]
    sectors = pd.Series({c.symbol: c.sector or "?" for c in kept})

    momentum = finalize_panel(momentum_inputs, sectors, cfg)
    quality = finalize_panel(quality_inputs, sectors, cfg)

    # Percentil del panel_raw dentro de la región -> A_pct / B_pct
    a_pct = percentile_rank(pd.Series({s: b.raw_score for s, b in momentum.items()})).fillna(50.0)
    b_pct = percentile_rank(pd.Series({s: b.raw_score for s, b in quality.items()})).fillna(50.0)

    for candidate in kept:
        symbol = candidate.symbol
        result = TickerResult(
            symbol=symbol,
            region=region.key,
            sector=candidate.sector,
            name=candidate.name,
            is_watchlist=candidate.is_watchlist,
            price=candidate.price,
            momentum=momentum.get(symbol),
            quality=quality.get(symbol),
            a_pct=float(a_pct.get(symbol, 50.0)),
            b_pct=float(b_pct.get(symbol, 50.0)),
            regime=regime_multiplier,
        )
        decide(result, cfg)
        scoring.results.append(result)

    return scoring
