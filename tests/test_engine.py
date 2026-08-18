"""Pipeline completo con un proveedor falso: universo -> gates -> ... -> decisión.

Sin red: valida el cableado del motor, no la calidad de los datos de Yahoo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.data.provider import DataProvider, Estimates, Statements
from screener.engine import build_sector_indices, run_region
from screener.models import Candidate
from tests.conftest import make_prices, statement

PERIODS = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]
QUARTERS = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
DAYS = 400


def price_frame(daily: float) -> pd.DataFrame:
    return make_prices(100.0 * np.cumprod(np.full(DAYS, 1.0 + daily)))


class FakeProvider:
    """Universo de 24 nombres: 16 en tendencia, 8 en caída."""

    def __init__(self, n_up=16, n_down=8, quality_scale=1.0):
        self.symbols = [f"UP{i}" for i in range(n_up)] + [f"DN{i}" for i in range(n_down)]
        self.sectors = {s: ("Technology" if i % 2 == 0 else "Industrials")
                        for i, s in enumerate(self.symbols)}
        self._prices = {}
        for i, symbol in enumerate(self.symbols):
            daily = 0.0008 + i * 0.00005 if symbol.startswith("UP") else -0.0015
            self._prices[symbol] = price_frame(daily)
        self._prices["SPY"] = price_frame(0.0006)
        self.quality_scale = quality_scale

    # -- PriceProvider --
    def history(self, symbols, *, period):
        return {s: self._prices[s] for s in symbols if s in self._prices}

    def close_series(self, symbol, *, period):
        frame = self._prices.get(symbol)
        return None if frame is None else frame["Close"]

    # -- UniverseProvider --
    def candidates(self, region):
        return [
            Candidate(
                symbol=symbol,
                name=symbol,
                sector=self.sectors[symbol],
                market_cap=5e9 + i * 1e9,
                avg_volume=1e6,
                price=float(self._prices[symbol]["Close"].iloc[-1]),
                currency="USD",
            )
            for i, symbol in enumerate(self.symbols)
        ]

    # -- FundamentalsProvider --
    def profile(self, symbol):
        return {
            "sector": self.sectors[symbol],
            "industry": "Software",
            "marketCap": 5e9,
            "enterpriseValue": 6e9,
            "currency": "USD",
        }

    def statements(self, symbol):
        rank = self.symbols.index(symbol) * self.quality_scale
        growth = 1.05 + rank * 0.02
        revenues = [100.0 * growth**k for k in range(4)]
        return Statements(
            income_a=statement(
                {"Total Revenue": revenues, "EBIT": [r * 0.2 for r in revenues],
                 "EBITDA": [r * 0.25 for r in revenues], "Net Income": [r * 0.15 for r in revenues],
                 "Operating Income": [r * 0.2 for r in revenues], "Tax Rate For Calcs": [0.21] * 4},
                PERIODS,
            ),
            income_q=statement(
                {"Total Revenue": [100.0, 105.0, 110.0, 115.0, 100.0 * growth],
                 "EBITDA": [25.0] * 5, "Net Income": [15.0] * 5,
                 "Operating Income": [20.0, 21.0, 22.0, 23.0, 20.0 * growth]},
                QUARTERS,
            ),
            cashflow_a=statement({"Free Cash Flow": [r * 0.18 for r in revenues],
                                  "Stock Based Compensation": [r * 0.05 for r in revenues]}, PERIODS),
            cashflow_q=statement({"Free Cash Flow": [18.0] * 5,
                                  "Stock Based Compensation": [5.0] * 5}, QUARTERS),
            balance_a=statement({"Invested Capital": [500.0] * 4, "Net Debt": [100.0] * 4}, PERIODS),
            balance_q=statement({"Invested Capital": [500.0] * 5, "Net Debt": [100.0] * 5}, QUARTERS),
        )

    def estimates(self, symbol):
        index = pd.to_datetime(["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"])
        rank = self.symbols.index(symbol)
        return Estimates(
            earnings_dates=pd.DataFrame(
                {"Reported EPS": [1.0] * 4, "EPS Estimate": [0.9] * 4,
                 "Surprise(%)": [float(rank)] * 4},
                index=index,
            ),
            eps_revisions=pd.DataFrame(
                {"upLast30days": [rank], "downLast30days": [1]}, index=["+1y"]
            ),
            eps_trend=pd.DataFrame(
                {"current": [10.0 + rank * 0.1], "90daysAgo": [10.0]}, index=["+1y"]
            ),
            shares_full=pd.Series(
                [100e6, 100e6], index=pd.to_datetime(["2022-01-01", "2025-01-01"])
            ),
        )


@pytest.fixture
def provider():
    fake = FakeProvider()
    return DataProvider(prices=fake, fundamentals=fake, universe=fake, name="fake")


@pytest.fixture
def run(cfg, provider):
    return run_region(cfg.region("us"), cfg, provider)


def test_pipeline_scores_only_the_trend_survivors(run):
    """El gate de tendencia descarta antes de puntuar: los DN no llegan al ranking."""
    assert run.universe_size == 24
    assert run.scored == 16
    assert all(r.symbol.startswith("UP") for r in run.results)

    failed = {g.symbol for g in run.gate_log if not g.passed and g.failed_gate == "tendencia"}
    assert failed == {f"DN{i}" for i in range(8)}


def test_percentiles_span_the_full_range(run):
    a_values = [r.a_pct for r in run.results]
    b_values = [r.b_pct for r in run.results]
    assert min(a_values) >= 0 and max(a_values) == pytest.approx(100.0)
    assert min(b_values) >= 0 and max(b_values) == pytest.approx(100.0)


def test_score_final_is_a_times_b_times_regime(run):
    for result in run.results:
        expected = result.a_pct * result.b_pct / 100.0 * run.regime.multiplier
        assert result.score_final == pytest.approx(expected)


def test_every_result_carries_its_breakdown(run):
    for result in run.results:
        assert result.momentum is not None and result.quality is not None
        assert len(result.momentum.metrics) > 0
        assert 0.0 <= result.momentum.coverage <= 1.0
        # la suma de contribuciones ES el panel_raw
        assert result.momentum.raw_score == pytest.approx(
            sum(m.contribution for m in result.momentum.metrics)
        )


def test_alerts_are_a_subset_that_clears_both_thresholds(run, cfg):
    a_threshold = float(cfg.alerting["a_threshold"])
    b_threshold = float(cfg.alerting["b_threshold"])
    for result in run.alerts:
        assert result.a_pct >= a_threshold
        assert result.b_pct >= b_threshold
        assert result.score_final >= float(cfg.alerting["final_cut"])


def test_regime_is_computed_over_the_whole_region(run, cfg):
    """Amplitud sobre los 24 nombres, no solo sobre los 16 que pasaron el gate."""
    assert run.regime is not None
    assert run.regime.breadth_pct == pytest.approx(16 / 24)
    assert float(cfg.regime["min_multiplier"]) <= run.regime.multiplier <= float(cfg.regime["max_multiplier"])


def test_excluded_industry_is_dropped_after_the_profile_is_known(cfg, provider):
    """El sector se filtra en el screener; la industria solo se sabe tras el perfil."""
    provider.fundamentals.profile = lambda symbol: {
        "sector": "Technology", "industry": "Banks - Regional",
        "marketCap": 5e9, "enterpriseValue": 6e9, "currency": "USD",
    }
    run = run_region(cfg.region("us"), cfg, provider)

    assert run.scored == 0
    assert all(g.failed_gate == "industria" for g in run.gate_log if not g.passed and g.symbol.startswith("UP"))


def test_sector_index_is_built_from_the_broad_universe():
    """Si se construyera solo con los supervivientes, A3/A4 medirían solo alcistas."""
    fake = FakeProvider()
    candidates = fake.candidates(region=None)
    prices = {c.symbol: fake._prices[c.symbol] for c in candidates}

    indices = build_sector_indices(candidates, prices)

    assert set(indices) == {"Technology", "Industrials"}
    for series in indices.values():
        assert len(series) == DAYS
        assert series.iloc[0] == pytest.approx(1.0)  # equiponderado y normalizado


def test_low_coverage_tickers_are_dropped_before_ranking(cfg):
    """Un valor sin fundamentales no puede desplazar el percentil de los demás."""
    fake = FakeProvider()
    empty = {"UP0", "UP1"}
    original = fake.statements
    fake.statements = lambda s: Statements() if s in empty else original(s)
    fake.estimates = lambda s: Estimates()

    run = run_region(cfg.region("us"), cfg, DataProvider(prices=fake, fundamentals=fake, universe=fake))

    assert set(run.dropped_low_coverage) >= empty
    assert not any(r.symbol in empty for r in run.results)
