"""Backtest: retornos forward, FX histórico, fechas de rebalanceo y barrido."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtest import (
    BacktestResult,
    FxHistory,
    Snapshot,
    forward_return,
    rebalance_dates,
    summarize,
    sweep_thresholds,
)
from tests.conftest import make_prices


# ---------------------------------------------------------------------------
# Retornos forward
# ---------------------------------------------------------------------------
def test_forward_return_measures_the_window_after_the_date():
    prices = make_prices(np.arange(100.0, 200.0), start="2024-01-01")
    when = prices.index[10]

    # de 110 a 120 en 10 sesiones
    assert forward_return(prices, when, 10) == pytest.approx(120.0 / 110.0 - 1.0)


def test_forward_return_is_none_when_the_window_has_not_closed():
    """Contarla como 0 sesgaría a la baja justo las fechas más frescas."""
    prices = make_prices(np.arange(100.0, 150.0), start="2024-01-01")
    assert forward_return(prices, prices.index[-1], 21) is None
    assert forward_return(prices, prices.index[-5], 21) is None


def test_forward_return_uses_the_last_session_at_or_before_the_date():
    prices = make_prices(np.arange(100.0, 160.0), start="2024-01-01")
    # un sábado: debe usar el viernes anterior como entrada
    friday = prices.index[4]
    saturday = friday + pd.Timedelta(days=1)
    assert forward_return(prices, saturday, 5) == forward_return(prices, friday, 5)


def test_forward_return_handles_missing_prices():
    assert forward_return(None, pd.Timestamp("2024-01-01"), 21) is None
    assert forward_return(pd.DataFrame(), pd.Timestamp("2024-01-01"), 21) is None


# ---------------------------------------------------------------------------
# FX histórico
# ---------------------------------------------------------------------------
class FxPrices:
    def __init__(self, series):
        self.series = series

    def history(self, symbols, *, period):
        return {}

    def close_series(self, symbol, *, period):
        return self.series.get(symbol)


def fx_history():
    index = pd.bdate_range("2024-01-01", periods=100)
    return FxHistory(
        provider=type("P", (), {"prices": FxPrices({
            "EURUSD=X": pd.Series(np.linspace(1.00, 1.20, 100), index=index),
            "GBPUSD=X": pd.Series(np.linspace(1.20, 1.40, 100), index=index),
        })})(),
        period="1y",
    )


def test_fx_uses_the_rate_of_the_date_not_todays():
    """Convertir con el tipo actual falsea los gates de tamaño del pasado."""
    fx = fx_history()
    index = pd.bdate_range("2024-01-01", periods=100)

    early = fx.rate("EUR", index[0])
    late = fx.rate("EUR", index[-1])
    assert early == pytest.approx(1.00)
    assert late == pytest.approx(1.20)
    assert early < late


def test_fx_falls_back_to_the_previous_session():
    fx = fx_history()
    # un domingo: usa el viernes
    sunday = pd.Timestamp("2024-01-07")
    assert fx.rate("EUR", sunday) == fx.rate("EUR", pd.Timestamp("2024-01-05"))


def test_fx_before_the_series_starts_uses_the_first_point():
    fx = fx_history()
    assert fx.rate("EUR", pd.Timestamp("2020-01-01")) == pytest.approx(1.00)


def test_fx_usd_is_identity_and_needs_no_series():
    assert fx_history().rate("USD", pd.Timestamp("2024-06-01")) == 1.0


def test_fx_rates_splits_quoted_and_major_units():
    """GBp: la capitalización va en libras y el precio en peniques."""
    fx = fx_history()
    when = pd.bdate_range("2024-01-01", periods=100)[-1]
    quoted, major = fx.rates("GBp", when)
    assert major == pytest.approx(1.40)
    assert quoted == pytest.approx(0.014)

    quoted, major = fx.rates("EUR", when)
    assert quoted == major


def test_fx_unknown_currency_degrades_to_one():
    assert fx_history().rate("XYZ", pd.Timestamp("2024-06-01")) == 1.0


# ---------------------------------------------------------------------------
# Fechas de rebalanceo
# ---------------------------------------------------------------------------
def test_rebalance_dates_land_on_real_sessions():
    """Coger fin de mes de calendario caería en festivos sin cotización."""
    prices = {"A": make_prices(np.arange(100.0, 400.0), start="2024-01-01")}
    sessions = set(prices["A"].index)

    dates = rebalance_dates(prices, None, None, "monthly")
    assert len(dates) >= 12
    assert all(d in sessions for d in dates)
    assert dates == sorted(dates)


def test_rebalance_dates_respect_the_window():
    prices = {"A": make_prices(np.arange(100.0, 400.0), start="2024-01-01")}
    dates = rebalance_dates(prices, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-09-30"), "monthly")
    assert all(pd.Timestamp("2024-06-01") <= d <= pd.Timestamp("2024-09-30") for d in dates)
    assert len(dates) == 4


def test_weekly_gives_more_dates_than_monthly():
    prices = {"A": make_prices(np.arange(100.0, 400.0), start="2024-01-01")}
    assert len(rebalance_dates(prices, None, None, "weekly")) > len(
        rebalance_dates(prices, None, None, "monthly")
    )


def test_rebalance_dates_empty_without_prices():
    assert rebalance_dates({}, None, None, "monthly") == []


# ---------------------------------------------------------------------------
# Barrido de umbrales e informe
# ---------------------------------------------------------------------------
def sample_result():
    result = BacktestResult(region="us", horizons=[21], universe_size=500)
    snapshot = Snapshot(date=pd.Timestamp("2024-06-28"), regime=0.9, scored=100)
    snapshot.universe_forward = {21: 0.01}
    snapshot.alerts = [
        {"date": snapshot.date, "symbol": "A", "a_pct": 99.0, "b_pct": 99.0, "fwd_21": 0.20},
        {"date": snapshot.date, "symbol": "B", "a_pct": 85.0, "b_pct": 85.0, "fwd_21": 0.05},
        {"date": snapshot.date, "symbol": "C", "a_pct": 81.0, "b_pct": 81.0, "fwd_21": -0.03},
    ]
    result.snapshots = [snapshot]
    result.disabled_metrics = ["estimate_revisions"]
    return result


def test_sweep_tightens_the_cut_and_reduces_the_count():
    sweep = sweep_thresholds(sample_result(), [80.0, 90.0], horizon=21)

    loose = sweep[(sweep.a_threshold == 80.0) & (sweep.b_threshold == 80.0)].iloc[0]
    tight = sweep[(sweep.a_threshold == 90.0) & (sweep.b_threshold == 90.0)].iloc[0]

    assert loose["alertas"] == 3
    assert tight["alertas"] == 1
    assert tight["media_pct"] > loose["media_pct"]
    assert tight["aciertos_pct"] == 100.0


def test_sweep_is_empty_without_alerts():
    empty = BacktestResult(region="us", horizons=[21])
    assert sweep_thresholds(empty, [80.0], horizon=21).empty


def test_summary_states_the_survivorship_bias():
    """La advertencia no es opcional: sin ella el número engaña."""
    report = summarize(sample_result())
    assert "SESGO DE SUPERVIVENCIA NO CORREGIDO" in report
    assert "estimate_revisions" in report
    # el listón es el universo, no el 0%
    assert "universo" in report


def test_summary_reports_excess_over_the_universe():
    report = summarize(sample_result())
    # media de alertas 7.33%, universo 1% -> exceso 6.33
    assert "7.33" in report
    assert "6.33" in report


def test_summary_handles_no_alerts():
    empty = BacktestResult(region="us", horizons=[21], universe_size=10)
    assert "nada que medir" in summarize(empty)
