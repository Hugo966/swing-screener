"""Point-in-time: lo que no se sabía en la fecha no puede entrar al scoring.

Un fallo aquí no rompe nada visiblemente — solo hace que el backtest salga
glorioso y en real no funcione. De ahí el detalle de estos tests.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.models import TickerData
from screener.pointintime import (
    as_of,
    publication_date,
    report_dates,
    truncate_statement,
)
from tests.conftest import make_prices, statement

PERIODS = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30"]
REPORTS = pd.to_datetime(["2024-01-25", "2024-04-25", "2024-07-25", "2024-10-24"])


def earnings(index=REPORTS) -> pd.DataFrame:
    return pd.DataFrame(
        {"Reported EPS": [1.0] * len(index), "Surprise(%)": [5.0] * len(index)}, index=index
    )


def full_ticker() -> TickerData:
    return TickerData(
        symbol="TEST",
        asof=date(2025, 1, 15),
        sector="Technology",
        prices=make_prices(np.linspace(100.0, 200.0, 520), start="2023-01-02"),
        income_q=statement({"Total Revenue": [100, 110, 120, 130]}, PERIODS),
        income_a=statement({"Total Revenue": [400, 460]}, ["2023-12-31", "2024-12-31"]),
        cashflow_q=statement({"Free Cash Flow": [20, 22, 24, 26]}, PERIODS),
        balance_q=statement({"Net Debt": [50] * 4, "Total Debt": [80] * 4}, PERIODS),
        earnings_dates=earnings(),
        eps_revisions=pd.DataFrame({"upLast30days": [5], "downLast30days": [1]}, index=["+1y"]),
        eps_trend=pd.DataFrame({"current": [11.0], "90daysAgo": [10.0]}, index=["+1y"]),
        shares_full=pd.Series(
            [100e6, 101e6, 102e6],
            index=pd.to_datetime(["2023-06-01", "2024-06-01", "2024-12-01"]),
        ),
        profile={"sector": "Technology", "marketCap": 999e9, "enterpriseValue": 999e9},
    )


# ---------------------------------------------------------------------------
# Fecha de publicación
# ---------------------------------------------------------------------------
def test_publication_is_the_first_report_after_the_period_ends():
    """El Q3 que cierra el 30/09 no está disponible hasta que se publica en octubre."""
    reports = report_dates(earnings())
    assert publication_date(pd.Timestamp("2024-09-30"), reports, 60) == pd.Timestamp("2024-10-24")
    assert publication_date(pd.Timestamp("2024-06-30"), reports, 60) == pd.Timestamp("2024-07-25")


def test_publication_falls_back_to_a_fixed_lag_without_a_calendar():
    empty = report_dates(None)
    assert publication_date(pd.Timestamp("2024-09-30"), empty, 60) == pd.Timestamp("2024-11-29")


def test_publication_falls_back_when_the_calendar_stops_short():
    reports = report_dates(earnings(pd.to_datetime(["2024-01-25"])))
    # no hay report posterior al Q3: se usa el retraso fijo
    assert publication_date(pd.Timestamp("2024-09-30"), reports, 45) == pd.Timestamp("2024-11-14")


# ---------------------------------------------------------------------------
# Truncado de estados
# ---------------------------------------------------------------------------
def test_statement_keeps_only_published_periods():
    """Este es EL test del backtest: usar el Q3 desde el 30/09 lo invalida."""
    frame = statement({"Total Revenue": [100, 110, 120, 130]}, PERIODS)
    reports = report_dates(earnings())

    # el 1 de octubre el Q3 ya cerró pero NO se ha publicado
    truncated = truncate_statement(frame, pd.Timestamp("2024-10-01"), reports, 60)
    assert [str(c.date()) for c in truncated.columns] == ["2023-12-31", "2024-03-31", "2024-06-30"]

    # el 25 de octubre, un día después del report, ya sí
    truncated = truncate_statement(frame, pd.Timestamp("2024-10-25"), reports, 60)
    assert "2024-09-30" in [str(c.date()) for c in truncated.columns]


def test_statement_is_none_when_nothing_was_published_yet():
    frame = statement({"Total Revenue": [100]}, ["2024-09-30"])
    assert truncate_statement(frame, pd.Timestamp("2024-02-01"), report_dates(earnings()), 60) is None


# ---------------------------------------------------------------------------
# as_of
# ---------------------------------------------------------------------------
def test_as_of_truncates_prices_to_the_date():
    past = as_of(full_ticker(), "2024-06-28")
    assert past.prices.index.max() <= pd.Timestamp("2024-06-28")
    assert len(past.prices) < 520


def test_as_of_drops_estimate_vintages():
    """`eps_revisions` es una foto de hoy: en una fecha pasada es look-ahead puro."""
    past = as_of(full_ticker(), "2024-06-28")
    assert past.eps_revisions is None
    assert past.eps_trend is None


def test_as_of_truncates_the_earnings_calendar():
    past = as_of(full_ticker(), "2024-06-28")
    assert past.earnings_dates.index.max() == pd.Timestamp("2024-04-25")
    assert len(past.earnings_dates) == 2


def test_as_of_truncates_the_share_count():
    """B7 mide dilución: el recuento de acciones futuro no puede estar."""
    past = as_of(full_ticker(), "2024-08-01")
    assert past.shares_full.index.max() == pd.Timestamp("2024-06-01")


def test_as_of_rebuilds_market_cap_instead_of_reusing_todays():
    """El marketCap de `info` es el de hoy: usarlo en una fecha pasada es look-ahead."""
    data = full_ticker()
    past = as_of(data, "2024-08-01")

    assert past.profile["marketCap"] != data.profile["marketCap"]
    expected = float(past.prices["Close"].iloc[-1]) * 101e6
    assert past.profile["marketCap"] == pytest.approx(expected)
    # EV = capitalización + deuda neta conocida entonces
    assert past.profile["enterpriseValue"] == pytest.approx(expected + 50.0)


def test_as_of_is_monotonic_in_information():
    """Cuanto más tarde la fecha, nunca menos información."""
    data = full_ticker()
    sizes = []
    for when in ("2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"):
        past = as_of(data, when)
        sizes.append((
            len(past.prices),
            0 if past.income_q is None else past.income_q.shape[1],
            0 if past.earnings_dates is None else len(past.earnings_dates),
        ))
    for earlier, later in zip(sizes, sizes[1:]):
        assert all(a <= b for a, b in zip(earlier, later)), (earlier, later)


def test_as_of_preserves_the_sector_index_and_benchmark_cut():
    data = full_ticker()
    index = pd.DatetimeIndex(data.prices.index)
    data.benchmark = pd.Series(np.linspace(1.0, 2.0, len(index)), index=index)
    data.sector_index = pd.Series(np.linspace(1.0, 1.5, len(index)), index=index)

    past = as_of(data, "2024-06-28")
    assert past.benchmark.index.max() <= pd.Timestamp("2024-06-28")
    assert past.sector_index.index.max() <= pd.Timestamp("2024-06-28")


def test_metrics_computed_as_of_a_past_date_differ_from_today(cfg):
    """Comprobación de conjunto: el scoring cambia con la fecha, como debe."""
    from screener.metrics import registry

    data = full_ticker()
    today = registry.compute("revenue_growth_level", data, cfg.metric_params("revenue_growth_level"))
    # en febrero de 2024 solo estaba publicado el Q4-2023: no hay interanual
    early = as_of(data, "2024-02-01")
    past = registry.compute("revenue_growth_level", early, cfg.metric_params("revenue_growth_level"))
    assert today is not None
    assert past is None or past != today
