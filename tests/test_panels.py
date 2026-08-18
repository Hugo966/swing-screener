"""Cobertura de datos, imputación neutra y desglose."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.models import TickerData
from screener.panels import (
    build_breakdowns,
    drop_uncovered_metrics,
    finalize_panel,
    prepare_panel,
    to_percentiles,
)
from tests.conftest import statement, trending_prices

PERIODS = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]


def make_ticker(symbol, revenues=None, sector="Technology"):
    return TickerData(
        symbol=symbol,
        asof=date(2025, 2, 1),
        sector=sector,
        prices=trending_prices(days=400),
        income_a=statement({"Total Revenue": revenues}, PERIODS) if revenues else None,
    )


def test_metric_with_no_regional_coverage_is_dropped_and_weights_rescale():
    """Si en una región no hay datos de una métrica, se cae sola y se renormaliza.

    Es la misma degradación que la spec prevé para la Fase 2, pero automática.
    """
    raw = pd.DataFrame(
        {"tiene_datos": [1.0, 2.0, 3.0, 4.0], "sin_datos": [None, None, None, None]},
        index=["a", "b", "c", "d"],
    )
    weights = {"tiene_datos": 60.0, "sin_datos": 40.0}

    effective, coverage = drop_uncovered_metrics(raw, weights, min_coverage=0.5)

    assert coverage["sin_datos"] == 0.0
    assert "sin_datos" not in effective
    assert effective["tiene_datos"] == pytest.approx(100.0)


def test_partially_covered_metric_survives():
    raw = pd.DataFrame({"parcial": [1.0, 2.0, None, None]}, index=["a", "b", "c", "d"])
    effective, coverage = drop_uncovered_metrics(raw, {"parcial": 100.0}, min_coverage=0.5)
    assert coverage["parcial"] == pytest.approx(0.5)
    assert "parcial" in effective


def test_missing_values_are_imputed_to_the_neutral_percentile(cfg):
    raw = pd.DataFrame(
        {"revenue_growth_level": [0.1, 0.5, None, 0.9]}, index=["a", "b", "c", "d"]
    )
    weights = {"revenue_growth_level": 100.0}
    sectors = pd.Series({s: "Technology" for s in raw.index})

    percentiles, imputed = to_percentiles(
        raw, weights, sectors, against="universe", min_sector_size=8
    )

    assert imputed.at["c", "revenue_growth_level"]
    assert percentiles.at["c", "revenue_growth_level"] == 50.0
    assert not imputed.at["a", "revenue_growth_level"]


def test_breakdown_reports_contribution_and_coverage():
    raw = pd.DataFrame({"m1": [1.0, 2.0], "m2": [None, 4.0]}, index=["a", "b"])
    percentiles = pd.DataFrame({"m1": [50.0, 100.0], "m2": [50.0, 100.0]}, index=["a", "b"])
    imputed = pd.DataFrame({"m1": [False, False], "m2": [True, False]}, index=["a", "b"])

    breakdowns = build_breakdowns("quality", raw, percentiles, imputed, {"m1": 70.0, "m2": 30.0})

    # 'a' tiene m2 imputada: solo el 70% del peso tiene dato real
    assert breakdowns["a"].coverage == pytest.approx(0.7)
    assert breakdowns["b"].coverage == pytest.approx(1.0)
    # panel_raw = suma ponderada de percentiles
    assert breakdowns["a"].raw_score == pytest.approx(50.0)
    assert breakdowns["b"].raw_score == pytest.approx(100.0)

    metric = next(m for m in breakdowns["a"].metrics if m.name == "m2")
    assert metric.imputed and metric.raw is None
    assert metric.contribution == pytest.approx(50.0 * 30.0 / 100.0)


def test_ticker_coverage_weights_by_metric_weight(cfg):
    """La cobertura por ticker pondera: faltar B4 (16 pts) pesa más que faltar B10 (3)."""
    tickers = {
        "completo": make_ticker("completo", [100, 130, 170, 220]),
        "vacio": make_ticker("vacio", None),
    }
    inputs = prepare_panel("quality", tickers, cfg.region("us").weights["quality"], cfg)
    coverage = inputs.ticker_coverage()

    assert coverage["completo"] > coverage["vacio"]
    assert coverage["vacio"] == pytest.approx(0.0)


def test_finalize_panel_produces_one_breakdown_per_ticker(cfg):
    tickers = {f"t{i}": make_ticker(f"t{i}", [100, 110 + i * 10, 130, 150 + i * 20]) for i in range(5)}
    weights = {"revenue_growth_level": 100.0}
    inputs = prepare_panel("quality", tickers, weights, cfg)
    sectors = pd.Series({s: "Technology" for s in tickers})

    breakdowns = finalize_panel(inputs, sectors, cfg)

    assert set(breakdowns) == set(tickers)
    scores = [b.raw_score for b in breakdowns.values()]
    assert min(scores) >= 0.0 and max(scores) <= 100.0
