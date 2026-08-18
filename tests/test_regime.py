from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.regime import breadth_score, compute_regime
from tests.conftest import make_prices


def series(closes):
    return pd.Series(closes, index=pd.bdate_range("2022-01-03", periods=len(closes)))


def bull_benchmark(days=400):
    return series(100.0 * np.cumprod(np.full(days, 1.0015)))


def bear_benchmark(days=400):
    return series(100.0 * np.cumprod(np.full(days, 0.9985)))


def universe(n_up: int, n_down: int, days: int = 300):
    prices = {}
    for i in range(n_up):
        prices[f"up{i}"] = make_prices(100.0 * np.cumprod(np.full(days, 1.002)))
    for i in range(n_down):
        prices[f"dn{i}"] = make_prices(100.0 * np.cumprod(np.full(days, 0.998)))
    return prices


def test_regime_stays_within_the_configured_bounds(cfg):
    low = float(cfg.regime["min_multiplier"])
    high = float(cfg.regime["max_multiplier"])

    for benchmark, prices in (
        (bull_benchmark(), universe(20, 0)),
        (bear_benchmark(), universe(0, 20)),
        (bull_benchmark(), universe(10, 10)),
        (None, {}),
    ):
        reading = compute_regime(benchmark, prices, cfg)
        assert low <= reading.multiplier <= high


def test_bull_scores_higher_than_bear(cfg):
    bull = compute_regime(bull_benchmark(), universe(20, 0), cfg)
    bear = compute_regime(bear_benchmark(), universe(0, 20), cfg)

    assert bull.multiplier == pytest.approx(float(cfg.regime["max_multiplier"]))
    assert bear.multiplier == pytest.approx(float(cfg.regime["min_multiplier"]))
    assert bull.multiplier > bear.multiplier


def test_regime_is_continuous_around_the_moving_average(cfg):
    """El punto del diseño: nada de escalones al cruzar la MM200.

    Un salto en seco haría que un valor alertase o no por un céntimo del índice.
    """
    band = float(cfg.regime["band"])
    prices = universe(10, 10)

    readings = []
    for distance in np.linspace(-band, band, 9):
        # benchmark plano y luego desplazado: la MM queda por debajo/encima
        flat = np.full(300, 100.0)
        flat[-1] = 100.0 * (1 + distance)
        readings.append(compute_regime(series(flat), prices, cfg).multiplier)

    deltas = np.diff(readings)
    assert (deltas >= -1e-9).all(), "el multiplicador debe ser monótono creciente"
    assert deltas.max() < 0.10, "ningún tramo puede dar un salto brusco"


def test_breadth_counts_names_above_their_moving_average(cfg):
    score, pct = breadth_score(universe(15, 5), ma_window=50, low=0.30, high=0.70)
    assert pct == pytest.approx(0.75)
    assert score == 1.0

    score, pct = breadth_score(universe(5, 15), ma_window=50, low=0.30, high=0.70)
    assert pct == pytest.approx(0.25)
    assert score == 0.0


def test_regime_is_neutral_when_the_benchmark_is_missing(cfg):
    """Sin benchmark no se puede inventar tendencia: 0.5 y aviso en el detalle."""
    reading = compute_regime(None, universe(10, 10), cfg)
    assert reading.trend_score == 0.5
    assert "sin benchmark" in reading.detail
