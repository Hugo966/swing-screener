"""Fixtures sintéticas: ningún test toca la red."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.config import load_config
from screener.models import TickerData


@pytest.fixture(scope="session")
def cfg():
    return load_config(load_env=False)


def make_prices(
    closes: list[float] | np.ndarray,
    *,
    volumes: list[float] | None = None,
    start: str = "2023-01-02",
) -> pd.DataFrame:
    """OHLCV diario a partir de una serie de cierres."""
    closes = np.asarray(closes, dtype=float)
    index = pd.bdate_range(start=start, periods=len(closes))
    volume = np.asarray(volumes, dtype=float) if volumes is not None else np.full(len(closes), 1e6)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": volume,
        },
        index=index,
    )


def trending_prices(days: int = 400, daily: float = 0.001, start_price: float = 100.0) -> pd.DataFrame:
    """Tendencia alcista perfectamente suave: sirve de caso de referencia."""
    closes = start_price * np.cumprod(np.full(days, 1.0 + daily))
    return make_prices(closes)


def statement(rows: dict[str, list[float]], periods: list[str]) -> pd.DataFrame:
    """Estado financiero al estilo yfinance: filas = concepto, columnas = periodo."""
    return pd.DataFrame(rows, index=pd.to_datetime(periods)).T


@pytest.fixture
def prices():
    return trending_prices()


@pytest.fixture
def ticker(prices):
    return TickerData(symbol="TEST", asof=date(2024, 7, 15), sector="Technology", prices=prices)
