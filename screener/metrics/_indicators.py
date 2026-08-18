"""Indicadores de precio compartidos por varias métricas.

El ATR% de aquí es el mismo que luego sirve para stop y sizing (§6, A6 de la
spec): un solo sitio donde se define.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def last(series: pd.Series | None) -> float | None:
    if series is None or len(series) == 0:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def total_return(series: pd.Series, window: int, *, offset: int = 0) -> float | None:
    """Retorno entre t-offset-window y t-offset. `offset` implementa el "12-1"."""
    end = len(series) - 1 - offset
    start = end - window
    if start < 0 or end < 0:
        return None
    a, b = series.iloc[start], series.iloc[end]
    if pd.isna(a) or pd.isna(b) or a <= 0:
        return None
    return float(b / a - 1.0)


def atr(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    """True range medio (Wilder, vía media móvil simple)."""
    high, low, close = prices["High"], prices["Low"], prices["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def atr_pct(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR normalizado por precio: comparable entre valores de distinto nominal."""
    return atr(prices, period) / prices["Close"]


def obv(prices: pd.DataFrame) -> pd.Series:
    """On-balance volume."""
    direction = np.sign(prices["Close"].diff().fillna(0.0))
    return (direction * prices["Volume"]).cumsum()


def slope(values: pd.Series | np.ndarray) -> float | None:
    """Pendiente de la regresión lineal simple sobre el índice 0..n-1."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return None
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def daily_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()
