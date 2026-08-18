"""Régimen de mercado por región (§5).

Multiplicador **continuo** en [min, max], no un escalón: un halving en seco al
cruzar la MM200 crea un acantilado en el que un valor pasa de alertar a no
alertar por un céntimo del benchmark.

`score_final = A_pct * B_pct/100 * regime`. En bear real nada llega al corte; en
transición el corte se endurece progresivamente. Este es el mecanismo que
protege de los momentum crashes (2009, 03/2020).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from screener.metrics._indicators import clip, sma

log = logging.getLogger(__name__)


@dataclass
class RegimeReading:
    multiplier: float
    trend_score: float
    breadth_score: float
    benchmark_vs_ma: float | None
    breadth_pct: float | None
    detail: str = ""


def benchmark_trend_score(benchmark: pd.Series | None, ma_window: int, band: float) -> tuple[float, float | None]:
    """Distancia del benchmark a su MM, mapeada a [0,1] por una rampa de ±band."""
    if benchmark is None or len(benchmark) < ma_window:
        return 0.5, None
    ma = sma(benchmark.dropna(), ma_window)
    if ma.empty or pd.isna(ma.iloc[-1]) or ma.iloc[-1] <= 0:
        return 0.5, None
    distance = float(benchmark.iloc[-1] / ma.iloc[-1] - 1.0)
    return clip((distance + band) / (2.0 * band), 0.0, 1.0), distance


def breadth_score(
    prices: dict[str, pd.DataFrame], ma_window: int, low: float, high: float
) -> tuple[float, float | None]:
    """% de valores de la región sobre su MM50, mapeado a [0,1]."""
    above = total = 0
    for frame in prices.values():
        if frame is None or frame.empty:
            continue
        close = frame["Close"].dropna()
        if len(close) < ma_window:
            continue
        ma = sma(close, ma_window).iloc[-1]
        if pd.isna(ma) or ma <= 0:
            continue
        total += 1
        above += int(float(close.iloc[-1]) > float(ma))

    if total == 0:
        return 0.5, None
    pct = above / total
    if high <= low:
        return 0.5, pct
    return clip((pct - low) / (high - low), 0.0, 1.0), pct


def compute_regime(
    benchmark: pd.Series | None, prices: dict[str, pd.DataFrame], cfg
) -> RegimeReading:
    params = cfg.regime
    trend, distance = benchmark_trend_score(
        benchmark, int(params["benchmark_ma"]), float(params["band"])
    )
    breadth, pct_above = breadth_score(
        prices, int(params["breadth_ma"]), float(params["breadth_lo"]), float(params["breadth_hi"])
    )

    weight_trend = float(params["weight_trend"])
    weight_breadth = float(params["weight_breadth"])
    total_weight = weight_trend + weight_breadth
    blended = (weight_trend * trend + weight_breadth * breadth) / total_weight

    low, high = float(params["min_multiplier"]), float(params["max_multiplier"])
    multiplier = low + (high - low) * blended

    detail = (
        f"benchmark {'+' if (distance or 0) >= 0 else ''}{(distance or 0) * 100:.1f}% vs MM"
        f"{params['benchmark_ma']}, amplitud {(pct_above or 0) * 100:.0f}% sobre MM{params['breadth_ma']}"
    )
    if distance is None:
        detail = "sin benchmark utilizable; tendencia neutra. " + detail
        log.warning("régimen: no se pudo leer el benchmark, se usa 0.5 de tendencia")

    return RegimeReading(
        multiplier=multiplier,
        trend_score=trend,
        breadth_score=breadth,
        benchmark_vs_ma=distance,
        breadth_pct=pct_above,
        detail=detail,
    )
