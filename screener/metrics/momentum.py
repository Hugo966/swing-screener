"""Panel A — Momentum (A1-A10).

Cada función es pura: `(TickerData, params) -> float | None`. Devuelve su
magnitud natural; el sentido (mayor/menor es mejor) se declara en el registry y
la conversión a percentil ocurre en `normalize.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.metrics._indicators import (
    atr_pct,
    clip,
    daily_returns,
    obv,
    sma,
    total_return,
)
from screener.metrics.registry import metric
from screener.models import TickerData


def _close(data: TickerData) -> pd.Series | None:
    if data.prices is None or data.prices.empty or "Close" not in data.prices:
        return None
    series = data.prices["Close"].dropna()
    return series if len(series) else None


# ---------------------------------------------------------------------------
# A1 — Fuerza relativa multi-ventana
# ---------------------------------------------------------------------------
@metric("rs_multi_window", panel="momentum", label="A1 RS multi-ventana")
def rs_multi_window(data: TickerData, p: dict) -> float | None:
    """Retornos 3m/6m/12m ponderados, todos hasta hace un mes, penalizando el +1m excesivo.

    Las **tres** ventanas se desplazan `skip_recent_days` hacia atrás, no solo la
    de 12 meses. Si el último mes contara dentro de la de 3m y 6m, un vertical
    reciente subiría el score por esa vía más de lo que lo baja la penalización
    de reversión, que es exactamente lo que la métrica quiere evitar. El último
    mes ya lo recogen A2 y A9.
    """
    close = _close(data)
    if close is None:
        return None

    windows, weights = p["windows"], p["weights"]
    skip = int(p["skip_recent_days"])

    r3 = total_return(close, int(windows["m3"]), offset=skip)
    r6 = total_return(close, int(windows["m6"]), offset=skip)
    r12_1 = total_return(close, int(windows["m12"]), offset=skip)
    if r3 is None or r6 is None or r12_1 is None:
        return None

    base = weights["m3"] * r3 + weights["m6"] * r6 + weights["m12"] * r12_1

    r1 = total_return(close, skip)
    if r1 is not None and r1 > p["reversal_threshold"]:
        base -= p["reversal_penalty"] * (r1 - p["reversal_threshold"])
    return base


# ---------------------------------------------------------------------------
# A2 — Distancia a máximos / ruptura de base
# ---------------------------------------------------------------------------
@metric("distance_to_high", panel="momentum", label="A2 distancia a máximos")
def distance_to_high(data: TickerData, p: dict) -> float | None:
    """% bajo el máximo de 52 semanas, con bonus si rompe la base con volumen."""
    close = _close(data)
    if close is None:
        return None
    lookback = int(p["lookback"])
    if len(close) < min(lookback, 60):
        return None

    window = close.iloc[-lookback:]
    high = float(window.max())
    if high <= 0:
        return None
    score = float(close.iloc[-1]) / high - 1.0  # <= 0; cuanto más cerca de 0, mejor

    breakout_window = int(p["breakout_window"])
    volume = data.prices["Volume"] if "Volume" in data.prices else None
    if volume is not None and len(close) > breakout_window:
        recent_high = float(close.iloc[-breakout_window:].max())
        prior_high = float(window.iloc[:-breakout_window].max()) if len(window) > breakout_window else high
        volume_ma = sma(volume, 50)
        recent_volume = float(volume.iloc[-breakout_window:].mean())
        baseline = volume_ma.iloc[-1]
        broke_out = recent_high >= prior_high > 0
        heavy = bool(
            not pd.isna(baseline) and baseline > 0
            and recent_volume >= p["breakout_volume_mult"] * float(baseline)
        )
        if broke_out and heavy:
            score += float(p["breakout_bonus"])
    return score


# ---------------------------------------------------------------------------
# A3 — Fuerza relativa vs su sector
# ---------------------------------------------------------------------------
@metric("rs_vs_sector", panel="momentum", label="A3 RS vs sector")
def rs_vs_sector(data: TickerData, p: dict) -> float | None:
    """Subir 20% en un sector que sube 25% es debilidad disfrazada."""
    close = _close(data)
    if close is None or data.sector_index is None or len(data.sector_index) == 0:
        return None
    window = int(p["window"])
    stock = total_return(close, window)
    sector = total_return(data.sector_index.dropna(), window)
    if stock is None or sector is None:
        return None
    return stock - sector


# ---------------------------------------------------------------------------
# A4 — Momentum del sector
# ---------------------------------------------------------------------------
@metric("sector_momentum", panel="momentum", label="A4 momentum del sector")
def sector_momentum(data: TickerData, p: dict) -> float | None:
    """El sector del valor, en tendencia o no."""
    index = data.sector_index
    if index is None:
        return None
    index = index.dropna()
    if len(index) == 0:
        return None

    ret = total_return(index, int(p["window"]))
    if ret is None:
        return None

    above_ma = 0.0
    ma = sma(index, int(p["ma"]))
    if not ma.empty and not pd.isna(ma.iloc[-1]) and ma.iloc[-1] > 0:
        above_ma = float(index.iloc[-1] / ma.iloc[-1] - 1.0)
    return 0.5 * ret + 0.5 * above_ma


# ---------------------------------------------------------------------------
# A5 — Consistencia (frog-in-the-pan)
# ---------------------------------------------------------------------------
@metric("consistency_fitp", panel="momentum", label="A5 consistencia (FITP)")
def consistency_fitp(data: TickerData, p: dict) -> float | None:
    """El momentum a goteo persiste; el de gaps, no.

    Information discreteness clásica: sign(retorno) * (%días negativos - %positivos).
    Un ID bajo = información continua = mejor, así que se devuelve con el signo
    cambiado. Se descuenta además la parte del movimiento que vino en saltos.
    """
    close = _close(data)
    if close is None:
        return None
    window = int(p["window"])
    if len(close) < window:
        return None

    segment = close.iloc[-window:]
    rets = daily_returns(segment)
    if len(rets) < 20:
        return None

    total = float(segment.iloc[-1] / segment.iloc[0] - 1.0)
    frac_pos = float((rets > 0).mean())
    frac_neg = float((rets < 0).mean())
    fip = np.sign(total) * (frac_neg - frac_pos)

    gap_threshold = float(p["gap_threshold"])
    abs_total = rets.abs().sum()
    gap_share = float(rets[rets.abs() > gap_threshold].abs().sum() / abs_total) if abs_total > 0 else 0.0

    return float(-fip - gap_share)


# ---------------------------------------------------------------------------
# A6 — Momentum ajustado por volatilidad
# ---------------------------------------------------------------------------
@metric("vol_adjusted_momentum", panel="momentum", label="A6 momentum / ATR%")
def vol_adjusted_momentum(data: TickerData, p: dict) -> float | None:
    """Retorno por unidad de ATR%: evita premiar small caps volátiles."""
    close = _close(data)
    if close is None or data.prices is None:
        return None
    window = int(p["window"])
    ret = total_return(close, window)
    if ret is None:
        return None

    atr_series = atr_pct(data.prices, int(p["atr_period"])).dropna()
    if len(atr_series) == 0:
        return None
    mean_atr = float(atr_series.iloc[-window:].mean())
    if mean_atr <= 0 or pd.isna(mean_atr):
        return None
    return ret / mean_atr


# ---------------------------------------------------------------------------
# A7 — Acumulación (OBV + volumen de ruptura)
# ---------------------------------------------------------------------------
@metric("accumulation_obv", panel="momentum", label="A7 acumulación (OBV)")
def accumulation_obv(data: TickerData, p: dict) -> float | None:
    """Demanda por volumen: cuánto del volumen del periodo fue comprador."""
    if data.prices is None or "Volume" not in data.prices:
        return None
    window = int(p["window"])
    prices = data.prices.dropna(subset=["Close", "Volume"])
    if len(prices) < window + 1:
        return None

    obv_series = obv(prices)
    net = float(obv_series.iloc[-1] - obv_series.iloc[-window])
    traded = float(prices["Volume"].iloc[-window:].sum())
    if traded <= 0:
        return None
    net_share = net / traded  # en [-1, 1]

    surge = 0.0
    volume_ma = sma(prices["Volume"], int(p["volume_ma"]))
    baseline = volume_ma.iloc[-1] if len(volume_ma) else np.nan
    if not pd.isna(baseline) and baseline > 0:
        surge = float(prices["Volume"].iloc[-5:].mean() / baseline - 1.0)

    return 0.7 * net_share + 0.3 * clip(surge, -1.0, 2.0)


# ---------------------------------------------------------------------------
# A8 — Reacción a los últimos resultados (PEAD)
# ---------------------------------------------------------------------------
@metric("pead_reaction", panel="momentum", label="A8 reacción a resultados")
def pead_reaction(data: TickerData, p: dict) -> float | None:
    """Saltó al publicar y mantuvo el salto -> continuación fiable."""
    close = _close(data)
    if close is None or data.earnings_dates is None or data.earnings_dates.empty:
        return None

    reported = data.earnings_dates
    if "Reported EPS" in reported.columns:
        reported = reported[reported["Reported EPS"].notna()]
    if reported.empty:
        return None

    event = pd.Timestamp(reported.index.max()).tz_localize(None).normalize()
    index = close.index.tz_localize(None) if close.index.tz is not None else close.index
    age = (index[-1] - event).days
    if age < 0 or age > int(p["max_age_days"]):
        return None

    positions = np.flatnonzero(index >= event)
    if len(positions) == 0:
        return None
    start = int(positions[0])
    if start == 0:
        return None

    before = float(close.iloc[start - 1])
    reaction_end = min(start + int(p["reaction_days"]), len(close) - 1)
    if before <= 0 or reaction_end <= start - 1:
        return None
    jump = float(close.iloc[reaction_end]) / before - 1.0

    hold_end = min(start + int(p["hold_days"]), len(close) - 1)
    hold = 0.0
    if hold_end > reaction_end:
        anchor = float(close.iloc[reaction_end])
        if anchor > 0:
            hold = float(close.iloc[hold_end]) / anchor - 1.0

    return jump + 0.5 * hold


# ---------------------------------------------------------------------------
# A9 — Precio sobre medias móviles
# ---------------------------------------------------------------------------
@metric("price_vs_ma", panel="momentum", label="A9 precio vs MM20/50")
def price_vs_ma(data: TickerData, p: dict) -> float | None:
    """Redundante con A1/A2 a propósito: peso testimonial."""
    close = _close(data)
    if close is None:
        return None
    parts = []
    for key in ("ma_fast", "ma_slow"):
        ma = sma(close, int(p[key]))
        if len(ma) == 0 or pd.isna(ma.iloc[-1]) or ma.iloc[-1] <= 0:
            continue
        parts.append(float(close.iloc[-1] / ma.iloc[-1] - 1.0))
    if not parts:
        return None
    return sum(parts) / len(parts)


# ---------------------------------------------------------------------------
# A10 — Proximidad de evento
# ---------------------------------------------------------------------------
@metric("event_proximity", panel="momentum", label="A10 proximidad de resultados")
def event_proximity(data: TickerData, p: dict) -> float | None:
    """1.0 sin evento a la vista; baja al acercarse resultados. Penaliza, no veta."""
    warn_days = int(p["warn_days"])
    min_score = float(p["min_score"])

    if data.earnings_dates is None or data.earnings_dates.empty:
        return 1.0

    index = pd.DatetimeIndex(data.earnings_dates.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    today = pd.Timestamp(data.asof)
    future = index[index >= today]
    if len(future) == 0:
        return 1.0

    days = int((future.min() - today).days)
    if days >= warn_days:
        return 1.0
    return min_score + (1.0 - min_score) * (days / warn_days)
