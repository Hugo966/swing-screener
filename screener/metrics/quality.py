"""Panel B — Calidad (B1-B10).

Funciones puras `(TickerData, params) -> float | None`. Las que miden algo malo
(dilución, deuda, valoración) devuelven la magnitud del mal y se registran con
`higher_is_better=False`: invertir es tarea de `normalize.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.metrics import _financials as fin
from screener.metrics._indicators import clip, slope
from screener.metrics.registry import metric
from screener.models import TickerData


def _revenue(data: TickerData) -> tuple[pd.Series | None, pd.Series | None]:
    return fin.line(data.income_q, fin.REVENUE), fin.line(data.income_a, fin.REVENUE)


# ---------------------------------------------------------------------------
# B1 — Crecimiento de revenue Y/Y (nivel)
# ---------------------------------------------------------------------------
@metric("revenue_growth_level", panel="quality", label="B1 crecimiento revenue Y/Y")
def revenue_growth_level(data: TickerData, p: dict) -> float | None:
    """Crecimiento interanual más reciente disponible (trimestral > anual)."""
    quarterly, annual = _revenue(data)
    return fin.growth_latest(quarterly, annual)


# ---------------------------------------------------------------------------
# B2 — Tendencia + duración del crecimiento
# ---------------------------------------------------------------------------
@metric("growth_trend_duration", panel="quality", label="B2 tendencia y duración")
def growth_trend_duration(data: TickerData, p: dict) -> float | None:
    """Persistencia del crecimiento por encima de la pendiente reciente.

    La regla tolerante ("2 de los últimos 3 creciendo") evita vetar una empresa
    por un trimestre flojo aislado.
    """
    quarterly, annual = _revenue(data)
    history = fin.growth_history(quarterly, annual)
    if history is None or len(history) < 2:
        return None

    strong = float(p["strong_growth"])
    persistence = float((history > strong).mean())

    k, n = (int(x) for x in p["tolerant_rule"])
    if len(history) >= n:
        recent_hits = int((history.iloc[-n:] > strong).sum())
        if recent_hits >= k:
            persistence = max(persistence, k / n)

    trend = slope(history.to_numpy())
    trend_component = clip(trend, -0.5, 0.5) if trend is not None else 0.0

    w = float(p["slope_weight"])
    return w * trend_component + (1.0 - w) * persistence


# ---------------------------------------------------------------------------
# B3 — Sorpresa media 4T + reacción
# ---------------------------------------------------------------------------
@metric("earnings_surprise_4q", panel="quality", label="B3 sorpresa 4T")
def earnings_surprise_4q(data: TickerData, p: dict) -> float | None:
    """Media de las últimas sorpresas, no solo la última, más la reacción del precio."""
    table = data.earnings_dates
    if table is None or table.empty or "Surprise(%)" not in table.columns:
        return None

    reported = table[table["Surprise(%)"].notna()].sort_index()
    if reported.empty:
        return None

    quarters = int(p["quarters"])
    reported = reported.iloc[-quarters:]
    mean_surprise = float(reported["Surprise(%)"].mean()) / 100.0

    reaction_weight = float(p["reaction_weight"])
    mean_reaction = _mean_earnings_reaction(data, reported.index, int(p["reaction_days"]))
    if mean_reaction is None:
        return mean_surprise
    return (1.0 - reaction_weight) * mean_surprise + reaction_weight * mean_reaction


def _mean_earnings_reaction(
    data: TickerData, dates: pd.Index, reaction_days: int
) -> float | None:
    if data.prices is None or data.prices.empty:
        return None
    close = data.prices["Close"].dropna()
    if len(close) < 2:
        return None
    index = close.index.tz_localize(None) if close.index.tz is not None else close.index

    reactions: list[float] = []
    for raw_date in dates:
        event = pd.Timestamp(raw_date)
        if event.tz is not None:
            event = event.tz_localize(None)
        event = event.normalize()
        positions = np.flatnonzero(index >= event)
        if len(positions) == 0:
            continue
        start = int(positions[0])
        end = min(start + reaction_days, len(close) - 1)
        if start == 0 or end <= start - 1:
            continue
        before = float(close.iloc[start - 1])
        if before <= 0:
            continue
        reactions.append(float(close.iloc[end]) / before - 1.0)

    return float(np.mean(reactions)) if reactions else None


# ---------------------------------------------------------------------------
# B4 — Calidad de caja (FCF/NI + FCF creciente)
# ---------------------------------------------------------------------------
@metric("cash_quality_fcf_ni", panel="quality", label="B4 calidad de caja")
def cash_quality_fcf_ni(data: TickerData, p: dict) -> float | None:
    """Anomalía de devengos (Sloan): beneficio sin caja detrás rinde peor."""
    fcf_q = fin.free_cash_flow(data.cashflow_q)
    fcf_a = fin.free_cash_flow(data.cashflow_a)
    ni_q = fin.line(data.income_q, fin.NET_INCOME)
    ni_a = fin.line(data.income_a, fin.NET_INCOME)

    fcf = fin.ttm_or_annual(fcf_q, fcf_a)
    net_income = fin.ttm_or_annual(ni_q, ni_a)
    if fcf is None or net_income is None or net_income <= 0:
        # Sin beneficio positivo la conversión no es interpretable.
        return None

    conversion = fcf / net_income
    red_flag = float(p["red_flag_ratio"])
    component = min(conversion, 3.0)
    if conversion < red_flag:
        component -= red_flag - conversion  # doble castigo bajo la bandera roja

    trend_weight = float(p["trend_weight"])
    fcf_growth = fin.growth(fcf_a, 1)
    if fcf_growth is None:
        return component
    return (1.0 - trend_weight) * component + trend_weight * clip(fcf_growth, -1.0, 2.0)


# ---------------------------------------------------------------------------
# B5 — ROIC
# ---------------------------------------------------------------------------
@metric("roic_vs_sector", panel="quality", label="B5 ROIC")
def roic_vs_sector(data: TickerData, p: dict) -> float | None:
    """NOPAT / capital invertido. Sustituye al ROE, que se infla apalancando."""
    ebit = fin.ttm_or_annual(
        fin.line(data.income_q, fin.EBIT), fin.line(data.income_a, fin.EBIT)
    )
    capital_series = fin.first(
        fin.invested_capital(data.balance_q), fin.invested_capital(data.balance_a)
    )
    if ebit is None or capital_series is None or capital_series.empty:
        return None

    # Capital medio de los dos últimos cierres si existen: menos ruido de un corte.
    capital = float(capital_series.iloc[-2:].mean()) if len(capital_series) >= 2 else float(capital_series.iloc[-1])
    if capital <= 0:
        return None

    tax_series = fin.first(
        fin.line(data.income_a, fin.TAX_RATE), fin.line(data.income_q, fin.TAX_RATE)
    )
    tax_rate = fin.latest(tax_series)
    if tax_rate is None or not 0.0 <= tax_rate < 1.0:
        tax_rate = 0.21  # tipo por defecto cuando el proveedor no lo da

    return ebit * (1.0 - tax_rate) / capital


# ---------------------------------------------------------------------------
# B6 — Márgenes en expansión
# ---------------------------------------------------------------------------
_MARGIN_LINES = {
    "operating": fin.OPERATING_INCOME,
    "gross": fin.GROSS_PROFIT,
    "net": fin.NET_INCOME,
}


@metric("margin_expansion", panel="quality", label="B6 expansión de márgenes")
def margin_expansion(data: TickerData, p: dict) -> float | None:
    """Variación interanual del margen, en puntos de margen."""
    aliases = _MARGIN_LINES.get(str(p.get("margin", "operating")), fin.OPERATING_INCOME)

    for numerator_df, revenue_df, lag in (
        (data.income_q, data.income_q, 4),
        (data.income_a, data.income_a, 1),
    ):
        numerator = fin.line(numerator_df, aliases)
        revenue = fin.line(revenue_df, fin.REVENUE)
        if numerator is None or revenue is None:
            continue
        margin = (numerator / revenue).replace([np.inf, -np.inf], np.nan).dropna()
        if len(margin) <= lag:
            continue
        return float(margin.iloc[-1] - margin.iloc[-1 - lag])
    return None


# ---------------------------------------------------------------------------
# B7 — Dilución / SBC  (mayor = peor)
# ---------------------------------------------------------------------------
@metric("dilution_sbc", panel="quality", higher_is_better=False, label="B7 dilución/SBC")
def dilution_sbc(data: TickerData, p: dict) -> float | None:
    """Crecer al 30% diluyendo 8% no es crecer al 25% recomprando."""
    years = int(p["years"])
    dilution = _share_growth_rate(data, years)

    revenue = fin.ttm_or_annual(
        fin.line(data.income_q, fin.REVENUE), fin.line(data.income_a, fin.REVENUE)
    )
    sbc = fin.ttm_or_annual(
        fin.line(data.cashflow_q, fin.SBC), fin.line(data.cashflow_a, fin.SBC)
    )

    penalty = 0.0
    if sbc is not None and revenue and revenue > 0:
        # El FCF reportado no descuenta el SBC: por encima del umbral, penaliza.
        excess = abs(sbc) / revenue - float(p["sbc_revenue_threshold"])
        penalty = float(p["sbc_weight"]) * max(0.0, excess)
    elif dilution is None:
        return None

    return (dilution or 0.0) + penalty


def _share_growth_rate(data: TickerData, years: int) -> float | None:
    """CAGR del número de acciones. Positivo = diluyendo."""
    shares = data.shares_full
    if shares is not None and len(shares) >= 2:
        series = shares.dropna().sort_index()
        if len(series) >= 2:
            end = pd.Timestamp(series.index[-1])
            if end.tz is not None:
                end = end.tz_localize(None)
            cutoff = end - pd.DateOffset(years=years)
            index = series.index.tz_localize(None) if series.index.tz is not None else series.index
            window = series[index >= cutoff]
            if len(window) >= 2 and float(window.iloc[0]) > 0:
                span_years = max((end - pd.Timestamp(
                    window.index[0].tz_localize(None) if window.index[0].tzinfo else window.index[0]
                )).days / 365.25, 0.5)
                ratio = float(window.iloc[-1]) / float(window.iloc[0])
                if ratio > 0:
                    return float(ratio ** (1.0 / span_years) - 1.0)

    diluted = fin.line(data.income_a, fin.DILUTED_SHARES)
    if diluted is not None and len(diluted) >= 2:
        span = min(years, len(diluted) - 1)
        first, last = float(diluted.iloc[-1 - span]), float(diluted.iloc[-1])
        if first > 0 and span > 0:
            return float((last / first) ** (1.0 / span) - 1.0)
    return None


# ---------------------------------------------------------------------------
# B8 — Revisiones de estimaciones
# ---------------------------------------------------------------------------
@metric("estimate_revisions", panel="quality", label="B8 revisiones de estimaciones")
def estimate_revisions(data: TickerData, p: dict) -> float | None:
    """El mejor proxy de guidance forward: anticipa el momentum de precio."""
    horizon = str(p["horizon"])
    trend_weight = float(p["trend_weight"])

    drift = None
    trend = data.eps_trend
    if trend is not None and not trend.empty and horizon in trend.index:
        row = trend.loc[horizon]
        current = pd.to_numeric(row.get("current"), errors="coerce")
        past = pd.to_numeric(row.get("90daysAgo"), errors="coerce")
        if pd.notna(current) and pd.notna(past) and abs(float(past)) > 1e-9:
            drift = float(current) / abs(float(past)) - np.sign(float(past))
            drift = clip(drift, -1.0, 1.0)

    net_ratio = None
    revisions = data.eps_revisions
    if revisions is not None and not revisions.empty and horizon in revisions.index:
        row = revisions.loc[horizon]
        up = pd.to_numeric(row.get("upLast30days"), errors="coerce")
        down = pd.to_numeric(row.get("downLast30days"), errors="coerce")
        up = 0.0 if pd.isna(up) else float(up)
        down = 0.0 if pd.isna(down) else float(down)
        if up + down > 0:
            net_ratio = (up - down) / (up + down)

    if drift is None and net_ratio is None:
        return None
    if drift is None:
        return net_ratio
    if net_ratio is None:
        return drift
    return trend_weight * drift + (1.0 - trend_weight) * net_ratio


# ---------------------------------------------------------------------------
# B9 — Salud de balance  (mayor = peor)
# ---------------------------------------------------------------------------
@metric("balance_health", panel="quality", higher_is_better=False, label="B9 net debt/EBITDA")
def balance_health(data: TickerData, p: dict) -> float | None:
    """Net debt / EBITDA. Caja neta da valor negativo, que es lo deseable."""
    debt_series = fin.first(fin.net_debt(data.balance_q), fin.net_debt(data.balance_a))
    debt = fin.latest(debt_series)
    ebitda = fin.ttm_or_annual(
        fin.line(data.income_q, fin.EBITDA), fin.line(data.income_a, fin.EBITDA)
    )
    if debt is None or ebitda is None or ebitda <= 0:
        return None
    cap = float(p["cap_net_debt_ebitda"])
    return clip(debt / ebitda, -cap, cap)


# ---------------------------------------------------------------------------
# B10 — Valoración  (mayor = peor)
# ---------------------------------------------------------------------------
@metric("valuation_ev_pct", panel="quality", higher_is_better=False, label="B10 valoración EV")
def valuation_ev_pct(data: TickerData, p: dict) -> float | None:
    """EV/EBITDA con fallback a EV/Sales. Nunca PEG: explota con crecimiento ~0."""
    ev = data.profile.get("enterpriseValue")
    if ev is None or not np.isfinite(float(ev)) or float(ev) <= 0:
        return None
    ev = float(ev)

    ebitda = fin.ttm_or_annual(
        fin.line(data.income_q, fin.EBITDA), fin.line(data.income_a, fin.EBITDA)
    )
    revenue = fin.ttm_or_annual(
        fin.line(data.income_q, fin.REVENUE), fin.line(data.income_a, fin.REVENUE)
    )

    if str(p.get("prefer", "ev_ebitda")) == "ev_ebitda" and ebitda and ebitda > 0:
        return ev / ebitda
    if revenue and revenue > 0:
        # EV/Sales y EV/EBITDA no son la misma escala, pero el percentil es por
        # sector y dentro de un sector la mezcla es minoritaria.
        return ev / revenue
    if ebitda and ebitda > 0:
        return ev / ebitda
    return None
