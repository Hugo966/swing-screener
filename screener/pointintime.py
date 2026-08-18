"""Reconstrucción de un `TickerData` tal como se veía en una fecha pasada (§11).

Es la pieza que decide si el backtest vale algo. Si el Q3 se usa desde el 30 de
septiembre en vez de desde que se publicó a finales de octubre, el backtest sale
glorioso y en real no funciona.

**Fecha de publicación.** yfinance no da el filing date. El sustituto es
`earnings_dates`, que sí trae las fechas de *report* históricas: el estado de un
periodo que cierra el 30/09 se considera disponible desde el primer report
posterior a esa fecha. Cuando no hay calendario se retrasa un número fijo de días
configurable, que es peor pero explícito.

**Lo que no se puede reconstruir.** `eps_revisions` y `eps_trend` son una foto de
hoy, no una serie con vintages: en una fecha pasada no hay forma de saber cuál era
la estimación vigente. Se anulan, y B8 se cae sola por el mecanismo de cobertura.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from screener.metrics import _financials as fin
from screener.models import TickerData

log = logging.getLogger(__name__)


def _naive(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz is not None else idx


def report_dates(earnings_dates: pd.DataFrame | None) -> pd.DatetimeIndex:
    """Fechas de publicación conocidas, ascendentes."""
    if earnings_dates is None or earnings_dates.empty:
        return pd.DatetimeIndex([])
    return _naive(earnings_dates.index).sort_values()


def publication_date(
    period_end: pd.Timestamp, reports: pd.DatetimeIndex, fallback_lag_days: int
) -> pd.Timestamp:
    """Cuándo se hizo público el estado de un periodo que cierra en `period_end`.

    El primer report posterior al cierre del periodo. Sin calendario, el cierre
    más un retraso fijo.
    """
    period_end = pd.Timestamp(period_end)
    if period_end.tz is not None:
        period_end = period_end.tz_localize(None)
    later = reports[reports > period_end]
    if len(later):
        return later[0]
    return period_end + pd.Timedelta(days=fallback_lag_days)


def truncate_statement(
    frame: pd.DataFrame | None,
    asof: pd.Timestamp,
    reports: pd.DatetimeIndex,
    fallback_lag_days: int,
) -> pd.DataFrame | None:
    """Deja solo los periodos ya publicados en `asof`."""
    if frame is None or frame.empty:
        return None
    keep = [
        column
        for column in frame.columns
        if publication_date(column, reports, fallback_lag_days) <= asof
    ]
    if not keep:
        return None
    return frame[keep]


def _truncate_prices(prices: pd.DataFrame | None, asof: pd.Timestamp) -> pd.DataFrame | None:
    if prices is None or prices.empty:
        return None
    index = _naive(prices.index)
    sliced = prices.loc[index <= asof]
    return sliced if len(sliced) else None


def _truncate_series(series: pd.Series | None, asof: pd.Timestamp) -> pd.Series | None:
    if series is None or len(series) == 0:
        return None
    index = _naive(series.index)
    sliced = series.loc[index <= asof]
    return sliced if len(sliced) else None


def _truncate_frame(frame: pd.DataFrame | None, asof: pd.Timestamp) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    index = _naive(frame.index)
    sliced = frame.loc[index <= asof]
    return sliced if len(sliced) else None


def reconstruct_profile(
    data: TickerData, asof: pd.Timestamp, prices: pd.DataFrame | None
) -> dict:
    """Capitalización y enterprise value **en la fecha**, no los de hoy.

    B10 necesita el EV. El de `info` es el actual, que es look-ahead puro en una
    fecha pasada, así que se reconstruye: EV = acciones × precio + deuda neta,
    todo con lo conocido en `asof`.
    """
    profile = {
        key: data.profile.get(key)
        for key in ("sector", "industry", "currency", "financialCurrency", "quoteType",
                    "exchange", "longName", "shortName")
    }

    price = None
    if prices is not None and len(prices):
        price = float(prices["Close"].iloc[-1])

    shares = None
    shares_series = _truncate_series(data.shares_full, asof)
    if shares_series is not None and len(shares_series):
        shares = float(shares_series.iloc[-1])
    else:
        diluted = fin.line(data.income_a, fin.DILUTED_SHARES)
        if diluted is not None and len(diluted):
            shares = float(diluted.iloc[-1])

    if price is not None and shares:
        market_cap = price * shares
        profile["marketCap"] = market_cap
        net_debt = fin.latest(fin.first(fin.net_debt(data.balance_q), fin.net_debt(data.balance_a)))
        profile["enterpriseValue"] = market_cap + (net_debt or 0.0)
        profile["sharesOutstanding"] = shares

    return profile


def as_of(
    data: TickerData,
    asof: date | pd.Timestamp,
    *,
    fallback_lag_days: int = 60,
    drop_estimates: bool = True,
) -> TickerData:
    """El mismo ticker, recortado a lo que se sabía en `asof`."""
    stamp = pd.Timestamp(asof)
    if stamp.tz is not None:
        stamp = stamp.tz_localize(None)

    reports = report_dates(data.earnings_dates)
    prices = _truncate_prices(data.prices, stamp)

    past = TickerData(
        symbol=data.symbol,
        asof=stamp.date(),
        sector=data.sector,
        industry=data.industry,
        prices=prices,
        benchmark=_truncate_series(data.benchmark, stamp),
        sector_index=_truncate_series(data.sector_index, stamp),
        income_q=truncate_statement(data.income_q, stamp, reports, fallback_lag_days),
        income_a=truncate_statement(data.income_a, stamp, reports, fallback_lag_days),
        cashflow_q=truncate_statement(data.cashflow_q, stamp, reports, fallback_lag_days),
        cashflow_a=truncate_statement(data.cashflow_a, stamp, reports, fallback_lag_days),
        balance_q=truncate_statement(data.balance_q, stamp, reports, fallback_lag_days),
        balance_a=truncate_statement(data.balance_a, stamp, reports, fallback_lag_days),
        earnings_dates=_truncate_frame(data.earnings_dates, stamp),
        # Vintages inexistentes: en una fecha pasada no se puede saber cuál era la
        # estimación vigente. B8 se cae sola por cobertura.
        eps_revisions=None if drop_estimates else data.eps_revisions,
        eps_trend=None if drop_estimates else data.eps_trend,
        shares_full=_truncate_series(data.shares_full, stamp),
    )
    past.profile = reconstruct_profile(past, stamp, prices)
    return past
