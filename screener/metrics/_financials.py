"""Acceso normalizado a los estados financieros.

Yahoo devuelve los estados como DataFrame con filas = concepto contable y
columnas = periodo, y el nombre de la fila cambia entre empresas ("Net Income"
vs "Net Income Common Stockholders"). Todo acceso pasa por `line()`, que prueba
alias y devuelve siempre una serie **ordenada de más antigua a más reciente**,
sin NaN.

Limitación de la fuente gratuita: Yahoo solo sirve ~5 trimestres de cuenta de
resultados, insuficiente para medir 4 crecimientos interanuales trimestrales.
Por eso `growth_history()` usa los anuales (4-5 ejercicios) y sube a trimestral
en cuanto hay profundidad — lo que ocurre solo cuando la caché acumulativa lleva
meses funcionando (`data.accumulate_statements`).
"""

from __future__ import annotations

import pandas as pd

REVENUE = ("Total Revenue", "Operating Revenue")
NET_INCOME = (
    "Net Income",
    "Net Income Common Stockholders",
    "Net Income Continuous Operations",
    "Net Income From Continuing Operation Net Minority Interest",
)
FREE_CASH_FLOW = ("Free Cash Flow",)
OPERATING_CASH_FLOW = ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
CAPEX = ("Capital Expenditure", "Purchase Of PPE")
SBC = ("Stock Based Compensation",)
EBIT = ("EBIT", "Operating Income", "Total Operating Income As Reported")
EBITDA = ("EBITDA", "Normalized EBITDA")
GROSS_PROFIT = ("Gross Profit",)
OPERATING_INCOME = ("Operating Income", "Total Operating Income As Reported", "EBIT")
NET_DEBT = ("Net Debt",)
TOTAL_DEBT = ("Total Debt", "Long Term Debt And Capital Lease Obligation")
CASH = ("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
INVESTED_CAPITAL = ("Invested Capital",)
STOCKHOLDERS_EQUITY = ("Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest")
TAX_RATE = ("Tax Rate For Calcs",)
DILUTED_SHARES = ("Diluted Average Shares", "Basic Average Shares")


def line(df: pd.DataFrame | None, aliases: tuple[str, ...]) -> pd.Series | None:
    """Primera fila que exista de entre `aliases`, ascendente por fecha y sin NaN."""
    if df is None or df.empty:
        return None
    for alias in aliases:
        if alias in df.index:
            series = pd.to_numeric(df.loc[alias], errors="coerce").dropna()
            if series.empty:
                continue
            series = series.sort_index()
            if len(series):
                return series
    return None


def first(*candidates: pd.Series | None) -> pd.Series | None:
    """Primera serie no vacía.

    Existe porque `serie_a or serie_b` lanza ValueError en pandas: el valor de
    verdad de una Series es ambiguo. Ese fallo lo silenciaría el registry y la
    métrica quedaría a cero de cobertura sin que se note.
    """
    for series in candidates:
        if series is not None and len(series):
            return series
    return None


def latest(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    return float(series.iloc[-1])


def ttm(series: pd.Series | None, periods: int = 4) -> float | None:
    """Suma de los últimos `periods` trimestres. None si no hay suficientes."""
    if series is None or len(series) < periods:
        return None
    return float(series.iloc[-periods:].sum())


def ttm_or_annual(
    quarterly: pd.Series | None, annual: pd.Series | None, periods: int = 4
) -> float | None:
    """TTM trimestral si hay 4 trimestres; si no, el último ejercicio anual."""
    value = ttm(quarterly, periods)
    return value if value is not None else latest(annual)


def growth(series: pd.Series | None, lag: int) -> float | None:
    """Crecimiento del último periodo contra `lag` periodos atrás."""
    if series is None or len(series) <= lag:
        return None
    now, before = float(series.iloc[-1]), float(series.iloc[-1 - lag])
    if before == 0 or pd.isna(before) or pd.isna(now):
        return None
    if before < 0:
        # crecer desde pérdidas no es un porcentaje interpretable
        return None
    return now / before - 1.0


def growth_series(series: pd.Series | None, lag: int) -> pd.Series | None:
    """Serie de crecimientos interanuales, ascendente."""
    if series is None or len(series) <= lag:
        return None
    prev = series.shift(lag)
    result = (series / prev - 1.0).replace([float("inf"), float("-inf")], pd.NA)
    result = result.where(prev > 0).dropna()
    return result if len(result) else None


def growth_latest(
    quarterly: pd.Series | None, annual: pd.Series | None
) -> float | None:
    """Crecimiento interanual más fresco disponible: trimestral > anual."""
    value = growth(quarterly, 4)
    return value if value is not None else growth(annual, 1)


def growth_history(
    quarterly: pd.Series | None, annual: pd.Series | None, *, min_points: int = 3
) -> pd.Series | None:
    """Historia de crecimiento interanual con la mayor resolución disponible.

    Prefiere trimestral (necesita >= min_points+4 trimestres, hoy raro con Yahoo)
    y cae a anual, que da 3-4 puntos.
    """
    quarterly_growth = growth_series(quarterly, 4)
    if quarterly_growth is not None and len(quarterly_growth) >= min_points:
        return quarterly_growth
    annual_growth = growth_series(annual, 1)
    if annual_growth is not None and len(annual_growth) >= 2:
        return annual_growth
    return quarterly_growth


def free_cash_flow(
    cashflow: pd.DataFrame | None, periods: int | None = None
) -> pd.Series | None:
    """FCF de la fila directa, o reconstruido como OCF - CapEx."""
    direct = line(cashflow, FREE_CASH_FLOW)
    if direct is not None:
        return direct
    ocf = line(cashflow, OPERATING_CASH_FLOW)
    capex = line(cashflow, CAPEX)
    if ocf is None or capex is None:
        return None
    # CapEx viene en negativo en Yahoo; sumar es restar la inversión.
    combined = (ocf + capex).dropna()
    return combined if len(combined) else None


def net_debt(balance: pd.DataFrame | None) -> pd.Series | None:
    direct = line(balance, NET_DEBT)
    if direct is not None:
        return direct
    debt = line(balance, TOTAL_DEBT)
    cash = line(balance, CASH)
    if debt is None:
        return None
    if cash is None:
        return debt
    combined = (debt - cash).dropna()
    return combined if len(combined) else None


def invested_capital(balance: pd.DataFrame | None) -> pd.Series | None:
    direct = line(balance, INVESTED_CAPITAL)
    if direct is not None:
        return direct
    debt = line(balance, TOTAL_DEBT)
    equity = line(balance, STOCKHOLDERS_EQUITY)
    if equity is None:
        return None
    if debt is None:
        return equity
    combined = (debt + equity).dropna()
    return combined if len(combined) else None
