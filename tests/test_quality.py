"""Panel B sobre estados financieros sintéticos."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.metrics import quality
from screener.models import TickerData
from tests.conftest import statement

ANNUAL = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]
QUARTERS = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]


def params(cfg, name):
    return cfg.metric_params(name)


def data(**kwargs) -> TickerData:
    return TickerData(symbol="TEST", asof=date(2025, 2, 1), sector="Technology", **kwargs)


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------
def test_revenue_growth_uses_the_freshest_yoy(cfg):
    """Con 5 trimestres hay un interanual: manda sobre el anual."""
    quarterly = statement({"Total Revenue": [100, 110, 120, 130, 150]}, QUARTERS)
    annual = statement({"Total Revenue": [200, 300, 400, 440]}, ANNUAL)

    value = quality.revenue_growth_level(
        data(income_q=quarterly, income_a=annual), params(cfg, "revenue_growth_level")
    )
    assert value == pytest.approx(0.5)  # 150 vs 100 un año antes


def test_revenue_growth_falls_back_to_annual(cfg):
    """Yahoo no siempre da 5 trimestres; entonces vale el interanual anual."""
    annual = statement({"Total Revenue": [200, 300, 400, 440]}, ANNUAL)
    value = quality.revenue_growth_level(data(income_a=annual), params(cfg, "revenue_growth_level"))
    assert value == pytest.approx(0.1)


def test_revenue_growth_is_none_without_statements(cfg):
    assert quality.revenue_growth_level(data(), params(cfg, "revenue_growth_level")) is None


# ---------------------------------------------------------------------------
# B2
# ---------------------------------------------------------------------------
def test_growth_duration_rewards_persistence(cfg):
    """Crecer >20% cuatro años seguidos puntúa más que hacerlo solo el último."""
    p = params(cfg, "growth_trend_duration")
    persistent = statement({"Total Revenue": [100, 130, 170, 220]}, ANNUAL)
    one_off = statement({"Total Revenue": [100, 101, 102, 140]}, ANNUAL)

    assert quality.growth_trend_duration(data(income_a=persistent), p) > quality.growth_trend_duration(
        data(income_a=one_off), p
    )


def test_growth_duration_tolerates_a_single_blip(cfg):
    """La regla '2 de los últimos 3' evita vetar por un trimestre flojo aislado."""
    p = params(cfg, "growth_trend_duration")
    k, n = p["tolerant_rule"]
    # crece 30%, se frena, vuelve a crecer 30%: 2 de los últimos 3 por encima
    blip = statement({"Total Revenue": [100, 130, 132, 172]}, ANNUAL)

    assert quality.growth_trend_duration(data(income_a=blip), p) >= (1 - p["slope_weight"]) * (k / n)


def test_growth_duration_needs_two_periods(cfg):
    single = statement({"Total Revenue": [100]}, ["2024-12-31"])
    assert quality.growth_trend_duration(data(income_a=single), params(cfg, "growth_trend_duration")) is None


# ---------------------------------------------------------------------------
# B3
# ---------------------------------------------------------------------------
def test_surprise_averages_four_quarters_not_one(cfg):
    """Un trimestre espectacular no compensa tres decepciones."""
    p = params(cfg, "earnings_surprise_4q")
    index = pd.to_datetime(["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"])

    steady = pd.DataFrame({"Surprise(%)": [5.0, 5.0, 5.0, 5.0]}, index=index)
    lumpy = pd.DataFrame({"Surprise(%)": [-10.0, -10.0, -10.0, 40.0]}, index=index)

    # sin precios no hay reacción: queda solo la media de sorpresas
    assert quality.earnings_surprise_4q(data(earnings_dates=steady), p) == pytest.approx(0.05)
    assert quality.earnings_surprise_4q(data(earnings_dates=lumpy), p) == pytest.approx(0.025)


def test_surprise_is_none_without_a_calendar(cfg):
    assert quality.earnings_surprise_4q(data(), params(cfg, "earnings_surprise_4q")) is None


# ---------------------------------------------------------------------------
# B4
# ---------------------------------------------------------------------------
def test_cash_quality_flags_accruals(cfg):
    """Beneficio contable sin caja detrás (Sloan): bandera roja bajo FCF/NI 0,8."""
    p = params(cfg, "cash_quality_fcf_ni")
    periods = QUARTERS[-4:]

    def build(fcf_per_quarter):
        return data(
            cashflow_q=statement({"Free Cash Flow": [fcf_per_quarter] * 4}, periods),
            income_q=statement({"Net Income": [25.0] * 4}, periods),
        )

    clean = quality.cash_quality_fcf_ni(build(30.0), p)   # FCF/NI = 1.2
    dirty = quality.cash_quality_fcf_ni(build(10.0), p)   # FCF/NI = 0.4, bajo el umbral

    assert clean > dirty
    # el doble castigo hace que caer bajo el umbral duela más que la diferencia lineal
    assert dirty < p["red_flag_ratio"]


def test_cash_quality_needs_positive_earnings(cfg):
    """Con pérdidas la conversión FCF/NI no es interpretable."""
    p = params(cfg, "cash_quality_fcf_ni")
    periods = QUARTERS[-4:]
    losing = data(
        cashflow_q=statement({"Free Cash Flow": [10.0] * 4}, periods),
        income_q=statement({"Net Income": [-25.0] * 4}, periods),
    )
    assert quality.cash_quality_fcf_ni(losing, p) is None


def test_cash_quality_reconstructs_fcf_from_ocf_and_capex(cfg):
    """Si falta la fila Free Cash Flow se reconstruye con OCF - CapEx."""
    p = params(cfg, "cash_quality_fcf_ni")
    periods = QUARTERS[-4:]
    reconstructed = data(
        # CapEx viene en negativo en Yahoo
        cashflow_q=statement(
            {"Operating Cash Flow": [40.0] * 4, "Capital Expenditure": [-10.0] * 4}, periods
        ),
        income_q=statement({"Net Income": [25.0] * 4}, periods),
    )
    direct = data(
        cashflow_q=statement({"Free Cash Flow": [30.0] * 4}, periods),
        income_q=statement({"Net Income": [25.0] * 4}, periods),
    )
    assert quality.cash_quality_fcf_ni(reconstructed, p) == quality.cash_quality_fcf_ni(direct, p)


# ---------------------------------------------------------------------------
# B5
# ---------------------------------------------------------------------------
def test_roic_uses_invested_capital_not_equity(cfg):
    """Dos empresas con igual EBIT: la que emplea menos capital tiene mejor ROIC."""
    p = params(cfg, "roic_vs_sector")
    periods = QUARTERS[-4:]
    income = statement({"EBIT": [25.0] * 4, "Tax Rate For Calcs": [0.2] * 4}, periods)

    light = data(income_q=income, balance_q=statement({"Invested Capital": [200.0] * 4}, periods))
    heavy = data(income_q=income, balance_q=statement({"Invested Capital": [1000.0] * 4}, periods))

    assert quality.roic_vs_sector(light, p) == pytest.approx(100 * 0.8 / 200)
    assert quality.roic_vs_sector(light, p) > quality.roic_vs_sector(heavy, p)


# ---------------------------------------------------------------------------
# B6
# ---------------------------------------------------------------------------
def test_margin_expansion_measures_the_yoy_delta(cfg):
    p = params(cfg, "margin_expansion")
    quarterly = statement(
        {"Operating Income": [10, 12, 14, 16, 20], "Total Revenue": [100, 100, 100, 100, 100]},
        QUARTERS,
    )
    # margen pasa de 10% a 20% en un año
    assert quality.margin_expansion(data(income_q=quarterly), p) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# B7
# ---------------------------------------------------------------------------
def test_dilution_is_positive_when_diluting_negative_when_buying_back(cfg):
    """Métrica de "mal": positivo = diluye. El registry la invierte al percentilar."""
    p = params(cfg, "dilution_sbc")
    index = pd.to_datetime(["2022-01-01", "2025-01-01"])

    diluting = data(shares_full=pd.Series([100e6, 121e6], index=index))
    buying_back = data(shares_full=pd.Series([100e6, 90e6], index=index))

    assert quality.dilution_sbc(diluting, p) > 0
    assert quality.dilution_sbc(buying_back, p) < 0


def test_sbc_above_the_threshold_penalises(cfg):
    """El FCF reportado no descuenta el SBC: por encima del 15% de revenue, penaliza."""
    p = params(cfg, "dilution_sbc")
    periods = QUARTERS[-4:]
    index = pd.to_datetime(["2022-01-01", "2025-01-01"])
    shares = pd.Series([100e6, 100e6], index=index)  # sin dilución de acciones

    heavy_sbc = data(
        shares_full=shares,
        income_q=statement({"Total Revenue": [100.0] * 4}, periods),
        cashflow_q=statement({"Stock Based Compensation": [30.0] * 4}, periods),  # 30% de revenue
    )
    light_sbc = data(
        shares_full=shares,
        income_q=statement({"Total Revenue": [100.0] * 4}, periods),
        cashflow_q=statement({"Stock Based Compensation": [5.0] * 4}, periods),
    )

    assert quality.dilution_sbc(heavy_sbc, p) > quality.dilution_sbc(light_sbc, p)
    assert quality.dilution_sbc(light_sbc, p) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# B8
# ---------------------------------------------------------------------------
def test_estimate_revisions_reads_drift_and_net_count(cfg):
    p = params(cfg, "estimate_revisions")
    horizon = p["horizon"]

    upgraded = data(
        eps_trend=pd.DataFrame({"current": [11.0], "90daysAgo": [10.0]}, index=[horizon]),
        eps_revisions=pd.DataFrame({"upLast30days": [8], "downLast30days": [2]}, index=[horizon]),
    )
    downgraded = data(
        eps_trend=pd.DataFrame({"current": [9.0], "90daysAgo": [10.0]}, index=[horizon]),
        eps_revisions=pd.DataFrame({"upLast30days": [2], "downLast30days": [8]}, index=[horizon]),
    )

    assert quality.estimate_revisions(upgraded, p) > 0 > quality.estimate_revisions(downgraded, p)


def test_estimate_revisions_survives_with_only_one_source(cfg):
    """En Europa y emergentes suele faltar una de las dos tablas."""
    p = params(cfg, "estimate_revisions")
    only_counts = data(
        eps_revisions=pd.DataFrame({"upLast30days": [6], "downLast30days": [0]}, index=[p["horizon"]])
    )
    assert quality.estimate_revisions(only_counts, p) == pytest.approx(1.0)
    assert quality.estimate_revisions(data(), p) is None


# ---------------------------------------------------------------------------
# B9
# ---------------------------------------------------------------------------
def test_balance_health_is_negative_with_net_cash(cfg):
    p = params(cfg, "balance_health")
    periods = QUARTERS[-4:]
    income = statement({"EBITDA": [25.0] * 4}, periods)

    net_cash = data(income_q=income, balance_q=statement({"Net Debt": [-50.0] * 4}, periods))
    levered = data(income_q=income, balance_q=statement({"Net Debt": [300.0] * 4}, periods))

    assert quality.balance_health(net_cash, p) < 0 < quality.balance_health(levered, p)


def test_balance_health_caps_outliers(cfg):
    """Sin tope, un solo apalancado extremo aplasta el percentil de todos los demás."""
    p = params(cfg, "balance_health")
    periods = QUARTERS[-4:]
    extreme = data(
        income_q=statement({"EBITDA": [1.0] * 4}, periods),
        balance_q=statement({"Net Debt": [10_000.0] * 4}, periods),
    )
    assert quality.balance_health(extreme, p) == pytest.approx(p["cap_net_debt_ebitda"])


# ---------------------------------------------------------------------------
# B10
# ---------------------------------------------------------------------------
def test_valuation_prefers_ev_ebitda_and_falls_back_to_ev_sales(cfg):
    p = params(cfg, "valuation_ev_pct")
    periods = QUARTERS[-4:]

    with_ebitda = data(
        profile={"enterpriseValue": 1000.0},
        income_q=statement({"EBITDA": [25.0] * 4, "Total Revenue": [50.0] * 4}, periods),
    )
    assert quality.valuation_ev_pct(with_ebitda, p) == pytest.approx(10.0)  # 1000 / 100

    no_ebitda = data(
        profile={"enterpriseValue": 1000.0},
        income_q=statement({"Total Revenue": [50.0] * 4}, periods),
    )
    assert quality.valuation_ev_pct(no_ebitda, p) == pytest.approx(5.0)  # 1000 / 200


def test_valuation_is_none_without_enterprise_value(cfg):
    assert quality.valuation_ev_pct(data(profile={}), params(cfg, "valuation_ev_pct")) is None
