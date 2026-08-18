"""Panel A sobre series construidas a mano, con valor esperado conocido."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.metrics import momentum
from screener.models import TickerData
from tests.conftest import make_prices, trending_prices


def params(cfg, name):
    return cfg.metric_params(name)


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------
def test_rs_multi_window_matches_hand_computation(cfg, ticker):
    p = params(cfg, "rs_multi_window")
    close = ticker.prices["Close"]

    # las tres ventanas terminan 21 sesiones atrás, no hoy
    r3 = close.iloc[-22] / close.iloc[-85] - 1
    r6 = close.iloc[-22] / close.iloc[-148] - 1
    r12 = close.iloc[-22] / close.iloc[-274] - 1
    r1 = close.iloc[-1] / close.iloc[-22] - 1
    expected = 0.3 * r3 + 0.3 * r6 + 0.4 * r12
    assert r1 < p["reversal_threshold"], "la serie suave no debe activar la penalización"

    assert momentum.rs_multi_window(ticker, p) == pytest.approx(expected)


def test_rs_multi_window_penalises_a_vertical_last_month(cfg):
    """Dos valores con igual retorno 12-1; el que lo hizo de golpe puntúa menos."""
    p = params(cfg, "rs_multi_window")
    smooth = trending_prices(days=400, daily=0.001)

    closes = smooth["Close"].to_numpy().copy()
    closes[-21:] = closes[-22] * np.cumprod(np.full(21, 1.02))  # +50% en un mes
    spiky = make_prices(closes)

    smooth_data = TickerData(symbol="S", asof=date(2024, 7, 15), prices=smooth)
    spiky_data = TickerData(symbol="V", asof=date(2024, 7, 15), prices=spiky)

    assert momentum.rs_multi_window(spiky_data, p) < momentum.rs_multi_window(smooth_data, p)


def test_rs_multi_window_needs_a_year_of_history(cfg):
    short = TickerData(symbol="S", asof=date(2024, 7, 15), prices=trending_prices(days=100))
    assert momentum.rs_multi_window(short, params(cfg, "rs_multi_window")) is None


# ---------------------------------------------------------------------------
# A2
# ---------------------------------------------------------------------------
def test_distance_to_high_is_zero_at_the_high(cfg, ticker):
    """Una serie monótona termina en máximos: distancia 0, sin contar el bonus."""
    value = momentum.distance_to_high(ticker, params(cfg, "distance_to_high"))
    bonus = params(cfg, "distance_to_high")["breakout_bonus"]
    assert value in (pytest.approx(0.0), pytest.approx(bonus))


def test_distance_to_high_penalises_a_drawdown(cfg):
    p = params(cfg, "distance_to_high")
    closes = trending_prices(days=300)["Close"].to_numpy().copy()
    closes[-30:] = closes[-31] * 0.8  # cae un 20% desde máximos
    data = TickerData(symbol="D", asof=date(2024, 7, 15), prices=make_prices(closes))

    assert momentum.distance_to_high(data, p) == pytest.approx(-0.2, abs=0.01)


def test_distance_to_high_rewards_a_breakout_on_volume(cfg):
    """Misma ruptura, distinto volumen: solo la que va con volumen cobra el bonus."""
    p = params(cfg, "distance_to_high")
    closes = np.concatenate([np.full(250, 100.0), np.linspace(100.0, 115.0, 20)])

    quiet = make_prices(closes, volumes=np.full(len(closes), 1e6))
    loud_volumes = np.full(len(closes), 1e6)
    loud_volumes[-20:] = 5e6
    loud = make_prices(closes, volumes=loud_volumes)

    quiet_value = momentum.distance_to_high(TickerData("Q", date(2024, 7, 15), prices=quiet), p)
    loud_value = momentum.distance_to_high(TickerData("L", date(2024, 7, 15), prices=loud), p)

    assert loud_value == pytest.approx(quiet_value + p["breakout_bonus"])


# ---------------------------------------------------------------------------
# A3 / A4
# ---------------------------------------------------------------------------
def test_rs_vs_sector_detects_disguised_weakness(cfg):
    """Subir 20% en un sector que sube 25% es debilidad, y sale negativo."""
    p = params(cfg, "rs_vs_sector")
    window = p["window"]
    stock = make_prices(np.linspace(100.0, 120.0, window + 10))
    sector = pd.Series(np.linspace(100.0, 125.0, window + 10), index=stock.index)

    data = TickerData("W", date(2024, 7, 15), prices=stock, sector_index=sector)
    assert momentum.rs_vs_sector(data, p) < 0


def test_rs_vs_sector_needs_a_sector_index(cfg, ticker):
    assert momentum.rs_vs_sector(ticker, params(cfg, "rs_vs_sector")) is None


def test_sector_momentum_is_positive_for_a_rising_sector(cfg, prices):
    p = params(cfg, "sector_momentum")
    rising = pd.Series(trending_prices(days=400)["Close"].to_numpy(), index=prices.index)
    data = TickerData("S", date(2024, 7, 15), prices=prices, sector_index=rising)
    assert momentum.sector_momentum(data, p) > 0


# ---------------------------------------------------------------------------
# A5
# ---------------------------------------------------------------------------
def test_consistency_prefers_the_drip_over_the_gap(cfg):
    """Mismo retorno total: a goteo persiste, a saltos no."""
    p = params(cfg, "consistency_fitp")
    window = p["window"]

    drip = make_prices(100.0 * np.cumprod(np.full(window + 10, 1.002)))

    gappy_closes = np.full(window + 10, 100.0)
    gappy_closes[-40] = 100.0
    gappy_closes[-39:] = drip["Close"].iloc[-1]  # todo el movimiento en un salto
    gappy = make_prices(gappy_closes)

    drip_value = momentum.consistency_fitp(TickerData("D", date(2024, 7, 15), prices=drip), p)
    gap_value = momentum.consistency_fitp(TickerData("G", date(2024, 7, 15), prices=gappy), p)

    assert drip_value > gap_value


# ---------------------------------------------------------------------------
# A6
# ---------------------------------------------------------------------------
def test_vol_adjusted_momentum_prefers_the_calm_riser(cfg):
    """Mismo retorno, distinta volatilidad: gana el que llegó sin sobresaltos."""
    p = params(cfg, "vol_adjusted_momentum")
    days = p["window"] + 30

    calm_closes = 100.0 * np.cumprod(np.full(days, 1.002))
    noise = np.tile([1.08, 1 / 1.08], days // 2 + 1)[:days]
    wild_closes = calm_closes * noise

    calm = TickerData("C", date(2024, 7, 15), prices=make_prices(calm_closes))
    wild = TickerData("W", date(2024, 7, 15), prices=make_prices(wild_closes))

    assert momentum.vol_adjusted_momentum(calm, p) > momentum.vol_adjusted_momentum(wild, p)


# ---------------------------------------------------------------------------
# A7
# ---------------------------------------------------------------------------
def test_accumulation_is_positive_when_volume_follows_the_rise(cfg):
    p = params(cfg, "accumulation_obv")
    days = p["window"] + 60
    rising = make_prices(100.0 * np.cumprod(np.full(days, 1.002)))
    falling = make_prices(100.0 * np.cumprod(np.full(days, 0.998)))

    up = momentum.accumulation_obv(TickerData("U", date(2024, 7, 15), prices=rising), p)
    down = momentum.accumulation_obv(TickerData("D", date(2024, 7, 15), prices=falling), p)

    assert up > 0 > down


# ---------------------------------------------------------------------------
# A8
# ---------------------------------------------------------------------------
def test_pead_reaction_measures_the_jump_and_the_hold(cfg):
    p = params(cfg, "pead_reaction")
    closes = np.full(300, 100.0)
    closes[-30:] = 110.0  # salta un 10% al publicar y lo mantiene
    prices = make_prices(closes)
    event = prices.index[-30]

    data = TickerData(
        "P",
        date(2024, 7, 15),
        prices=prices,
        earnings_dates=pd.DataFrame({"Reported EPS": [1.0]}, index=[event]),
    )
    # salta 10% y mantiene: jump = 0.10, hold = 0
    assert momentum.pead_reaction(data, p) == pytest.approx(0.10)


def test_pead_reaction_ignores_stale_earnings(cfg):
    p = params(cfg, "pead_reaction")
    prices = make_prices(np.full(400, 100.0))
    stale = prices.index[0]  # muy anterior a max_age_days
    data = TickerData(
        "P",
        date(2024, 7, 15),
        prices=prices,
        earnings_dates=pd.DataFrame({"Reported EPS": [1.0]}, index=[stale]),
    )
    assert momentum.pead_reaction(data, p) is None


# ---------------------------------------------------------------------------
# A10
# ---------------------------------------------------------------------------
def test_event_proximity_penalises_but_never_vetoes(cfg):
    p = params(cfg, "event_proximity")
    today = date(2024, 7, 15)
    prices = make_prices(np.full(300, 100.0))

    def score(days_ahead):
        event = pd.Timestamp(today) + pd.Timedelta(days=days_ahead)
        data = TickerData(
            "E", today, prices=prices, earnings_dates=pd.DataFrame({"EPS Estimate": [1.0]}, index=[event])
        )
        return momentum.event_proximity(data, p)

    assert score(0) == pytest.approx(p["min_score"])      # penaliza al máximo...
    assert score(0) > 0                                    # ...pero nunca veta
    assert score(30) == 1.0
    assert score(0) < score(5) < score(30)


def test_event_proximity_is_neutral_without_a_calendar(cfg, ticker):
    assert momentum.event_proximity(ticker, params(cfg, "event_proximity")) == 1.0
