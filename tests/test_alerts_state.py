"""Formato de mensajes, troceo y cooldown de alertas."""

from __future__ import annotations

from datetime import date, timedelta


from screener.alerts import (
    TELEGRAM_LIMIT,
    format_alert,
    split_message,
    to_plain_text,
)
from screener.models import MetricResult, PanelBreakdown, TickerResult
from screener.state import AlertState


def sample_result(**kwargs):
    momentum = PanelBreakdown(
        panel="momentum",
        raw_score=88.0,
        coverage=1.0,
        metrics=[
            MetricResult("rs_multi_window", 0.42, 95.0, 18.0),
            MetricResult("distance_to_high", -0.03, 90.0, 14.0),
            MetricResult("pead_reaction", None, 50.0, 8.0, imputed=True),
        ],
    )
    quality = PanelBreakdown(
        panel="quality",
        raw_score=81.0,
        coverage=0.9,
        metrics=[MetricResult("revenue_growth_level", 0.35, 88.0, 14.0)],
    )
    defaults = dict(
        symbol="AMD",
        region="us",
        sector="Technology",
        name="Advanced Micro Devices",
        price=234.5,
        momentum=momentum,
        quality=quality,
        a_pct=82.0,
        b_pct=88.0,
        regime=0.86,
        score_final=62.7,
        alert=True,
    )
    defaults.update(kwargs)
    return TickerResult(**defaults)


def test_alert_includes_the_per_metric_breakdown():
    """El desglose no es opcional: sin él no se puede tunear ni entender la alerta."""
    message = format_alert(sample_result(), "benchmark +6.4% vs MM200")

    assert "AMD" in message
    assert "A1 RS multi-ventana" in message
    assert "A2 distancia a máximos" in message
    assert "B1 crecimiento revenue Y/Y" in message
    assert "82" in message and "88" in message and "0.86" in message
    assert "benchmark +6.4% vs MM200" in message


def test_imputed_metrics_are_flagged():
    """Hay que poder distinguir un percentil real de uno imputado por falta de dato."""
    message = format_alert(sample_result())
    pead_line = next(line for line in message.split("\n") if "A8" in line)
    assert "s/d" in pead_line and "·imp" in pead_line


def test_watchlist_names_are_marked():
    assert "👁" in format_alert(sample_result(is_watchlist=True))
    assert "👁" not in format_alert(sample_result(is_watchlist=False))


def test_html_is_escaped_in_company_names():
    """Un nombre con < o & rompería el parse_mode=HTML de Telegram."""
    message = format_alert(sample_result(name="Smith & Wesson <Brands>"))
    assert "Smith &amp; Wesson &lt;Brands&gt;" in message


def test_plain_text_strips_the_markup():
    plain = to_plain_text(format_alert(sample_result()))
    assert "<b>" not in plain and "</b>" not in plain
    assert "AMD" in plain


def test_split_respects_the_telegram_limit():
    message = "\n".join(f"línea {i} con algo de texto" for i in range(2000))
    chunks = split_message(message)

    assert len(chunks) > 1
    assert all(len(chunk) <= TELEGRAM_LIMIT for chunk in chunks)
    # no se pierde contenido
    assert "línea 0 " in chunks[0] and "línea 1999 " in chunks[-1]


def test_split_handles_a_single_overlong_line():
    chunks = split_message("x" * (TELEGRAM_LIMIT * 2 + 10))
    assert all(len(chunk) <= TELEGRAM_LIMIT for chunk in chunks)
    assert sum(len(c) for c in chunks) == TELEGRAM_LIMIT * 2 + 10


def test_short_message_is_not_split():
    assert split_message("corto") == ["corto"]


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------
def test_cooldown_suppresses_the_repeat_alert(tmp_path):
    """Un valor que aguanta semanas en el top no debe alertar cada tarde."""
    state = AlertState(tmp_path / "state.sqlite")
    result = sample_result()
    today = date(2025, 3, 10)

    assert not state.in_cooldown("AMD", "us", 10, today)
    state.record(result, today)

    assert state.in_cooldown("AMD", "us", 10, today + timedelta(days=3))
    assert not state.in_cooldown("AMD", "us", 10, today + timedelta(days=11))


def test_cooldown_is_per_symbol_and_region(tmp_path):
    state = AlertState(tmp_path / "state.sqlite")
    today = date(2025, 3, 10)
    state.record(sample_result(), today)

    assert not state.in_cooldown("NVDA", "us", 10, today)
    assert not state.in_cooldown("AMD", "europe_dev", 10, today)


def test_cooldown_disabled_when_zero(tmp_path):
    state = AlertState(tmp_path / "state.sqlite")
    today = date(2025, 3, 10)
    state.record(sample_result(), today)
    assert not state.in_cooldown("AMD", "us", 0, today)


def test_recording_twice_the_same_day_does_not_fail(tmp_path):
    state = AlertState(tmp_path / "state.sqlite")
    today = date(2025, 3, 10)
    state.record(sample_result(), today)
    state.record(sample_result(), today)
    assert len(state.recent("us", days=30, today=today)) == 1
