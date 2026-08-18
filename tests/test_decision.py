"""Tabla de verdad de la decisión de alerta (§7) y renormalización de pesos."""

from __future__ import annotations

import pytest

from screener.config import ConfigError, resolve_weights
from screener.engine import decide
from screener.models import TickerResult


def result(a_pct, b_pct, regime=1.0, watchlist=False):
    r = TickerResult(symbol="TEST", region="us", a_pct=a_pct, b_pct=b_pct, regime=regime)
    r.is_watchlist = watchlist
    return r


# ---------------------------------------------------------------------------
# AND de dos umbrales altos
# ---------------------------------------------------------------------------
def test_both_panels_must_clear_their_threshold(cfg):
    """B decide *si* la empresa merece la pena, A decide *si ahora*."""
    good = result(95, 95)
    decide(good, cfg)
    assert good.alert

    # cohete de momentum con fundamentales de basura
    momentum_only = result(99, 40)
    decide(momentum_only, cfg)
    assert not momentum_only.alert
    assert "B_pct" in momentum_only.alert_reason

    # empresa excelente con timing pésimo
    quality_only = result(40, 99)
    decide(quality_only, cfg)
    assert not quality_only.alert
    assert "A_pct" in quality_only.alert_reason


def test_mediocre_but_uniform_never_alerts(cfg):
    """La media de los dos paneles daría 65: justo lo que el AND debe rechazar."""
    uniform = result(65, 65)
    decide(uniform, cfg)
    assert not uniform.alert


def test_final_cut_blocks_the_marginal_case(cfg):
    """Pasar los dos umbrales no basta: el score combinado tiene su propio corte."""
    a = float(cfg.alerting["a_threshold"])
    b = float(cfg.alerting["b_threshold"])
    marginal = result(a, b, regime=1.0)
    decide(marginal, cfg)

    combined = a * b / 100.0
    assert marginal.score_final == pytest.approx(combined)
    assert marginal.alert is (combined >= float(cfg.alerting["final_cut"]))


def test_bear_regime_shuts_the_alerts_down(cfg):
    """En bear real nada llega al corte: es la protección ante momentum crashes."""
    bull = result(90, 90, regime=1.0)
    bear = result(90, 90, regime=float(cfg.regime["min_multiplier"]))
    decide(bull, cfg)
    decide(bear, cfg)

    assert bull.alert
    assert bear.score_final < bull.score_final
    assert not bear.alert
    assert "corte" in bear.alert_reason


def test_watchlist_uses_its_own_looser_thresholds(cfg):
    """Nombres que Hugo ya vigila: mismo gate, umbral más laxo."""
    base_a = float(cfg.alerting["a_threshold"])
    watch_a = float(cfg.alerting["watchlist"]["a_threshold"])
    assert watch_a < base_a

    between = (watch_a + base_a) / 2

    regular = result(between, between, watchlist=False)
    watched = result(between, between, watchlist=True)
    decide(regular, cfg)
    decide(watched, cfg)

    assert not regular.alert
    assert watched.alert


# ---------------------------------------------------------------------------
# Renormalización de pesos (mecanismo del panel B reducido de Fase 2)
# ---------------------------------------------------------------------------
def test_disabling_metrics_rescales_the_rest_to_100(cfg):
    """Korea desactiva B3/B7/B8: los 71 pts restantes se reescalan a 100."""
    korea = cfg.region("korea").weights["quality"]
    base = cfg.raw["panels"]["quality"]["weights"]

    assert len(korea) == 7
    assert sum(korea.values()) == pytest.approx(100.0)
    for name in ("earnings_surprise_4q", "dilution_sbc", "estimate_revisions"):
        assert name not in korea

    # se conservan las proporciones relativas
    assert korea["revenue_growth_level"] == pytest.approx(base["revenue_growth_level"] * 100 / 71)
    assert korea["cash_quality_fcf_ni"] / korea["revenue_growth_level"] == pytest.approx(
        base["cash_quality_fcf_ni"] / base["revenue_growth_level"]
    )


def test_us_keeps_all_twenty_metrics(cfg):
    us = cfg.region("us")
    assert len(us.weights["momentum"]) == 10
    assert len(us.weights["quality"]) == 10
    assert sum(us.weights["quality"].values()) == pytest.approx(100.0)


def test_resolve_weights_rejects_unknown_metrics():
    with pytest.raises(ConfigError, match="sin peso definido"):
        resolve_weights({"a": 60.0, "b": 40.0}, ["a", "inventada"], context="test")


def test_yaml_booleans_in_region_codes_are_rejected(tmp_path, cfg):
    """`no` (Noruega) es el booleano falso en YAML 1.1.

    Sin esta comprobación se cuela como el literal "False", el screener devuelve
    cero resultados para esa bolsa y nada avisa de por qué falta Noruega.
    """
    import copy

    import yaml

    from screener.config import load_config

    raw = copy.deepcopy(cfg.raw)
    raw["regions"] = {"prueba": copy.deepcopy(cfg.raw["regions"]["europe_dev"])}
    raw["regions"]["prueba"]["yahoo_regions"] = ["de", False, "fr"]

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="Entrecomíllalos"):
        load_config(path, load_env=False)


def test_real_config_has_no_yaml_boolean_region_codes(cfg):
    for key, region in cfg.regions.items():
        for code in region.yahoo_regions:
            assert isinstance(code, str) and code not in ("True", "False"), (key, code)
