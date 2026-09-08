"""Tabla de verdad de la decisión de alerta (§7) y renormalización de pesos."""

from __future__ import annotations

import pytest

from screener.config import ConfigError, resolve_weights
from screener.engine import _active_metrics, _floor_failure, decide
from screener.metrics import registry
from screener.models import MetricResult, PanelBreakdown, TickerResult

# Valores crudos que pasan holgadamente todos los suelos de config.yaml. Los
# suelos son ahora el criterio principal, así que un helper que no midiera nada
# haría fallar cada caso por falta de dato en vez de por lo que se quiere probar.
HOLGADO = {
    "revenue_growth_level": 0.40,
    "roic_vs_sector": 0.25,
    "cash_quality_fcf_ni": 1.10,
    "rs_multi_window": 0.80,
}


def result(a_pct, b_pct, regime=1.0, watchlist=False, *, raws=None, panels=True):
    """TickerResult con los suelos satisfechos salvo que `raws` diga otra cosa.

    `raws=None` deja los cuatro suelos holgados; un dict los sobreescribe (con
    None para simular falta de dato). `panels=False` deja los paneles sin
    construir, que es el caso degenerado que los suelos deben rechazar.
    """
    r = TickerResult(symbol="TEST", region="us", a_pct=a_pct, b_pct=b_pct, regime=regime)
    r.is_watchlist = watchlist
    if not panels:
        return r

    values = dict(HOLGADO)
    values.update(raws or {})
    by_panel: dict[str, list[MetricResult]] = {"momentum": [], "quality": []}
    for name, raw in values.items():
        by_panel[registry.get(name).panel].append(MetricResult(name=name, raw=raw))
    r.momentum = PanelBreakdown(panel="momentum", metrics=by_panel["momentum"])
    r.quality = PanelBreakdown(panel="quality", metrics=by_panel["quality"])
    return r


# ---------------------------------------------------------------------------
# Suelos absolutos: el criterio de "¿es buena?"
# ---------------------------------------------------------------------------
def test_a_floor_is_not_compensable_by_a_great_percentile(cfg):
    """Lo que motivó el cambio: un p99 de momentum tapaba un ROIC negativo.

    El 12-ago entraron cuatro nombres con ROIC negativo (el peor, -30,5%) porque
    la suma ponderada los compensaba. Un suelo no se compensa.
    """
    excelente_pero_destruye_valor = result(99, 95, raws={"roic_vs_sector": -0.305})
    decide(excelente_pero_destruye_valor, cfg)

    assert not excelente_pero_destruye_valor.alert
    assert "B5 ROIC" in excelente_pero_destruye_valor.alert_reason
    assert "suelo" in excelente_pero_destruye_valor.alert_reason


def test_decent_percentiles_with_good_fundamentals_do_alert(cfg):
    """El otro lado del cambio: dejar de ser una cuota del 20%.

    Con los umbrales viejos (80/80) esto no alertaba. Una empresa que crece,
    renta por encima del coste de capital y convierte a caja es una buena
    oportunidad aunque no esté en el decil de nada.
    """
    buena_sin_ser_top = result(62, 60)
    decide(buena_sin_ser_top, cfg)

    assert buena_sin_ser_top.alert


def test_every_floor_can_veto_on_its_own(cfg):
    """Cada suelo veta por separado: no hay media entre ellos."""
    for name, malo in (
        ("revenue_growth_level", -0.023),   # el que entró el 12-ago con -2,3%
        ("roic_vs_sector", 0.01),
        ("cash_quality_fcf_ni", 0.10),
        ("rs_multi_window", 0.05),
    ):
        r = result(95, 95, raws={name: malo})
        decide(r, cfg)
        assert not r.alert, f"{name} no vetó"
        assert registry.get(name).label in r.alert_reason


def test_missing_data_fails_the_floor(cfg):
    """"Buena" exige evidencia, no ausencia de pruebas."""
    sin_dato = result(95, 95, raws={"revenue_growth_level": None})
    decide(sin_dato, cfg)

    assert not sin_dato.alert
    assert "sin dato" in sin_dato.alert_reason


def test_a_ticker_without_panels_cannot_pass_by_emptiness(cfg):
    """Sin métricas medidas no hay nada que compare: no puede pasar por vacío."""
    degenerado = result(99, 99, panels=False)
    decide(degenerado, cfg)

    assert not degenerado.alert


def test_floor_on_a_metric_the_region_does_not_compute_is_ignored(cfg):
    """Korea no calcula B3/B7/B8: un suelo ahí no debe silenciar la región.

    La ausencia por configuración de región y la ausencia por falta de dato se
    parecen, y confundirlas dejaría regiones enteras sin alertas.
    """
    assert "estimate_revisions" not in cfg.region("korea").weights["quality"]

    r = result(95, 95)
    r.region = "korea"

    # un suelo imposible sobre la métrica que korea no calcula, más uno que sí
    floors = {"estimate_revisions": 9.0, "revenue_growth_level": 0.15}
    active = _active_metrics(cfg, "korea")

    assert _floor_failure(r, floors, True, active) is None
    # y en una región que sí la calcula, ese mismo suelo veta
    assert _floor_failure(r, floors, True, _active_metrics(cfg, "us")) is not None


# ---------------------------------------------------------------------------
# Percentiles: la red de equilibrio
# ---------------------------------------------------------------------------
def test_both_panels_must_clear_their_threshold(cfg):
    """B decide *si* la empresa merece la pena, A decide *si ahora*."""
    good = result(95, 95)
    decide(good, cfg)
    assert good.alert

    # cohete de momentum con fundamentales de basura
    momentum_only = result(99, 20)
    decide(momentum_only, cfg)
    assert not momentum_only.alert
    assert "B_pct" in momentum_only.alert_reason

    # empresa excelente con timing pésimo
    quality_only = result(20, 99)
    decide(quality_only, cfg)
    assert not quality_only.alert
    assert "A_pct" in quality_only.alert_reason


def test_the_net_is_a_net_and_not_the_criterion(cfg):
    """Los percentiles ya no deciden: solo vetan lo desequilibrado de verdad.

    65/65 se rechazaba con los umbrales viejos por mediocre. Ahora lo que decide
    es el crudo, y ese mismo 65/65 con un suelo roto sigue fuera.
    """
    uniforme_y_buena = result(65, 65)
    uniforme_y_mala = result(65, 65, raws={"cash_quality_fcf_ni": 0.10})
    decide(uniforme_y_buena, cfg)
    decide(uniforme_y_mala, cfg)

    assert uniforme_y_buena.alert
    assert not uniforme_y_mala.alert


def test_final_cut_blocks_the_marginal_case(cfg):
    """Pasar los dos umbrales no basta: el score combinado tiene su propio corte."""
    a = float(cfg.alerting["a_threshold"])
    b = float(cfg.alerting["b_threshold"])
    marginal = result(a, b, regime=1.0)
    decide(marginal, cfg)

    combined = a * b / 100.0
    assert marginal.score_final == pytest.approx(combined)
    assert marginal.alert is (combined >= float(cfg.alerting["final_cut"]))


def test_worse_regime_demands_higher_percentiles(cfg):
    """El freno de mercado es continuo, no un escalón.

    `final_cut` va sobre `A_pct * B_pct/100 * regime`, así que cuanto peor está
    el mercado más percentil hace falta para el mismo corte. El régimen bajó de
    0,86 en 55 de los 155 meses del panel largo (mínimo 0,618 en 08-2022), así
    que este freno trabaja de verdad.

    Ojo: con los umbrales bajos, un nombre excelente **sí** puede alertar en
    régimen mínimo. La protección ante momentum crashes descansa ahora en el
    gate de tendencia (precio > MM200 y MM50 > MM200) y en el suelo de
    `rs_multi_window`, que son por valor y absolutos, no en este corte.
    """
    cut = float(cfg.alerting["final_cut"])
    floor_regime = float(cfg.regime["min_multiplier"])

    bull = result(60, 60, regime=1.0)
    bear = result(60, 60, regime=floor_regime)
    decide(bull, cfg)
    decide(bear, cfg)

    assert bull.alert
    assert bear.score_final < bull.score_final
    assert not bear.alert
    assert "corte" in bear.alert_reason

    # el percentil que el mismo corte exige sube al empeorar el régimen
    assert cut / 1.0 < cut / floor_regime


def test_watchlist_uses_its_own_looser_thresholds(cfg):
    """Nombres que Hugo ya vigila: mismo gate, umbral y suelos más laxos."""
    base_a = float(cfg.alerting["a_threshold"])
    watch = cfg.alerting["watchlist"]
    watch_a = float(watch["a_threshold"])
    assert watch_a < base_a

    between = (watch_a + base_a) / 2
    assert between * between / 100.0 >= float(watch["final_cut"]), (
        "el final_cut de watchlist anularía sus propios umbrales"
    )

    regular = result(between, between, watchlist=False)
    watched = result(between, between, watchlist=True)
    decide(regular, cfg)
    decide(watched, cfg)

    assert not regular.alert
    assert watched.alert


def test_watchlist_floors_are_looser_but_present(cfg):
    """Más laxos, no ausentes: un nombre vigilado que destruye valor sigue fuera."""
    watch_floors = cfg.alerting["watchlist"]["floors"]
    base_floors = cfg.alerting["floors"]
    for name, value in watch_floors.items():
        assert value < base_floors[name]

    r = result(90, 90, watchlist=True, raws={"roic_vs_sector": -0.30})
    decide(r, cfg)
    assert not r.alert


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
