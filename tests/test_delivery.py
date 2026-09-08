"""Interruptores de envío a Telegram, por región y por sector."""

from __future__ import annotations

from datetime import date

import pytest

from screener.config import load_config
from screener.engine import RegionRun
from screener.models import TickerResult
from screener.runner import dispatch
from screener.state import AlertState

TODAY = date(2025, 3, 10)


@pytest.fixture
def state(tmp_path):
    return AlertState(tmp_path / "state.sqlite")


def alerting(symbol, region="us", sector="Technology", score=70.0):
    r = TickerResult(symbol=symbol, region=region, sector=sector,
                     a_pct=90.0, b_pct=90.0, regime=1.0)
    r.score_final = score
    r.alert = True
    return r


# ---------------------------------------------------------------------------
# Contrato del estado
# ---------------------------------------------------------------------------
def test_everything_is_delivered_by_default(state):
    """La ausencia de fila es "activado": una instalación existente no cambia."""
    assert state.should_deliver("us", "Technology")
    assert state.delivery_map("region") == {}


def test_a_disabled_region_stops_delivering(state):
    state.set_delivery("region", "korea", False)

    assert not state.should_deliver("korea", "Technology")
    assert state.should_deliver("us", "Technology")


def test_a_disabled_sector_applies_across_regions(state):
    """El filtro de sector es global: apagar Energy lo apaga en todas."""
    state.set_delivery("sector", "Energy", False)

    assert not state.should_deliver("us", "Energy")
    assert not state.should_deliver("emerging", "Energy")
    assert state.should_deliver("us", "Technology")


def test_the_two_filters_are_an_and(state):
    state.set_delivery("region", "us", False)
    state.set_delivery("sector", "Energy", False)

    assert not state.should_deliver("us", "Technology")   # región apagada
    assert not state.should_deliver("korea", "Energy")    # sector apagado
    assert state.should_deliver("korea", "Technology")    # ninguno


def test_a_name_without_sector_is_delivered(state):
    """Sin sector no se puede filtrar, y silenciarlo esconderá justo lo raro."""
    state.set_delivery("sector", "Energy", False)
    assert state.should_deliver("us", None)


def test_re_enabling_restores_delivery_and_invalidates_the_cache(state):
    """El mapa se cachea por instancia: reactivar tiene que verse ya."""
    state.set_delivery("region", "korea", False)
    assert not state.should_deliver("korea", "Technology")

    state.set_delivery("region", "korea", True)
    assert state.should_deliver("korea", "Technology")


# ---------------------------------------------------------------------------
# Integración con el envío
# ---------------------------------------------------------------------------
@pytest.fixture
def cfg_tmp(tmp_path):
    """Config real apuntando a una base de estado desechable."""
    cfg = load_config(load_env=False)
    cfg.raw["run"] = dict(cfg.raw["run"], state_db=str(tmp_path / "state.sqlite"))
    return cfg


def test_a_disabled_region_does_not_burn_the_cooldown(cfg_tmp):
    """Apagado significa "no me lo cuentes", no "dalo por contado".

    Si el filtro fuese después de clasificar, apagar una región le consumiría el
    cooldown a cada nombre y al reactivarla no llegaría nada hasta que mejorara.

    Va con `dry_run=False` a propósito: en dry-run no se anota nada de todas
    formas, así que el test no probaría el filtro. Sin credenciales en el
    entorno, `build_notifier` devuelve None y `dispatch` imprime en vez de
    enviar, pero sí anota — que es justo lo que aquí no debe pasar.
    """
    state = AlertState(cfg_tmp.run["state_db"])
    state.set_delivery("region", "us", False)

    run = RegionRun(region="us", asof=TODAY, results=[alerting("AAA")])
    assert dispatch(run, cfg_tmp, dry_run=False, send_summary=False) == 0

    # ni enviada ni anotada: la tabla de alertas sigue vacía
    assert state.last_alert("AAA", "us", date.min) is None
    # pero el histórico se guarda entero, que es lo que ve el panel
    assert len(state.history("us")) == 1


def test_an_enabled_region_does_record_the_alert(cfg_tmp):
    """El contraste del test anterior: sin apagar, sí se anota."""
    state = AlertState(cfg_tmp.run["state_db"])
    run = RegionRun(region="us", asof=TODAY, results=[alerting("AAA")])

    assert dispatch(run, cfg_tmp, dry_run=False, send_summary=False) == 1
    assert state.last_alert("AAA", "us", date.min) is not None


def test_an_enabled_region_still_delivers(cfg_tmp):
    run = RegionRun(region="us", asof=TODAY, results=[alerting("AAA")])
    assert dispatch(run, cfg_tmp, dry_run=True, send_summary=False) == 1


def test_a_disabled_sector_is_skipped_but_its_peers_are_not(cfg_tmp):
    state = AlertState(cfg_tmp.run["state_db"])
    state.set_delivery("sector", "Energy", False)

    run = RegionRun(region="us", asof=TODAY, results=[
        alerting("OIL", sector="Energy"),
        alerting("CHIP", sector="Technology"),
    ])
    assert dispatch(run, cfg_tmp, dry_run=True, send_summary=False) == 1
