"""Preferencias que el panel escribe y el cron lee: envío y suelos.

Las dos comparten el mismo patrón —viven en `state.sqlite`, que es lo único
que comparten los dos procesos, y la ausencia de fila significa "sin tocar"—
así que se testean juntas.
"""

from __future__ import annotations

from datetime import date

import pytest

from screener.config import load_config
from screener.engine import RegionRun
from screener.models import TickerResult
from screener.runner import apply_floor_overrides, dispatch
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


# ---------------------------------------------------------------------------
# Suelos absolutos elegidos desde el panel
# ---------------------------------------------------------------------------
def test_sin_override_manda_config_yaml(state, cfg_tmp):
    configurados = dict(cfg_tmp.alerting["floors"])
    assert state.floor_overrides() == {}
    assert state.effective_floors(configurados) == configurados


def test_el_override_pisa_solo_el_suelo_indicado(state, cfg_tmp):
    configurados = dict(cfg_tmp.alerting["floors"])
    state.set_floor_overrides({"roic_vs_sector": 0.22})

    efectivos = state.effective_floors(configurados)
    assert efectivos["roic_vs_sector"] == 0.22
    for name, valor in configurados.items():
        if name != "roic_vs_sector":
            assert efectivos[name] == valor


def test_el_override_conserva_el_orden_de_config_yaml(state, cfg_tmp):
    """`decide` evalúa los suelos en orden y el primero que falla es el motivo.

    Un override que reordenara el dict cambiaría los mensajes de rechazo sin
    cambiar ninguna decisión, que es la clase de bug que nadie encuentra.
    """
    configurados = dict(cfg_tmp.alerting["floors"])
    state.set_floor_overrides({"rs_multi_window": 0.9, "revenue_growth_level": 0.9})

    assert list(state.effective_floors(configurados)) == list(configurados)


def test_vaciar_el_override_vuelve_a_config_yaml(state, cfg_tmp):
    configurados = dict(cfg_tmp.alerting["floors"])
    state.set_floor_overrides({"roic_vs_sector": 0.22})
    state.set_floor_overrides(None)

    assert state.floor_overrides() == {}
    assert state.effective_floors(configurados) == configurados


def test_se_reemplaza_el_juego_entero_no_metrica_a_metrica(state, cfg_tmp):
    """Un juego de suelos es una decisión conjunta.

    Dejar tres nuevos y uno viejo daría un cuarto juego que nadie eligió.
    """
    state.set_floor_overrides({"roic_vs_sector": 0.22, "revenue_growth_level": 0.4})
    state.set_floor_overrides({"roic_vs_sector": 0.19})

    assert state.floor_overrides() == {"roic_vs_sector": 0.19}


def test_una_metrica_inventada_se_rechaza(state):
    """Sería un suelo que no filtra y que nunca daría error.

    Los suelos de métricas que la región no calcula se ignoran a propósito, así
    que un nombre mal escrito pasaría desapercibido para siempre.
    """
    with pytest.raises(ValueError, match="no registradas"):
        state.set_floor_overrides({"revenue_growht_level": 0.25})  # typo a propósito


def test_un_suelo_fuera_de_config_yaml_no_se_cuela(state, cfg_tmp):
    """El override mueve un umbral; no inventa un suelo que el YAML no documenta."""
    configurados = dict(cfg_tmp.alerting["floors"])
    state.set_floor_overrides({"estimate_revisions": 9.0})

    assert state.effective_floors(configurados) == configurados


def test_el_runner_aplica_el_override_al_arrancar(cfg_tmp):
    """Sin esto el panel escribiría en una tabla que nadie lee.

    Se comprueba sobre el `cfg` que el runner usa después para puntuar, no sobre
    el valor devuelto: es `cfg.alerting["floors"]` lo que `decide` va a leer.
    """
    configurados = dict(cfg_tmp.alerting["floors"])
    AlertState(cfg_tmp.run["state_db"]).set_floor_overrides({"roic_vs_sector": 0.33})

    cambiados = apply_floor_overrides(cfg_tmp)

    assert cambiados == {"roic_vs_sector": 0.33}
    assert cfg_tmp.alerting["floors"]["roic_vs_sector"] == 0.33
    assert cfg_tmp.alerting["floors"]["revenue_growth_level"] == configurados["revenue_growth_level"]


def test_el_runner_no_toca_nada_si_no_hay_override(cfg_tmp):
    configurados = dict(cfg_tmp.alerting["floors"])
    AlertState(cfg_tmp.run["state_db"])  # crea la base, sin overrides

    assert apply_floor_overrides(cfg_tmp) == {}
    assert cfg_tmp.alerting["floors"] == configurados


def test_un_override_igual_a_config_no_cuenta_como_cambio(cfg_tmp):
    """Guardar los mismos valores no debe ensuciar el log de cada corrida."""
    AlertState(cfg_tmp.run["state_db"]).set_floor_overrides(dict(cfg_tmp.alerting["floors"]))

    assert apply_floor_overrides(cfg_tmp) == {}


def test_los_suelos_se_releen_del_disco_en_cada_consulta(tmp_path):
    """Sin caché por instancia, a diferencia de `delivery_map`.

    El panel guarda su `AlertState` en `st.cache_resource`, que vive lo que vive
    el proceso. Con caché, una escritura hecha desde otra instancia dejaba el
    aviso de "suelos override activos" mostrando algo falso — y ese aviso dice
    qué está usando Telegram, así que mentir ahí es lo peor que puede hacer esa
    pantalla. Se puede permitir releer: se consulta una vez por corrida y una o
    dos por render, no una vez por alerta.
    """
    ruta = tmp_path / "state.sqlite"
    panel = AlertState(ruta)
    cron = AlertState(ruta)

    assert panel.floor_overrides() == {}  # deja "cacheada" la respuesta vacía
    cron.set_floor_overrides({"roic_vs_sector": 0.22})

    assert panel.floor_overrides() == {"roic_vs_sector": 0.22}
