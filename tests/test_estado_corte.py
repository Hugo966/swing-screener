"""El recálculo vectorizado del gate debe coincidir con `decide()` fila a fila.

La pestaña "Estado del corte" no lee la columna `alert` de los CSV ni
`snapshots.passed`: recalcula el §7 desde los valores crudos para poder mover los
suelos y ver las entradas y salidas con ellos. Eso significa que hay dos
implementaciones de la misma regla, y si divergen la vista pinta de verde lo que
el motor no alertó. Una vista que miente es peor que no tenerla, así que aquí se
atan la una a la otra — incluido el texto del motivo, que es lo que se muestra en
la tabla y que un orden de evaluación distinto cambiaría sin cambiar el booleano.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from screener.engine import decide
from screener.metrics import registry
from screener.models import MetricResult, PanelBreakdown, TickerResult
from screener.panel import (
    EXIGENCIA_ANCLAS,
    FLOOR_KEYS,
    FLOOR_LIMITS,
    etiqueta_exigencia,
    metricas_activas,
    recompute_gate,
    suelos_de_exigencia,
    tramos,
)

ROOT = Path(__file__).resolve().parent.parent

JUEGOS_DE_SUELOS = [
    None,  # los de config.yaml
    {"revenue_growth_level": 0.05, "roic_vs_sector": 0.05,
     "cash_quality_fcf_ni": 0.30, "rs_multi_window": 0.10},
    dict.fromkeys(FLOOR_KEYS, 0.0),  # el extremo laxo: pasa quien tenga dato
    dict.fromkeys(FLOOR_KEYS, 5.0),  # el extremo severo: no pasa nadie
]


def referencia(frame: pd.DataFrame, region: str, cfg) -> pd.DataFrame:
    """Lo que decide el motor, fila a fila, reconstruyendo el `TickerResult`.

    Es la implementación lenta a propósito: `decide` es la verdad, y este test
    existe para atarle la versión vectorizada.
    """
    metricas = [c[: -len("__raw")] for c in frame.columns if c.endswith("__raw")]
    alertas, motivos = [], []
    for _, fila in frame.iterrows():
        resultado = TickerResult(
            symbol=str(fila["symbol"]),
            region=region,
            a_pct=float(fila["a_pct"]),
            b_pct=float(fila["b_pct"]),
            regime=float(fila["regime"]),
        )
        resultado.is_watchlist = bool(fila.get("watchlist", False))
        por_panel: dict[str, list[MetricResult]] = {"momentum": [], "quality": []}
        for nombre in metricas:
            valor = fila[f"{nombre}__raw"]
            por_panel[registry.get(nombre).panel].append(
                MetricResult(name=nombre, raw=None if pd.isna(valor) else float(valor))
            )
        resultado.momentum = PanelBreakdown(panel="momentum", metrics=por_panel["momentum"])
        resultado.quality = PanelBreakdown(panel="quality", metrics=por_panel["quality"])
        decide(resultado, cfg)
        alertas.append(resultado.alert)
        motivos.append(resultado.alert_reason)
    return pd.DataFrame({"alert": alertas, "reason": motivos}, index=frame.index)


def comparar(frame: pd.DataFrame, region: str, cfg, floors) -> None:
    # `decide` lee los suelos del cfg y `recompute_gate` los recibe por
    # argumento: para comparar hay que darles el mismo juego. La copia profunda
    # es obligatoria porque el `cfg` del fixture es de sesión y mutarlo
    # contaminaría los demás tests.
    local = copy.deepcopy(cfg)
    if floors is not None:
        local.raw["alerting"]["floors"] = dict(floors)
    usados = local.alerting["floors"]

    esperado = referencia(frame, region, local)
    obtenido = recompute_gate(
        frame,
        active=metricas_activas(local, region),
        floors=usados,
        a_threshold=float(local.alerting["a_threshold"]),
        b_threshold=float(local.alerting["b_threshold"]),
        final_cut=float(local.alerting["final_cut"]),
        require_data=bool(local.alerting.get("floors_require_data", True)),
    )

    distinta = frame.loc[obtenido["pasa"].to_numpy() != esperado["alert"].to_numpy(), "symbol"]
    assert distinta.empty, f"decisión distinta en {list(distinta)[:10]}"
    distinta = frame.loc[obtenido["motivo"].to_numpy() != esperado["reason"].to_numpy(), "symbol"]
    assert distinta.empty, f"motivo distinto en {list(distinta)[:10]}"


# ---------------------------------------------------------------------------
# Casos frontera, sin depender de out/
# ---------------------------------------------------------------------------
def frame_sintetico() -> pd.DataFrame:
    """Una fila por cada caso que el gate distingue."""
    filas = [
        # holgado en todo: pasa
        dict(symbol="PASA", a_pct=90, b_pct=90, regime=1.0,
             revenue_growth_level__raw=0.40, roic_vs_sector__raw=0.25,
             cash_quality_fcf_ni__raw=1.10, rs_multi_window__raw=0.80),
        # exactamente en cada suelo y en cada umbral: el borde entra
        dict(symbol="JUSTO", a_pct=50, b_pct=50, regime=1.0,
             revenue_growth_level__raw=0.25, roic_vs_sector__raw=0.18,
             cash_quality_fcf_ni__raw=0.90, rs_multi_window__raw=0.30),
        # un pelo por debajo de un único suelo
        dict(symbol="ROZA", a_pct=99, b_pct=99, regime=1.0,
             revenue_growth_level__raw=0.2499, roic_vs_sector__raw=0.25,
             cash_quality_fcf_ni__raw=1.10, rs_multi_window__raw=0.80),
        # métrica activa sin dato: falla con floors_require_data
        dict(symbol="SINDATO", a_pct=99, b_pct=99, regime=1.0,
             revenue_growth_level__raw=0.40, roic_vs_sector__raw=0.25,
             cash_quality_fcf_ni__raw=np.nan, rs_multi_window__raw=0.80),
        # dos suelos fallando: el motivo lo fija el primero del dict
        dict(symbol="DOSFALLOS", a_pct=99, b_pct=99, regime=1.0,
             revenue_growth_level__raw=0.01, roic_vs_sector__raw=-0.30,
             cash_quality_fcf_ni__raw=1.10, rs_multi_window__raw=0.80),
        # suelos OK pero percentil corto, por cada lado
        dict(symbol="ACORTO", a_pct=49, b_pct=99, regime=1.0,
             revenue_growth_level__raw=0.40, roic_vs_sector__raw=0.25,
             cash_quality_fcf_ni__raw=1.10, rs_multi_window__raw=0.80),
        dict(symbol="BCORTO", a_pct=99, b_pct=49, regime=1.0,
             revenue_growth_level__raw=0.40, roic_vs_sector__raw=0.25,
             cash_quality_fcf_ni__raw=1.10, rs_multi_window__raw=0.80),
        # percentiles OK pero el régimen hunde el score bajo el corte
        dict(symbol="REGIMEN", a_pct=55, b_pct=55, regime=0.50,
             revenue_growth_level__raw=0.40, roic_vs_sector__raw=0.25,
             cash_quality_fcf_ni__raw=1.10, rs_multi_window__raw=0.80),
        # ROIC negativo con momentum de p99: el caso que motivó los suelos
        dict(symbol="ROICNEG", a_pct=99, b_pct=95, regime=1.0,
             revenue_growth_level__raw=0.40, roic_vs_sector__raw=-0.305,
             cash_quality_fcf_ni__raw=1.10, rs_multi_window__raw=0.80),
    ]
    frame = pd.DataFrame(filas)
    frame["watchlist"] = False
    frame["score_final"] = frame["a_pct"] * frame["b_pct"] / 100 * frame["regime"]
    return frame


@pytest.mark.parametrize("floors", JUEGOS_DE_SUELOS)
def test_el_recalculo_replica_decide_en_los_casos_frontera(cfg, floors):
    comparar(frame_sintetico(), "us", cfg, floors)


def test_una_columna_de_suelo_ausente_veta_igual_que_un_nan(cfg):
    """`drop_uncovered_metrics` puede tirar una métrica en una corrida concreta.

    Entonces el CSV no trae la columna, pero la métrica sigue activa en la
    región: `decide` no la encuentra entre las medidas, la trata como sin dato y
    el suelo falla. Es el caso que rompería una unión de CSV descuidada.
    """
    comparar(frame_sintetico().drop(columns=["cash_quality_fcf_ni__raw"]), "us", cfg, None)


def test_un_suelo_sobre_metrica_inactiva_en_la_region_se_ignora(cfg):
    """Korea tiene el panel B reducido: un suelo ahí no debe silenciar la región."""
    assert "estimate_revisions" not in metricas_activas(cfg, "korea")

    frame = frame_sintetico()
    floors = {**cfg.alerting["floors"], "estimate_revisions": 9.0}
    comparar(frame, "korea", cfg, floors)

    # y el mismo suelo en una región que sí la calcula veta a todo el mundo
    assert "estimate_revisions" in metricas_activas(cfg, "us")


def test_un_frame_vacio_no_revienta(cfg):
    vacio = frame_sintetico().iloc[:0]
    salida = recompute_gate(
        vacio, active=metricas_activas(cfg, "us"), floors=cfg.alerting["floors"],
        a_threshold=50.0, b_threshold=50.0, final_cut=30.0, require_data=True,
    )
    assert salida.empty
    assert {"pasa", "score_calc", "motivo"} <= set(salida.columns)


# ---------------------------------------------------------------------------
# Datos reales, si están en disco
# ---------------------------------------------------------------------------
def csvs_reales() -> list[Path]:
    return sorted((ROOT / "out").glob("*_20*.csv"))


@pytest.mark.skipif(not csvs_reales(), reason="no hay CSV de corridas en out/")
@pytest.mark.parametrize("floors", JUEGOS_DE_SUELOS)
def test_el_recalculo_replica_decide_sobre_las_corridas_reales(cfg, floors):
    """Fila a fila sobre cada CSV de cada región.

    Los CSV en disco pueden ser de una versión anterior del config —los de agosto
    llevan `A_pct 76 < 80` en su `reason`—, así que NO se compara contra su
    columna `alert`: se compara contra `decide` con el config de hoy. Esa
    diferencia es justamente el motivo de que la pestaña recalcule.
    """
    for ruta in csvs_reales():
        region, _, _day = ruta.stem.rpartition("_")
        if region not in cfg.regions:
            continue
        comparar(pd.read_csv(ruta), region, cfg, floors)


# ---------------------------------------------------------------------------
# Barra maestra de exigencia
# ---------------------------------------------------------------------------
def test_las_anclas_devuelven_exactamente_sus_valores_medidos():
    """Son juegos medidos, no interpolados: la barra no puede desviarlos."""
    for nivel, _nombre, esperados in EXIGENCIA_ANCLAS:
        assert suelos_de_exigencia(nivel) == esperados


def test_el_ancla_de_produccion_coincide_con_config_yaml(cfg):
    """Si alguien cambia los suelos del YAML, la senda se queda descolgada.

    Este test es el aviso: la tercera ancla debe seguir siendo lo que corre, o la
    barra maestra estaría ofreciendo un "nivel de producción" que ya no lo es.
    """
    _nivel, nombre, suelos = EXIGENCIA_ANCLAS[2]
    assert nombre == "exquisito"
    assert suelos == {name: float(v) for name, v in cfg.alerting["floors"].items()}


def test_la_exigencia_es_monotona_y_cae_en_la_rejilla_de_los_sliders():
    anterior = suelos_de_exigencia(0)
    for nivel in range(0, 101):
        actual = suelos_de_exigencia(nivel)
        for name in FLOOR_KEYS:
            minimo, maximo, paso = FLOOR_LIMITS[name]
            assert actual[name] >= anterior[name] - 1e-9, f"{name} baja en {nivel}"
            assert minimo <= actual[name] <= maximo
            # que encaje en el paso del widget, o Streamlit recibe un valor
            # que su propia rejilla no puede representar
            assert abs(actual[name] / paso - round(actual[name] / paso)) < 1e-6
        anterior = actual


def test_fuera_de_rango_se_recorta_a_las_anclas():
    assert suelos_de_exigencia(-50) == EXIGENCIA_ANCLAS[0][2]
    assert suelos_de_exigencia(500) == EXIGENCIA_ANCLAS[-1][2]


def test_la_etiqueta_nombra_el_ancla_o_el_tramo():
    assert etiqueta_exigencia(0) == "laxo"
    assert etiqueta_exigencia(67) == "exquisito"
    assert etiqueta_exigencia(50) == "entre medio y exquisito"


# ---------------------------------------------------------------------------
# Tramos de entrada y salida
# ---------------------------------------------------------------------------
def matriz(filas: dict[str, list[bool]], fechas: list[str]) -> pd.DataFrame:
    return pd.DataFrame(filas, index=fechas).T.astype(bool)


def test_los_tramos_leen_bien_los_casos_de_una_sola_observacion():
    info = tramos(matriz({"DENTRO": [True], "FUERA": [False]}, ["2026-09-09"]))

    assert info.loc["DENTRO", "dentro"]
    assert info.loc["DENTRO", "desde_el_inicio"]  # no se puede saber cuándo entró
    assert info.loc["DENTRO", "corridas_dentro"] == 1
    assert not info.loc["FUERA", "visto"]
    assert info.loc["FUERA", "salio_el"] is None


def test_los_tramos_distinguen_dentro_fuera_y_nunca():
    fechas = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]
    info = tramos(matriz({
        "SIEMPRE": [True, True, True, True],
        "ENTRO":   [False, False, True, True],
        "SALIO":   [True, True, False, False],
        "NUNCA":   [False, False, False, False],
        "REENTRO": [True, False, False, True],
    }, fechas))

    assert info.loc["SIEMPRE", "desde_el_inicio"]
    assert info.loc["SIEMPRE", "corridas_dentro"] == 4

    assert info.loc["ENTRO", "dentro"]
    assert not info.loc["ENTRO", "desde_el_inicio"]
    assert info.loc["ENTRO", "desde"] == "2026-09-03"
    assert info.loc["ENTRO", "corridas_dentro"] == 2

    assert not info.loc["SALIO", "dentro"]
    assert info.loc["SALIO", "visto"]
    assert info.loc["SALIO", "ultima_dentro"] == "2026-09-02"
    assert info.loc["SALIO", "salio_el"] == "2026-09-03"
    assert info.loc["SALIO", "corridas_fuera"] == 2

    assert not info.loc["NUNCA", "visto"]
    assert info.loc["NUNCA", "corridas_dentro"] == 0

    # Una reentrada cuenta solo el tramo actual: es lo que se quiere saber.
    assert info.loc["REENTRO", "corridas_dentro"] == 1
    assert info.loc["REENTRO", "desde"] == "2026-09-04"


@pytest.mark.parametrize("n_fechas", [1, 2, 3, 5, 13])
def test_los_tramos_coinciden_con_la_fuerza_bruta(n_fechas):
    """`tramos` usa dos argmax sobre la matriz del revés en vez de un bucle.

    Es la parte con más aritmética de índices del módulo, así que se compara
    contra la versión obvia y lenta sobre matrices aleatorias, incluidas las
    filas todo-True y todo-False que son los casos que rompen un `argmax`.
    """
    rng = np.random.default_rng(20260910 + n_fechas)
    fechas = [f"2026-09-{d:02d}" for d in range(1, n_fechas + 1)]
    datos = rng.random((200, n_fechas)) > 0.6
    datos[0, :] = True
    datos[1, :] = False
    frame = pd.DataFrame(datos, index=[f"T{i}" for i in range(len(datos))], columns=fechas)

    info = tramos(frame)
    for simbolo, fila in frame.iterrows():
        serie = list(fila)
        dentro = serie[-1]
        visto = any(serie)
        assert bool(info.loc[simbolo, "dentro"]) == dentro
        assert bool(info.loc[simbolo, "visto"]) == visto

        # tramo final de True contado a mano
        racha = 0
        for valor in reversed(serie):
            if not valor:
                break
            racha += 1
        assert info.loc[simbolo, "corridas_dentro"] == racha

        if dentro:
            assert info.loc[simbolo, "desde"] == fechas[len(serie) - racha]
            assert bool(info.loc[simbolo, "desde_el_inicio"]) == (racha == len(serie))
        elif visto:
            ultimo = max(i for i, valor in enumerate(serie) if valor)
            assert info.loc[simbolo, "ultima_dentro"] == fechas[ultimo]
            assert info.loc[simbolo, "salio_el"] == fechas[ultimo + 1]
            assert info.loc[simbolo, "corridas_fuera"] == len(serie) - 1 - ultimo
