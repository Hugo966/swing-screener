"""Registro de métricas.

Cada métrica se registra con **el mismo nombre que su clave de peso en
config.yaml**. Eso es lo que hace que `active_metrics` sea declarativo: filtrar
el registry y renormalizar los pesos supervivientes basta para el panel B
reducido de la Fase 2.

El sentido de la métrica se declara aquí, no dentro de la función:
`higher_is_better=False` (valoración, dilución, deuda, proximidad de evento) lo
invierte `normalize.py`. Así toda métrica devuelve su magnitud natural.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from screener.models import MetricSpec, Panel, TickerData

log = logging.getLogger(__name__)

MetricFunc = Callable[[TickerData, dict], float | None]

REGISTRY: dict[str, MetricSpec] = {}


def metric(
    name: str,
    *,
    panel: Panel,
    higher_is_better: bool = True,
    label: str = "",
) -> Callable[[MetricFunc], MetricFunc]:
    """Registra una función pura `(TickerData, params) -> float | None`."""

    def wrap(func: MetricFunc) -> MetricFunc:
        if name in REGISTRY:
            raise ValueError(f"métrica duplicada en el registry: {name}")
        REGISTRY[name] = MetricSpec(
            name=name,
            panel=panel,
            func=func,
            higher_is_better=higher_is_better,
            label=label or name,
            description=(func.__doc__ or "").strip().split("\n")[0],
        )
        return func

    return wrap


def get(name: str) -> MetricSpec:
    if name not in REGISTRY:
        raise KeyError(f"métrica desconocida: {name}")
    return REGISTRY[name]


def names_for_panel(panel: Panel) -> list[str]:
    return [n for n, s in REGISTRY.items() if s.panel == panel]


def compute(name: str, data: TickerData, params: dict) -> float | None:
    """Evalúa una métrica aislando sus fallos.

    Un dato ausente o corrupto de Yahoo no puede tumbar la corrida entera: se
    degrada a None y el engine lo imputa a percentil neutro descontando cobertura.
    """
    spec = get(name)
    try:
        value = spec.func(data, params)
    except Exception as exc:  # noqa: BLE001 — degradar, nunca propagar
        # Se registra: un bug en la métrica y un dato ausente se ven igual desde
        # fuera (cobertura 0), y sin esta traza no hay forma de distinguirlos.
        log.debug("métrica %s falló en %s: %s: %s", name, data.symbol, type(exc).__name__, exc)
        return None
    if value is None:
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return value
