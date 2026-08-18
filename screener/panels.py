"""Paneles: de valores crudos a `panel_raw` con desglose (§6).

El flujo por panel y región es:
  1. valor crudo de cada métrica (funciones puras)  -> DataFrame ticker x métrica
  2. percentil cross-sectional (universo o sector)  -> normalize.py
  3. suma ponderada de percentiles                  -> panel_raw
  4. percentil del propio panel_raw en la región    -> A_pct / B_pct  (engine)

El desglose no es opcional: cuando salte una alerta hay que poder ver *por qué*,
y para tunear los pesos hace falta la contribución métrica a métrica.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from screener.metrics import registry
from screener.models import MetricResult, Panel, PanelBreakdown, TickerData
from screener.normalize import NEUTRAL, percentile_by_group, percentile_rank

log = logging.getLogger(__name__)


@dataclass
class PanelInputs:
    """Valores crudos de un panel, con los pesos ya ajustados a la región."""

    panel: Panel
    raw: pd.DataFrame
    weights: dict[str, float]
    metric_coverage: dict[str, float]

    def ticker_coverage(self) -> pd.Series:
        """Fracción del peso del panel con dato real, ticker a ticker."""
        total = sum(self.weights.values())
        if not total or self.raw.empty:
            return pd.Series(0.0, index=self.raw.index, dtype="float64")
        covered = sum(
            self.raw[name].notna().astype(float) * weight for name, weight in self.weights.items()
        )
        return covered / total


def compute_raw_metrics(
    panel: Panel, tickers: dict[str, TickerData], weights: dict[str, float], cfg
) -> pd.DataFrame:
    """Matriz de valores crudos: filas = ticker, columnas = métrica."""
    rows: dict[str, dict[str, float | None]] = {}
    for symbol, data in tickers.items():
        rows[symbol] = {
            name: registry.compute(name, data, cfg.metric_params(name)) for name in weights
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(weights))


def drop_uncovered_metrics(
    raw: pd.DataFrame, weights: dict[str, float], min_coverage: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Desactiva las métricas sin datos suficientes en la región y renormaliza.

    Es la misma degradación que la spec prevé para la Fase 2, pero automática:
    si en una región no hay revisiones de analistas, esa métrica se cae sola en
    vez de repartir percentiles neutros a todo el mundo.
    """
    if raw.empty:
        return weights, {}

    coverage = {name: float(raw[name].notna().mean()) for name in weights}
    kept = {name: w for name, w in weights.items() if coverage[name] >= min_coverage}

    if not kept:
        log.warning("ninguna métrica supera la cobertura mínima; se conservan todas")
        return weights, coverage

    dropped = sorted(set(weights) - set(kept))
    if dropped:
        log.warning(
            "métricas desactivadas por cobertura < %.0f%%: %s",
            min_coverage * 100,
            ", ".join(f"{d} ({coverage[d] * 100:.0f}%)" for d in dropped),
        )

    total = sum(kept.values())
    return {name: w * 100.0 / total for name, w in kept.items()}, coverage


def to_percentiles(
    raw: pd.DataFrame,
    weights: dict[str, float],
    sectors: pd.Series,
    *,
    against: str,
    min_sector_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convierte cada columna cruda en percentil. Devuelve (percentiles, imputados)."""
    percentiles = pd.DataFrame(index=raw.index, columns=list(weights), dtype="float64")
    imputed = pd.DataFrame(index=raw.index, columns=list(weights), dtype="bool")

    for name in weights:
        spec = registry.get(name)
        column = raw[name]
        if against == "sector":
            ranked = percentile_by_group(
                column,
                sectors,
                higher_is_better=spec.higher_is_better,
                min_group_size=min_sector_size,
            )
        else:
            ranked = percentile_rank(column, higher_is_better=spec.higher_is_better)

        imputed[name] = column.isna()
        percentiles[name] = ranked.fillna(NEUTRAL)

    return percentiles, imputed


def build_breakdowns(
    panel: Panel,
    raw: pd.DataFrame,
    percentiles: pd.DataFrame,
    imputed: pd.DataFrame,
    weights: dict[str, float],
) -> dict[str, PanelBreakdown]:
    """`panel_raw` (suma ponderada de percentiles) + desglose, por ticker."""
    total_weight = sum(weights.values())
    breakdowns: dict[str, PanelBreakdown] = {}

    for symbol in raw.index:
        metrics: list[MetricResult] = []
        covered_weight = 0.0
        for name, weight in weights.items():
            was_imputed = bool(imputed.at[symbol, name])
            if not was_imputed:
                covered_weight += weight
            metrics.append(
                MetricResult(
                    name=name,
                    raw=None if was_imputed else float(raw.at[symbol, name]),
                    percentile=float(percentiles.at[symbol, name]),
                    weight=weight,
                    imputed=was_imputed,
                )
            )

        score = sum(m.contribution for m in metrics)
        breakdowns[symbol] = PanelBreakdown(
            panel=panel,
            raw_score=score,
            metrics=metrics,
            coverage=covered_weight / total_weight if total_weight else 0.0,
        )

    return breakdowns


def prepare_panel(
    panel: Panel, tickers: dict[str, TickerData], weights: dict[str, float], cfg
) -> PanelInputs:
    """Paso 1: valores crudos + desactivación de métricas sin cobertura regional."""
    raw = compute_raw_metrics(panel, tickers, weights, cfg)
    if raw.empty:
        return PanelInputs(panel, raw, weights, {})
    effective_weights, coverage = drop_uncovered_metrics(
        raw, weights, float(cfg.coverage["min_metric_coverage"])
    )
    return PanelInputs(panel, raw[list(effective_weights)], effective_weights, coverage)


def finalize_panel(inputs: PanelInputs, sectors: pd.Series, cfg) -> dict[str, PanelBreakdown]:
    """Paso 2: percentilar y componer el desglose.

    Se llama **después** de descartar los tickers con poca cobertura, para que
    un valor con datos basura no desplace el percentil de los demás.
    """
    if inputs.raw.empty:
        return {}
    percentiles, imputed = to_percentiles(
        inputs.raw,
        inputs.weights,
        sectors,
        against=cfg.normalize_against(inputs.panel),
        min_sector_size=int(cfg.universe["min_sector_size"]),
    )
    return build_breakdowns(inputs.panel, inputs.raw, percentiles, imputed, inputs.weights)


def score_panel(
    panel: Panel,
    tickers: dict[str, TickerData],
    weights: dict[str, float],
    sectors: pd.Series,
    cfg,
) -> tuple[dict[str, PanelBreakdown], dict[str, float]]:
    """Pipeline de un panel sin filtro de cobertura por ticker (atajo para tests)."""
    inputs = prepare_panel(panel, tickers, weights, cfg)
    return finalize_panel(inputs, sectors, cfg), inputs.metric_coverage
