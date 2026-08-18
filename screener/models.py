"""Dataclasses del dominio.

`TickerData` es el contrato de entrada de toda métrica: se construye una vez por
ticker en el engine y las métricas solo leen de él. Ninguna métrica hace I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

Panel = str  # "momentum" | "quality"


# ---------------------------------------------------------------------------
# Universo
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """Un valor del universo antes de puntuar. Lo que devuelve el screener."""

    symbol: str
    name: str = ""
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None  # en la divisa de cotización
    market_cap_usd: float | None = None
    avg_volume: float | None = None  # acciones/día
    price: float | None = None
    currency: str = "USD"
    exchange: str = ""
    is_watchlist: bool = False

    @property
    def avg_dollar_volume(self) -> float | None:
        if self.avg_volume is None or self.price is None:
            return None
        return self.avg_volume * self.price


@dataclass
class GateResult:
    """Por qué un candidato pasó o no los gates. Se registra para diagnóstico."""

    symbol: str
    passed: bool
    failed_gate: str | None = None
    detail: str = ""


# ---------------------------------------------------------------------------
# Datos de entrada de las métricas
# ---------------------------------------------------------------------------
@dataclass
class TickerData:
    """Todo lo que una métrica puede necesitar de un ticker, ya descargado.

    Los DataFrames de estados financieros siguen la forma de yfinance: filas =
    conceptos contables, columnas = periodos (Timestamp), la más reciente primero.
    Cualquier campo puede ser None/vacío: las métricas devuelven None en ese caso
    y el engine lo trata como dato ausente (percentil neutro + descuento de
    cobertura), nunca como error.
    """

    symbol: str
    asof: date
    sector: str | None = None
    industry: str | None = None

    # Precios
    prices: pd.DataFrame | None = None  # OHLCV diario, índice DatetimeIndex asc
    benchmark: pd.Series | None = None  # cierre del benchmark de la región
    sector_index: pd.Series | None = None  # índice sectorial (sintético o ETF)

    # Perfil / valoración
    profile: dict[str, Any] = field(default_factory=dict)

    # Estados financieros
    income_q: pd.DataFrame | None = None
    income_a: pd.DataFrame | None = None
    cashflow_q: pd.DataFrame | None = None
    cashflow_a: pd.DataFrame | None = None
    balance_q: pd.DataFrame | None = None
    balance_a: pd.DataFrame | None = None

    # Estimaciones y calendario
    earnings_dates: pd.DataFrame | None = None  # índice: fecha; cols EPS/Surprise
    eps_revisions: pd.DataFrame | None = None
    eps_trend: pd.DataFrame | None = None
    shares_full: pd.Series | None = None

    def has_prices(self, minimum: int = 2) -> bool:
        return self.prices is not None and len(self.prices) >= minimum


# ---------------------------------------------------------------------------
# Métricas y paneles
# ---------------------------------------------------------------------------
@dataclass
class MetricSpec:
    """Metadatos de registro de una métrica pura."""

    name: str
    panel: Panel
    func: Any
    higher_is_better: bool = True
    label: str = ""
    description: str = ""


@dataclass
class MetricResult:
    name: str
    raw: float | None
    percentile: float = 50.0
    weight: float = 0.0
    imputed: bool = False  # True si raw era None y se imputó el percentil neutro

    @property
    def contribution(self) -> float:
        return self.percentile * self.weight / 100.0


@dataclass
class PanelBreakdown:
    panel: Panel
    raw_score: float = 0.0  # suma ponderada de percentiles, escala 0..100
    metrics: list[MetricResult] = field(default_factory=list)
    coverage: float = 0.0  # fracción del peso del panel con dato real

    def top(self, n: int = 5) -> list[MetricResult]:
        return sorted(self.metrics, key=lambda m: m.contribution, reverse=True)[:n]


@dataclass
class TickerResult:
    symbol: str
    region: str
    sector: str | None = None
    name: str = ""
    is_watchlist: bool = False
    price: float | None = None

    momentum: PanelBreakdown | None = None
    quality: PanelBreakdown | None = None

    a_pct: float = 0.0  # percentil del panel_raw de momentum dentro de la región
    b_pct: float = 0.0
    regime: float = 1.0
    score_final: float = 0.0

    alert: bool = False
    alert_reason: str = ""
    dropped: str | None = None  # motivo si se descartó por cobertura

    @property
    def combined(self) -> float:
        """A_pct * B_pct/100 * regime — el score del §7 de la spec."""
        return self.a_pct * self.b_pct / 100.0 * self.regime


# ---------------------------------------------------------------------------
# Región
# ---------------------------------------------------------------------------
@dataclass
class Region:
    """Una región es su universo, su benchmark, su pool de percentil, sus
    proveedores y su set de métricas activas.

    `weights` ya viene resuelto: filtrado por `active_metrics` y renormalizado a
    100. Eso es lo que permite el panel B reducido de Fase 2 sin tocar código.
    """

    key: str
    enabled: bool
    phase: int
    benchmark: str
    currency: str
    provider: str
    price_provider: str
    yahoo_regions: list[str] = field(default_factory=list)
    close_time_utc: str = "21:00"
    sector_index: str = "synthetic"
    weights: dict[Panel, dict[str, float]] = field(default_factory=dict)
    # Domicilios admitidos. Vacío = sin filtro. Sirve para echar los DR/BDR de
    # empresas extranjeras, que en emergentes copan la cabeza del universo.
    countries: list[str] = field(default_factory=list)

    def metric_names(self, panel: Panel) -> list[str]:
        return list(self.weights.get(panel, {}).keys())
