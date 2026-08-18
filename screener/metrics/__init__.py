"""Métricas puras. Importar este paquete registra las 20 métricas A1-A10/B1-B10."""

from screener.metrics import momentum, quality  # noqa: F401  (efecto: registro)
from screener.metrics.registry import REGISTRY, get, metric, names_for_panel

__all__ = ["REGISTRY", "get", "metric", "names_for_panel"]
