"""Interfaz de proveedor de datos.

El motor programa **solo** contra estos protocolos. Cambiar de yfinance a EODHD
en la Fase 2 es cambiar `provider`/`price_provider` en config.yaml, no reescribir
nada aguas arriba.

Se separa en tres roles porque el config ya distingue quién sirve los precios de
quién sirve los fundamentales (en Fase 1 coinciden; en Fase 2 no tienen por qué).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from screener.models import Candidate, Region


@dataclass
class Statements:
    """Estados financieros de un ticker, tal como los consumen las métricas."""

    income_q: pd.DataFrame | None = None
    income_a: pd.DataFrame | None = None
    cashflow_q: pd.DataFrame | None = None
    cashflow_a: pd.DataFrame | None = None
    balance_q: pd.DataFrame | None = None
    balance_a: pd.DataFrame | None = None

    def is_empty(self) -> bool:
        return all(
            frame is None or frame.empty
            for frame in (self.income_q, self.income_a, self.cashflow_q,
                          self.cashflow_a, self.balance_q, self.balance_a)
        )


@dataclass
class Estimates:
    """Calendario de resultados, sorpresas y revisiones de estimaciones."""

    earnings_dates: pd.DataFrame | None = None
    eps_revisions: pd.DataFrame | None = None
    eps_trend: pd.DataFrame | None = None
    shares_full: pd.Series | None = None


@runtime_checkable
class PriceProvider(Protocol):
    def history(self, symbols: list[str], *, period: str) -> dict[str, pd.DataFrame]:
        """OHLCV diario por símbolo. Los que fallen simplemente no aparecen."""

    def close_series(self, symbol: str, *, period: str) -> pd.Series | None:
        """Serie de cierres de un único símbolo (benchmark, ETF sectorial)."""


@runtime_checkable
class FundamentalsProvider(Protocol):
    def profile(self, symbol: str) -> dict:
        """Sector, industria, market cap, enterprise value, divisa."""

    def statements(self, symbol: str) -> Statements: ...

    def estimates(self, symbol: str) -> Estimates: ...


@runtime_checkable
class UniverseProvider(Protocol):
    def candidates(self, region: Region) -> list[Candidate]:
        """Universo amplio de la región, ya etiquetado por sector."""


@dataclass
class DataProvider:
    """Los tres roles resueltos para una región concreta."""

    prices: PriceProvider
    fundamentals: FundamentalsProvider
    universe: UniverseProvider
    name: str = ""
    fx: dict[str, float] = field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


def build_provider(region: Region, cfg) -> DataProvider:
    """Fábrica: resuelve los nombres de proveedor del config a implementaciones."""
    from screener.data.yfinance_provider import YFinanceProvider

    names = {region.provider, region.price_provider}
    unsupported = names - {"yfinance"}
    if unsupported:
        raise ProviderError(
            f"región {region.key!r}: proveedor(es) no implementados en Fase 1: "
            f"{sorted(unsupported)}. EODHD llega en la Fase 2."
        )

    yahoo = YFinanceProvider(cfg)
    return DataProvider(prices=yahoo, fundamentals=yahoo, universe=yahoo, name="yfinance")
