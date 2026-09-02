"""Fundamentales de la SEC, perfil y precios de Yahoo.

Ninguna de las dos fuentes basta sola:

- La **SEC** llega a 2012 y trae la fecha real de presentación, pero no tiene
  precios, ni capitalización, ni sector en la taxonomía que usa el motor. Su
  `sic` es un código de cuatro cifras que no se corresponde con los sectores de
  Yahoo, y traducirlo a mano sería inventar una equivalencia discutible.
- **Yahoo** tiene precios, sector y capitalización, pero solo sirve 4-5
  ejercicios, lo que ancla cualquier backtest a partir de 2024.

Este proveedor los compone: perfil y precios de Yahoo, estados y calendario de
la SEC. El resultado es un backtest que puede empezar en 2013 con estados
fechados por presentación real en vez de aproximados.

**Lo que NO arregla**, y conviene no olvidarlo al leer los resultados:

- El **sesgo de supervivencia** sigue intacto. El universo lo construye Yahoo
  con lo que cotiza hoy, así que las quebradas y deslistadas no aparecen aunque
  la SEC tenga sus cuentas. Los retornos siguen sesgados al alza.
- La **capitalización es la de hoy**, no la de la fecha simulada. Es una
  limitación que ya tenía el backtest con Yahoo; no la introduce este módulo,
  pero tampoco la resuelve.
- **B3 y B8 se quedan sin datos**: sorpresa de resultados y revisiones de
  estimaciones vienen de analistas, no de los XBRL. `min_metric_coverage` las
  desactiva sola y renormaliza los pesos, igual que en el panel B reducido.
"""

from __future__ import annotations

import logging
from pathlib import Path

from screener.data.provider import Estimates, Statements
from screener.data.sec_provider import SecProvider

log = logging.getLogger("sec.hybrid")


class HybridFundamentals:
    """`FundamentalsProvider` que reparte cada rol a quien mejor lo sirve."""

    def __init__(self, yahoo, directorio_sec: Path | str = "./.cache/sec") -> None:
        self.yahoo = yahoo
        self.sec = SecProvider(directorio_sec)
        self._sin_cik: set[str] = set()

    def profile(self, symbol: str) -> dict:
        """De Yahoo: sector, capitalización y divisa.

        El `sic` de la SEC se añade como referencia, pero el sector que usa la
        normalización por percentil sigue siendo el de Yahoo para no romper la
        comparabilidad con el resto del proyecto.
        """
        perfil = dict(self.yahoo.profile(symbol))
        cik = self.sec.cik(symbol)
        if cik is not None:
            perfil.setdefault("cik", cik)
        return perfil

    def statements(self, symbol: str) -> Statements:
        """De la SEC, con Yahoo como red de seguridad.

        Un símbolo sin CIK —un ADR, un fondo, una extranjera— no está en los
        XBRL. En vez de dejarlo sin fundamentales, se cae a Yahoo: perderá
        profundidad histórica, pero seguirá puntuando.
        """
        estados = self.sec.statements(symbol)
        if estados.is_empty():
            if symbol not in self._sin_cik:
                self._sin_cik.add(symbol)
                log.debug("%s sin datos en la SEC, se usa Yahoo", symbol)
            return self.yahoo.statements(symbol)
        return estados

    def estimates(self, symbol: str) -> Estimates:
        """Calendario de la SEC; el resto de Yahoo si lo hay.

        Las fechas de la SEC son de presentación real, que es lo que
        `pointintime` necesita para decidir qué se sabía en cada momento. Las
        de Yahoo son de anuncio de resultados, unos días antes, y solo cubren
        los últimos años.

        `eps_revisions` y `eps_trend` se traen de Yahoo por si la corrida es
        actual, pero en un backtest `as_of` los anula igualmente: son una foto
        de hoy sin histórico, no una serie con vintages.
        """
        de_sec = self.sec.estimates(symbol)
        # Se mira el índice, no `.empty`: un DataFrame con fechas pero sin
        # columnas cuenta como vacío en pandas y nos devolvería a Yahoo.
        if de_sec.earnings_dates is None or not len(de_sec.earnings_dates.index):
            return self.yahoo.estimates(symbol)

        de_yahoo = self.yahoo.estimates(symbol)
        return Estimates(
            earnings_dates=de_sec.earnings_dates,
            eps_revisions=de_yahoo.eps_revisions,
            eps_trend=de_yahoo.eps_trend,
            shares_full=de_yahoo.shares_full,
        )
