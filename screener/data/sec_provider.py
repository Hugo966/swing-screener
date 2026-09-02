"""Proveedor de fundamentales sobre los datos de la SEC.

Implementa la parte de `FundamentalsProvider` que se puede servir desde los
XBRL: estados financieros y fecha real de presentación. Lo que no está en la
SEC —estimaciones de analistas, sorpresa de resultados, revisiones— se devuelve
vacío, y `coverage.min_metric_coverage` desactiva esas métricas solo, que es el
mismo mecanismo del panel B reducido de la Fase 2.

**Por qué existe.** Yahoo sirve 4-5 ejercicios y ~5 trimestres, lo que ancla
cualquier backtest a partir de 2024. La SEC llega a 2012 con cobertura completa
y, sobre todo, trae la fecha en que cada dato se hizo público.

**La primera publicación manda.** Un 10-K de 2025 incluye cifras comparativas de
2023 y 2022, así que el mismo periodo reaparece en muchos trimestres. Para
reconstruir el pasado hay que quedarse con la **primera vez** que un dato se
publicó, no con la última: si hubo reexpresión, el mercado de entonces vio la
cifra original. Es el mismo criterio del sidecar `.seen.json` del caché de Yahoo.

**Sobre la fecha.** `filed` es cuando se presentó el informe, que llega unos días
después de la nota de prensa de resultados. Así que este proveedor es algo
conservador: supone que la información se conoció un poco más tarde de lo real.
En un backtest, equivocarse hacia tarde es el lado correcto.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from screener.data.provider import Estimates, Statements

log = logging.getLogger("sec.provider")

# Filas de balance: saldos puntuales, no flujos de un periodo.
BALANCE = {
    "Total Assets", "Stockholders Equity", "Cash And Cash Equivalents",
    "Short Term Investments", "Long Term Debt", "Current Debt",
}
FLUJO = {"Operating Cash Flow", "Capital Expenditure"}
# El resto (revenue, márgenes, impuestos, acciones) va a la cuenta de resultados.


def consolidar(directorio: Path, forzar: bool = False) -> Path:
    """Une los parquets trimestrales en uno solo, con la primera publicación.

    El resultado es la base de todo lo demás: (cik, etiqueta, ddate, qtrs) ->
    valor y fecha de primera publicación.
    """
    destino = directorio / "consolidado.parquet"
    if destino.exists() and not forzar:
        return destino

    piezas = sorted((directorio / "parsed").glob("*.parquet"))
    if not piezas:
        raise FileNotFoundError(
            f"No hay parquets en {directorio / 'parsed'}. Corre antes sec_parse.")

    marcos = [pd.read_parquet(p) for p in piezas]
    todo = pd.concat(marcos, ignore_index=True)
    log.info("filas leídas: %d", len(todo))

    # Erratas de las propias presentaciones: hay periodos fechados en 1927 y en
    # 2923. Son el 0,05%, pero un periodo fantasma en el futuro o cuarenta años
    # atrás se cuela en el pivote y aparece como una columna más.
    # Un periodo no puede cerrar después de presentarse, y nada anterior a 2005
    # tiene sentido en un panel que arranca en 2012.
    antes = len(todo)
    todo = todo[(todo["ddate"] <= todo["filed"])
                & (todo["ddate"] >= pd.Timestamp("2005-01-01"))]
    descartadas = antes - len(todo)
    if descartadas:
        log.info("descartadas %d filas con fecha imposible (%.3f%%)",
                 descartadas, descartadas / antes * 100)

    # La primera publicación de cada dato. `sort_values` + `drop_duplicates`
    # deja la fila con el `filed` más antiguo de cada periodo.
    todo = todo.sort_values("filed")
    claves = ["cik", "etiqueta", "ddate", "qtrs"]
    if "componente" in todo.columns:
        # El capex son varias partidas que se suman; deduplicar sin el
        # componente se quedaría solo con una de ellas.
        claves.insert(2, "componente")
    todo = todo.drop_duplicates(subset=claves, keep="first")
    log.info("filas tras quedarnos con la primera publicación: %d", len(todo))

    todo = todo.sort_values(["cik", "etiqueta", "ddate"])
    todo.to_parquet(destino, index=False)
    return destino


def _mapa_tickers(ruta: Path) -> dict[str, int]:
    """ticker en mayúsculas -> CIK, del fichero oficial de la SEC."""
    datos = json.loads(ruta.read_text())
    return {v["ticker"].upper(): int(v["cik_str"]) for v in datos.values()}



def _con_deuda_total(marco: pd.DataFrame | None) -> pd.DataFrame | None:
    """Compone las filas de balance que las métricas esperan ya agregadas.

    `_financials.net_debt` busca "Total Debt" o "Net Debt"; la SEC publica la
    deuda partida en tramo corriente y no corriente, así que sin componerla B9
    se queda con cobertura 0% y el motor la desactiva.

    Lo mismo con el efectivo: la fila preferida de Yahoo es la que suma efectivo
    e inversiones a corto. Usar solo el efectivo sobreestimaría la deuda neta.
    """
    if marco is None or marco.empty:
        return marco
    marco = marco.copy()

    tramos = [t for t in ("Long Term Debt", "Current Debt") if t in marco.index]
    if tramos:
        # `sum` con min_count=1: si ninguno tiene dato el periodo queda en NaN en
        # vez de en cero, que se leería como "sin deuda".
        marco.loc["Total Debt"] = marco.loc[tramos].sum(min_count=1)

    liquido = [c for c in ("Cash And Cash Equivalents", "Short Term Investments")
               if c in marco.index]
    if liquido:
        marco.loc["Cash Cash Equivalents And Short Term Investments"] = (
            marco.loc[liquido].sum(min_count=1))
    return marco


def _con_ebitda(marco: pd.DataFrame | None) -> pd.DataFrame | None:
    """Añade la fila EBITDA, que la SEC no publica.

    `EBITDA` no es un concepto XBRL: aparece 21 veces en un trimestre con miles
    de declarantes, porque no es una magnitud GAAP. Yahoo la calcula y B9
    (net debt / EBITDA) la espera, así que aquí se compone igual: resultado
    operativo declarado más amortizaciones.
    """
    if marco is None or marco.empty:
        return marco
    operativo = "Total Operating Income As Reported"
    amortizacion = "Depreciation And Amortization"
    if operativo not in marco.index or amortizacion not in marco.index:
        return marco
    ebitda = marco.loc[operativo] + marco.loc[amortizacion]
    if ebitda.notna().sum() == 0:
        return marco
    marco = marco.copy()
    marco.loc["EBITDA"] = ebitda
    return marco


class SecProvider:
    """Fundamentales point-in-time desde los datos de la SEC.

    Se carga entero en memoria (~15 M filas, del orden de 1 GB). Es un proveedor
    para estudios y backtests largos, no para la corrida diaria: la máquina de
    producción tiene 1 GB de RAM y no lo aguantaría.
    """

    def __init__(self, directorio: Path | str = "./.cache/sec") -> None:
        self.directorio = Path(directorio)
        self._datos: pd.DataFrame | None = None
        self._por_cik: dict[int, pd.DataFrame] | None = None
        self._tickers: dict[str, int] | None = None

    # -- carga diferida ------------------------------------------------
    @property
    def datos(self) -> pd.DataFrame:
        if self._datos is None:
            ruta = consolidar(self.directorio)
            self._datos = pd.read_parquet(ruta)
            log.info("SEC en memoria: %d filas · %d empresas",
                     len(self._datos), self._datos["cik"].nunique())
        return self._datos

    @property
    def tickers(self) -> dict[str, int]:
        if self._tickers is None:
            self._tickers = _mapa_tickers(self.directorio / "company_tickers.json")
        return self._tickers

    def _del_cik(self, cik: int) -> pd.DataFrame:
        if self._por_cik is None:
            self._por_cik = dict(tuple(self.datos.groupby("cik")))
        return self._por_cik.get(cik, pd.DataFrame())

    def cik(self, symbol: str) -> int | None:
        return self.tickers.get(symbol.upper())

    # -- interfaz FundamentalsProvider ---------------------------------
    def profile(self, symbol: str) -> dict:
        """Solo lo que la SEC sabe: el código SIC.

        No hay capitalización ni valor de empresa: eso sale de los precios, que
        vienen del proveedor de cotizaciones. El sector se deja en manos de quien
        componga los dos, porque traducir SIC a la taxonomía de Yahoo es una
        decisión del que consume, no de este módulo.
        """
        cik = self.cik(symbol)
        if cik is None:
            return {}
        filas = self._del_cik(cik)
        if filas.empty:
            return {}
        sic = filas["sic"].iloc[-1]
        return {"cik": cik, "sic": sic}

    def statements(self, symbol: str) -> Statements:
        cik = self.cik(symbol)
        if cik is None:
            return Statements()
        filas = self._del_cik(cik)
        if filas.empty:
            return Statements()

        def marco(etiquetas: set[str], qtrs: int) -> pd.DataFrame | None:
            """Periodos en columnas y conceptos en filas, como yfinance."""
            trozo = filas[filas["etiqueta"].isin(etiquetas) & (filas["qtrs"] == qtrs)]
            if trozo.empty:
                return None
            # `sum` y no `first`: el capex llega troceado en varias partidas
            # (inmovilizado, software, equipos cedidos) que hay que sumar. Para
            # las etiquetas de un solo componente, sumar da lo mismo.
            tabla = trozo.pivot_table(index="etiqueta", columns="ddate",
                                      values="valor", aggfunc="sum")
            return tabla.sort_index(axis=1, ascending=False) if not tabla.empty else None

        resto = set(filas["etiqueta"].unique()) - BALANCE - FLUJO
        return Statements(
            # qtrs=1 es un trimestre; qtrs=4 un ejercicio; qtrs=0 un saldo.
            income_q=_con_ebitda(marco(resto, 1)),
            income_a=_con_ebitda(marco(resto, 4)),
            cashflow_q=marco(FLUJO, 1),
            cashflow_a=marco(FLUJO, 4),
            balance_q=_con_deuda_total(marco(BALANCE, 0)),
            balance_a=_con_deuda_total(marco(BALANCE, 0)),
        )

    def estimates(self, symbol: str) -> Estimates:
        """Solo el calendario, y con las fechas REALES de presentación.

        `pointintime.publication_date` toma la primera fecha posterior al cierre
        del periodo, que para una empresa que presenta cada trimestre es
        exactamente la presentación que reportó ese periodo. Así el mecanismo
        point-in-time que ya existe funciona sin tocarlo, y con datos mejores.

        Las estimaciones de analistas no están en la SEC: B3 y B8 se quedan sin
        datos y el motor las desactiva solo por cobertura.
        """
        cik = self.cik(symbol)
        if cik is None:
            return Estimates()
        filas = self._del_cik(cik)
        if filas.empty:
            return Estimates()
        fechas = pd.DatetimeIndex(sorted(filas["filed"].unique()))
        # Con índice pero sin columnas, pandas considera el DataFrame vacío y
        # `pointintime.report_dates` lo descartaría, cayendo al retraso fijo de
        # 60 días. La columna `filed` no la lee nadie, pero evita eso.
        return Estimates(
            earnings_dates=pd.DataFrame({"filed": fechas}, index=fechas))
