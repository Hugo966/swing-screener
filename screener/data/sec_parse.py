"""Reduce los ZIP de la SEC a un parquet con lo que el panel B necesita.

`num.txt` pesa ~500 MB por trimestre y trae miles de conceptos XBRL. Aquí se
recorta a las ~25 etiquetas que consumen las métricas y se traduce al
vocabulario de yfinance, que es el que ya esperan `metrics/_financials.py` y el
resto del motor. Así el proveedor de la SEC es intercambiable con el de Yahoo
sin tocar ninguna métrica.

Tres cosas que hay que hacer bien o los datos salen plausibles pero falsos:

- **El signo del capex.** Yahoo publica `Capital Expenditure` en negativo y el
  código hace `ocf + capex`. La SEC publica `PaymentsToAcquirePropertyPlant...`
  en positivo, porque es un pago. Sin invertirlo, el flujo de caja libre sale
  inflado y nada avisa.
- **Consolidado, no segmentos.** Una fila con `segments` o `coreg` no vacíos es
  el desglose de una división o de un coemisor. Sumarlas con el total duplica.
- **La duración importa.** `qtrs=0` es un saldo puntual (balance), `qtrs=1` un
  trimestre y `qtrs=4` un ejercicio. Mezclarlas compara un trimestre con un año.

El campo `filed` viaja con cada dato: es la fecha real de presentación y el
motivo entero de preferir la SEC a Yahoo.
"""

from __future__ import annotations

import csv
import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger("sec.parse")


def _componentes(*grupos: tuple[str, ...]) -> tuple[str, ...]:
    """Marca una etiqueta como suma de componentes.

    Devuelve todas las etiquetas XBRL implicadas; `COMPONENTES` recuerda a qué
    componente pertenece cada una para que el proveedor sume una sola por grupo.
    """
    return tuple(t for grupo in grupos for t in grupo)


# etiqueta XBRL -> (etiqueta yfinance, índice de componente). Solo para las que
# se suman; el resto compite dentro de una única cadena.
COMPONENTE_DE: dict[str, int] = {}


def _registrar_componentes(etiqueta: str, *grupos: tuple[str, ...]) -> None:
    for i, grupo in enumerate(grupos):
        for tag in grupo:
            COMPONENTE_DE[tag] = i


# Vocabulario de yfinance -> etiquetas XBRL, en orden de preferencia.
# Las cadenas largas existen porque la NIIF 15 (ASC 606) cambió en 2018 cómo se
# declaran los ingresos, y conviven varias formas según el sector y el año.
MAPA: dict[str, tuple[str, ...]] = {
    "Total Revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "RevenueFromContractsWithCustomers",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromSaleOfGoods",
        "RevenuesNetOfInterestExpense",
        "Revenue",
    ),
    "Net Income": ("NetIncomeLoss", "ProfitLoss",
                   "NetIncomeLossAvailableToCommonStockholdersBasic"),
    "Gross Profit": ("GrossProfit",),
    # Yahoo tiene DOS filas distintas: "Operating Income" es una cifra
    # normalizada que excluye extraordinarios, y "Total Operating Income As
    # Reported" es el GAAP. Para Intel en 2025 son -23 M y -2.214 M. La SEC
    # publica el GAAP, así que se etiqueta como tal: llamarlo "Operating
    # Income" haría que B5 y B6 midieran cosas distintas según la fuente.
    # `_financials.OPERATING_INCOME` ya lo busca como alternativa.
    "Total Operating Income As Reported": ("OperatingIncomeLoss",),
    "Operating Cash Flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    # El capex NO es una etiqueta sino la suma de partidas distintas, y cada
    # empresa usa las que le aplican. Mastercard declara 371 M de inmovilizado
    # y 717 M de software: solo la suma cuadra con los 1.088 M que publica
    # Yahoo. Caterpillar añade equipos cedidos en arrendamiento. Quedarse con la
    # primera etiqueta infravalora la inversión y, al calcular OCF - capex,
    # infla el flujo de caja libre justo en la métrica de más peso del panel B.
    #
    # Se declara como componentes con su propia cadena de alternativas: dentro
    # de un componente gana la primera (son formas distintas de decir lo mismo),
    # y los componentes se suman entre sí.
    # Inversión en activo productivo. Se deja fuera el M&A
    # (`PaymentsToAcquireBusinesses...`), la inversión financiera (valores,
    # participadas) y la compra de intangibles.
    #
    # Lo de los intangibles se probó y se descartó con datos: Yahoo los incluye
    # en el capex de Johnson & Johnson pero no en el de Philip Morris ni el de
    # Verizon, así que añadirlos arreglaba un caso y rompía dos —la
    # concordancia global bajaba del 97% al 95%. El criterio que queda es el
    # defendible por sí mismo: capacidad productiva sí, compra de activos
    # intangibles no. JNJ queda como diferencia conocida.
    #
    # `PaymentsToAcquireOtherProductiveAssets` es la que usa Verizon para casi
    # todo su capex: sin ella salían 450 M en vez de 17.500 M.
    "Capital Expenditure": _componentes(
        ("PaymentsToAcquirePropertyPlantAndEquipment",
         "PaymentsToAcquireOtherProductiveAssets",
         "PaymentsToAcquireProductiveAssets",
         "PaymentsForCapitalImprovements"),
        ("PaymentsToAcquireSoftware", "PaymentsToDevelopSoftware"),
        ("PaymentsToAcquireEquipmentOnLease",),
    ),
    "Stock Based Compensation": ("ShareBasedCompensation",
                                 "AllocatedShareBasedCompensationExpense"),
    "Diluted Average Shares": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    "Basic Average Shares": ("WeightedAverageNumberOfSharesOutstandingBasic",
                            "WeightedAverageNumberOfSharesOutstanding"),
    "Total Assets": ("Assets",),
    "Stockholders Equity": ("StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "Cash And Cash Equivalents": ("CashAndCashEquivalentsAtCarryingValue",
                                  "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "Short Term Investments": ("ShortTermInvestments",
                               "AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
    # La deuda viene repartida en muchas etiquetas segun el emisor y el año.
    # Se recogen por tramo (no corriente / corriente) y el proveedor las suma;
    # dentro de cada tramo gana la primera declarada, sin acumular, para no
    # contar dos veces la misma deuda expresada de dos formas.
    "Long Term Debt": ("LongTermDebtNoncurrent", "LongTermDebt",
                       "LongTermDebtAndCapitalLeaseObligations",
                       "NotesPayable"),
    "Current Debt": ("LongTermDebtCurrent", "DebtCurrent",
                     "ShortTermBorrowings", "NotesPayableCurrent",
                     "LongTermDebtAndCapitalLeaseObligationsCurrent"),
    "Depreciation And Amortization": (
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ),
    "Interest Expense": ("InterestExpense", "InterestExpenseDebt"),
    "Tax Provision": ("IncomeTaxExpenseBenefit",),
    "Pretax Income": ("IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
                      "ExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterest"
                      "AndIncomeLossFromEquityMethodInvestments"),
}

_registrar_componentes(
    "Capital Expenditure",
    ("PaymentsToAcquirePropertyPlantAndEquipment",
     "PaymentsToAcquireOtherProductiveAssets",
     "PaymentsToAcquireProductiveAssets",
     "PaymentsForCapitalImprovements"),
    ("PaymentsToAcquireSoftware", "PaymentsToDevelopSoftware"),
    ("PaymentsToAcquireEquipmentOnLease",),
)

# Etiquetas que la SEC publica con signo contrario al de yfinance.
INVERTIR = {"Capital Expenditure"}

# XBRL -> etiqueta de yfinance, para resolver en O(1) al recorrer num.txt.
DESDE_XBRL: dict[str, str] = {
    xbrl: etiqueta for etiqueta, xbrls in MAPA.items() for xbrl in xbrls
}
# Prioridad dentro de cada cadena: si una empresa declara varias, gana la primera.
PRIORIDAD: dict[str, int] = {
    xbrl: i for xbrls in MAPA.values() for i, xbrl in enumerate(xbrls)
}

FORMAS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}


def _submissions(zf: zipfile.ZipFile) -> dict[str, dict]:
    """`adsh` -> metadatos, solo de informes anuales y trimestrales."""
    salida = {}
    with zf.open("sub.txt") as bruto:
        texto = io.TextIOWrapper(bruto, encoding="utf-8", errors="replace")
        for fila in csv.DictReader(texto, delimiter="\t"):
            if fila["form"] not in FORMAS or not fila["filed"]:
                continue
            try:
                cik = int(fila["cik"])
            except (TypeError, ValueError):
                continue
            salida[fila["adsh"]] = {
                "cik": cik,
                "form": fila["form"],
                "filed": fila["filed"],
                "sic": fila.get("sic") or "",
                "fy": fila.get("fy") or "",
                "fp": fila.get("fp") or "",
            }
    return salida


def parsear(ruta_zip: Path) -> pd.DataFrame:
    """Un trimestre -> filas (cik, etiqueta, fin de periodo, duración, valor, filed)."""
    with zipfile.ZipFile(ruta_zip) as zf:
        subs = _submissions(zf)
        if not subs:
            return pd.DataFrame()

        # (cik, etiqueta, ddate, qtrs) -> (prioridad, fila). La prioridad decide
        # cuando una empresa declara el mismo concepto con dos etiquetas XBRL.
        mejor: dict[tuple, tuple[int, dict]] = {}

        with zf.open("num.txt") as bruto:
            texto = io.TextIOWrapper(bruto, encoding="utf-8", errors="replace")
            for fila in csv.DictReader(texto, delimiter="\t"):
                etiqueta = DESDE_XBRL.get(fila["tag"])
                if etiqueta is None:
                    continue
                # Solo el consolidado: `segments` es el desglose por división y
                # `coreg` el de un coemisor. Sumarlos con el total duplicaría.
                if fila.get("segments") or fila.get("coreg"):
                    continue
                meta = subs.get(fila["adsh"])
                if meta is None or not fila["value"]:
                    continue
                try:
                    valor = float(fila["value"])
                    qtrs = int(fila["qtrs"])
                except (TypeError, ValueError):
                    continue

                if etiqueta in INVERTIR:
                    valor = -valor

                componente = COMPONENTE_DE.get(fila["tag"], 0)
                clave = (meta["cik"], etiqueta, componente, fila["ddate"], qtrs)
                prio = PRIORIDAD[fila["tag"]]
                previo = mejor.get(clave)
                if previo is None or prio < previo[0]:
                    mejor[clave] = (prio, {
                        "cik": meta["cik"],
                        "etiqueta": etiqueta,
                        "componente": componente,
                        "ddate": fila["ddate"],
                        "qtrs": qtrs,
                        "valor": valor,
                        "filed": meta["filed"],
                        "form": meta["form"],
                        "sic": meta["sic"],
                    })

    if not mejor:
        return pd.DataFrame()
    marco = pd.DataFrame([v[1] for v in mejor.values()])
    marco["ddate"] = pd.to_datetime(marco["ddate"], format="%Y%m%d", errors="coerce")
    marco["filed"] = pd.to_datetime(marco["filed"], format="%Y%m%d", errors="coerce")
    return marco.dropna(subset=["ddate", "filed"])


def sincronizar(directorio: Path) -> Path:
    """Parsea los ZIP que falten y devuelve el directorio de parquets."""
    destino = directorio / "parsed"
    destino.mkdir(parents=True, exist_ok=True)

    for zip_path in sorted(directorio.glob("*.zip")):
        salida = destino / f"{zip_path.stem}.parquet"
        if salida.exists():
            continue
        try:
            marco = parsear(zip_path)
        except zipfile.BadZipFile:
            log.warning("%s corrupto, se salta", zip_path.name)
            continue
        if marco.empty:
            log.warning("%s sin filas útiles", zip_path.name)
            continue
        marco.to_parquet(salida, index=False)
        log.info("%s: %d filas · %d empresas", zip_path.stem,
                 len(marco), marco["cik"].nunique())
    return destino


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    raiz = Path(sys.argv[1] if len(sys.argv) > 1 else "./.cache/sec")
    destino = sincronizar(raiz)
    piezas = sorted(destino.glob("*.parquet"))
    if piezas:
        total = sum(pd.read_parquet(p, columns=["cik"]).shape[0] for p in piezas)
        print(f"\n{len(piezas)} trimestres parseados · {total:,} filas")
