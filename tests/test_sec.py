"""Tests del proveedor de la SEC. Sin red: se fabrica un ZIP igual que el real."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from screener.data import sec_download, sec_parse
from screener.data.sec_provider import SecProvider, consolidar

SUB_COLS = ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed"]
NUM_COLS = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg",
            "value", "footnote"]


def _zip(tmp: Path, nombre: str, subs: list[dict], nums: list[dict]) -> Path:
    """Un ZIP con la misma forma que los de la SEC."""
    def tsv(cols, filas):
        buf = io.StringIO()
        buf.write("\t".join(cols) + "\n")
        for f in filas:
            buf.write("\t".join(str(f.get(c, "")) for c in cols) + "\n")
        return buf.getvalue()

    destino = tmp / nombre
    with zipfile.ZipFile(destino, "w") as zf:
        zf.writestr("sub.txt", tsv(SUB_COLS, subs))
        zf.writestr("num.txt", tsv(NUM_COLS, nums))
    return destino


def _sub(adsh: str, cik: int = 320193, form: str = "10-K", filed: str = "20240201") -> dict:
    return {"adsh": adsh, "cik": cik, "name": "TEST CO", "sic": "3571",
            "form": form, "period": "20231231", "fy": "2023", "fp": "FY",
            "filed": filed}


def _num(adsh: str, tag: str, valor, ddate: str = "20231231", qtrs: int = 4,
         **extra) -> dict:
    fila = {"adsh": adsh, "tag": tag, "version": "us-gaap/2023", "ddate": ddate,
            "qtrs": qtrs, "uom": "USD", "segments": "", "coreg": "",
            "value": valor, "footnote": ""}
    fila.update(extra)
    return fila


# ---------------------------------------------------------------- parseo


def test_capex_cambia_de_signo(tmp_path):
    """La SEC lo publica positivo y yfinance negativo; el código suma OCF+capex.

    Sin invertirlo el flujo de caja libre saldría inflado sin que nada avise, que
    es el error más caro de todo el módulo.
    """
    z = _zip(tmp_path, "2024q1.zip", [_sub("a-1")],
             [_num("a-1", "PaymentsToAcquirePropertyPlantAndEquipment", 5000)])
    salida = sec_parse.parsear(z)
    fila = salida[salida["etiqueta"] == "Capital Expenditure"].iloc[0]
    assert fila["valor"] == -5000


def test_ignora_segmentos_y_coemisores(tmp_path):
    """Sumar el desglose de una división con el total duplicaría los ingresos."""
    z = _zip(tmp_path, "2024q1.zip", [_sub("a-1")], [
        _num("a-1", "Revenues", 1000),
        _num("a-1", "Revenues", 400, segments="ProductoA"),
        _num("a-1", "Revenues", 600, coreg="FilialB"),
    ])
    salida = sec_parse.parsear(z)
    revenue = salida[salida["etiqueta"] == "Total Revenue"]
    assert len(revenue) == 1
    assert revenue.iloc[0]["valor"] == 1000


def test_prioridad_entre_etiquetas_xbrl(tmp_path):
    """Si una empresa declara el concepto de dos formas, gana la preferente.

    ASC 606 partió los ingresos en varias etiquetas y algunas empresas publican
    más de una; sin un orden, el valor elegido dependería del orden de lectura.
    """
    z = _zip(tmp_path, "2024q1.zip", [_sub("a-1")], [
        _num("a-1", "Revenues", 999),
        _num("a-1", "RevenueFromContractWithCustomerExcludingAssessedTax", 1000),
    ])
    salida = sec_parse.parsear(z)
    revenue = salida[salida["etiqueta"] == "Total Revenue"]
    assert len(revenue) == 1
    assert revenue.iloc[0]["valor"] == 1000      # gana la primera de la cadena


def test_solo_informes_periodicos(tmp_path):
    """Un 8-K no trae estados completos; colarlo mezclaría datos parciales."""
    z = _zip(tmp_path, "2024q1.zip",
             [_sub("a-1", form="8-K"), _sub("b-2", form="10-Q")],
             [_num("a-1", "Revenues", 500), _num("b-2", "Revenues", 700)])
    salida = sec_parse.parsear(z)
    assert salida["valor"].tolist() == [700]


def test_conserva_la_fecha_de_presentacion(tmp_path):
    """`filed` es el motivo entero de usar la SEC en vez de Yahoo."""
    z = _zip(tmp_path, "2024q1.zip", [_sub("a-1", filed="20240215")],
             [_num("a-1", "Revenues", 1000)])
    salida = sec_parse.parsear(z)
    assert salida.iloc[0]["filed"] == pd.Timestamp("2024-02-15")
    # El cierre del periodo es muy anterior: esa distancia es lo que Yahoo no da.
    assert salida.iloc[0]["ddate"] == pd.Timestamp("2023-12-31")


# ------------------------------------------------------------ consolidación


def _monta(tmp_path, zips: list[tuple[str, list, list]]) -> Path:
    raiz = tmp_path / "sec"
    (raiz / "parsed").mkdir(parents=True)
    for nombre, subs, nums in zips:
        z = _zip(raiz, nombre, subs, nums)
        marco = sec_parse.parsear(z)
        if not marco.empty:
            marco.to_parquet(raiz / "parsed" / f"{Path(nombre).stem}.parquet",
                             index=False)
    (raiz / "company_tickers.json").write_text(
        json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "TEST CO"}}))
    return raiz


def test_gana_la_primera_publicacion(tmp_path):
    """Un 10-K posterior repite periodos antiguos; vale la cifra original.

    Si hubo reexpresión, el mercado de entonces vio el número viejo. Quedarse con
    el revisado sería look-ahead disfrazado de dato limpio.
    """
    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1", filed="20240201")],
         [_num("a-1", "Revenues", 1000)]),
        # Un año después el mismo periodo reaparece, reexpresado.
        ("2025q1.zip", [_sub("b-2", filed="20250201")],
         [_num("b-2", "Revenues", 1234)]),
    ])
    datos = pd.read_parquet(consolidar(raiz))
    revenue = datos[datos["etiqueta"] == "Total Revenue"]
    assert len(revenue) == 1
    assert revenue.iloc[0]["valor"] == 1000
    assert revenue.iloc[0]["filed"] == pd.Timestamp("2024-02-01")


# -------------------------------------------------------------- proveedor


def test_statements_tiene_forma_de_yfinance(tmp_path):
    """Periodos en columnas y conceptos en filas: lo que esperan las métricas."""
    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1")], [
            _num("a-1", "Revenues", 1000),
            _num("a-1", "NetIncomeLoss", 100),
            _num("a-1", "Assets", 5000, qtrs=0),
            _num("a-1", "NetCashProvidedByUsedInOperatingActivities", 300),
        ]),
    ])
    prov = SecProvider(raiz)
    est = prov.statements("AAPL")

    assert est.income_a is not None
    assert "Total Revenue" in est.income_a.index
    assert pd.Timestamp("2023-12-31") in est.income_a.columns
    assert est.income_a.loc["Total Revenue"].iloc[0] == 1000
    # El balance va aparte: es un saldo puntual, no un flujo.
    assert "Total Assets" in est.balance_q.index
    assert "Total Assets" not in est.income_a.index
    assert "Operating Cash Flow" in est.cashflow_a.index


def test_estimates_devuelve_fechas_de_presentacion(tmp_path):
    """`pointintime` las usa para decidir qué se sabía en cada fecha."""
    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1", filed="20240215")], [_num("a-1", "Revenues", 1)]),
    ])
    est = SecProvider(raiz).estimates("AAPL")
    assert list(est.earnings_dates.index) == [pd.Timestamp("2024-02-15")]
    # B3 y B8 no existen en la SEC: se quedan vacías a propósito.
    assert est.eps_revisions is None


def test_simbolo_desconocido_no_revienta(tmp_path):
    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1")], [_num("a-1", "Revenues", 1)]),
    ])
    prov = SecProvider(raiz)
    assert prov.cik("NOEXISTE") is None
    assert prov.statements("NOEXISTE").is_empty()
    assert prov.estimates("NOEXISTE").earnings_dates is None


def test_el_punto_en_el_tiempo_respeta_la_presentacion(tmp_path):
    """La prueba que justifica todo el módulo.

    Un periodo cerrado en diciembre pero presentado en febrero no puede verse en
    enero. Con el cierre de periodo se vería, y ahí está el look-ahead.
    """
    from screener.models import TickerData
    from screener.pointintime import as_of

    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1", filed="20240215")],
         [_num("a-1", "Revenues", 1000)]),
    ])
    prov = SecProvider(raiz)
    datos = TickerData(
        symbol="AAPL", asof=pd.Timestamp("2024-03-01").date(), sector="Technology",
        income_a=prov.statements("AAPL").income_a,
        earnings_dates=prov.estimates("AAPL").earnings_dates,
    )

    # El retraso por defecto se pone absurdo a propósito: si el test pasara con
    # él, estaría midiendo el respaldo y no las fechas reales de la SEC.
    absurdo = 9999

    antes = as_of(datos, pd.Timestamp("2024-02-14"), fallback_lag_days=absurdo)
    assert antes.income_a is None, "un día antes de presentarlo no puede verse"

    despues = as_of(datos, pd.Timestamp("2024-02-15"), fallback_lag_days=absurdo)
    assert despues.income_a is not None, "el día de la presentación ya es público"
    assert despues.income_a.loc["Total Revenue"].iloc[0] == 1000


# -------------------------------------------------------------- descarga


def test_trimestres_arrancan_en_2012():
    """Antes de 2012 el XBRL cubre tan pocas empresas que sesgaría el panel."""
    from datetime import date
    t = sec_download.trimestres(hasta=date(2026, 8, 25))
    assert t[0] == "2012q1"
    assert "2011q4" not in t
    # El trimestre en curso no está: su ZIP aún no existe.
    assert "2026q3" not in t


def test_user_agent_exige_correo(monkeypatch):
    """Comprobado contra la SEC: sin correo devuelve 403."""
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    with pytest.raises(RuntimeError, match="SEC_CONTACT_EMAIL"):
        sec_download.user_agent()

    monkeypatch.setenv("SEC_CONTACT_EMAIL", "alguien@ejemplo.com")
    assert "alguien@ejemplo.com" in sec_download.user_agent()


# ---------------------------------------------------------------- híbrido


class _YahooFalso:
    """Lo mínimo de un FundamentalsProvider, sin red."""

    def __init__(self):
        self.pedidos = []

    def profile(self, symbol):
        self.pedidos.append(("profile", symbol))
        return {"sector": "Technology", "marketCap": 1e12, "currency": "USD"}

    def statements(self, symbol):
        self.pedidos.append(("statements", symbol))
        from screener.data.provider import Statements
        return Statements(income_a=pd.DataFrame(
            {pd.Timestamp("2023-12-31"): [42.0]}, index=["Total Revenue"]))

    def estimates(self, symbol):
        self.pedidos.append(("estimates", symbol))
        from screener.data.provider import Estimates
        return Estimates(
            earnings_dates=pd.DataFrame(index=pd.DatetimeIndex(["2024-01-25"])),
            eps_revisions=pd.DataFrame({"up": [3]}),
        )


def _hibrido(tmp_path, con_datos=True):
    from screener.data.hybrid_provider import HybridFundamentals
    zips = [("2024q1.zip", [_sub("a-1", filed="20240215")],
             [_num("a-1", "Revenues", 1000)])] if con_datos else []
    raiz = _monta(tmp_path, zips) if zips else _monta(
        tmp_path, [("2024q1.zip", [_sub("a-1", cik=999)], [_num("a-1", "Revenues", 5)])])
    yahoo = _YahooFalso()
    return HybridFundamentals(yahoo, raiz), yahoo


def test_hibrido_toma_el_perfil_de_yahoo(tmp_path):
    """La SEC solo tiene el SIC; el sector y la capitalización son de Yahoo."""
    hib, yahoo = _hibrido(tmp_path)
    perfil = hib.profile("AAPL")
    assert perfil["sector"] == "Technology"
    assert perfil["marketCap"] == 1e12
    assert ("profile", "AAPL") in yahoo.pedidos


def test_hibrido_prefiere_los_estados_de_la_sec(tmp_path):
    """Es el motivo de existir: profundidad histórica y fecha real."""
    hib, yahoo = _hibrido(tmp_path)
    est = hib.statements("AAPL")
    assert est.income_a.loc["Total Revenue"].iloc[0] == 1000    # SEC, no los 42 de Yahoo
    assert ("statements", "AAPL") not in yahoo.pedidos


def test_hibrido_cae_a_yahoo_sin_cik(tmp_path):
    """Un ADR o una extranjera no presentan XBRL: mejor Yahoo que nada."""
    hib, yahoo = _hibrido(tmp_path, con_datos=False)
    est = hib.statements("AAPL")
    assert est.income_a.loc["Total Revenue"].iloc[0] == 42      # el de Yahoo
    assert ("statements", "AAPL") in yahoo.pedidos


def test_hibrido_combina_el_calendario(tmp_path):
    """Fechas de presentación de la SEC, revisiones de Yahoo."""
    hib, _ = _hibrido(tmp_path)
    est = hib.estimates("AAPL")
    assert list(est.earnings_dates.index) == [pd.Timestamp("2024-02-15")]  # SEC
    assert est.eps_revisions is not None                                   # Yahoo


def test_la_fabrica_monta_el_hibrido():
    """`provider: sec` en el config debe componer SEC + Yahoo, no fallar."""
    from screener.config import load_config
    from screener.data.provider import build_provider
    from screener.data.hybrid_provider import HybridFundamentals

    cfg = load_config()
    region = cfg.region("us")
    object.__setattr__(region, "provider", "sec")
    try:
        prov = build_provider(region, cfg)
        assert isinstance(prov.fundamentals, HybridFundamentals)
        assert prov.name == "sec+yfinance"
        # Los precios siguen siendo de Yahoo: la SEC no cotiza nada.
        assert prov.prices is not prov.fundamentals
    finally:
        object.__setattr__(region, "provider", "yfinance")


def test_la_fabrica_rechaza_precios_de_la_sec():
    """Pedir cotizaciones a la SEC es un error de config: mejor fallar pronto."""
    from screener.config import load_config
    from screener.data.provider import ProviderError, build_provider

    cfg = load_config()
    region = cfg.region("us")
    object.__setattr__(region, "price_provider", "sec")
    try:
        with pytest.raises(ProviderError, match="no sirve precios"):
            build_provider(region, cfg)
    finally:
        object.__setattr__(region, "price_provider", "yfinance")


def test_descarta_fechas_imposibles(tmp_path):
    """Hay presentaciones con periodos fechados en 1927 y en 2923.

    Son erratas del declarante. Un periodo no puede cerrar después de haberse
    presentado, y colarlo añadiría una columna fantasma al pivote de estados.
    """
    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1", filed="20240201")], [
            _num("a-1", "Revenues", 1000, ddate="20231231"),
            _num("a-1", "Revenues", 777, ddate="29230630"),   # futuro imposible
            _num("a-1", "Revenues", 555, ddate="19270228"),   # errata antigua
        ]),
    ])
    datos = pd.read_parquet(consolidar(raiz))
    assert len(datos) == 1
    assert datos.iloc[0]["valor"] == 1000


def test_capex_suma_sus_componentes(tmp_path):
    """El capex son varias partidas, no una.

    Mastercard declara 371 M de inmovilizado y 717 M de software; solo la suma
    cuadra con lo que publica Yahoo. Quedarse con una infravalora la inversión
    e infla el flujo de caja libre, que es lo que mide B4 con peso 16.
    """
    z = _zip(tmp_path, "2024q1.zip", [_sub("a-1")], [
        _num("a-1", "PaymentsToAcquirePropertyPlantAndEquipment", 371),
        _num("a-1", "PaymentsToAcquireSoftware", 717),
    ])
    salida = sec_parse.parsear(z)
    capex = salida[salida["etiqueta"] == "Capital Expenditure"]
    assert len(capex) == 2, "cada componente sobrevive por separado"
    assert capex["valor"].sum() == -(371 + 717)


def test_dentro_de_un_componente_no_se_suma(tmp_path):
    """Dos formas de decir lo mismo compiten; no se acumulan.

    `PaymentsToAcquireProductiveAssets` es la alternativa que usan algunos
    declarantes en lugar de la de inmovilizado, no una partida adicional.
    """
    z = _zip(tmp_path, "2024q1.zip", [_sub("a-1")], [
        _num("a-1", "PaymentsToAcquirePropertyPlantAndEquipment", 100),
        _num("a-1", "PaymentsToAcquireProductiveAssets", 999),
    ])
    salida = sec_parse.parsear(z)
    capex = salida[salida["etiqueta"] == "Capital Expenditure"]
    assert len(capex) == 1
    assert capex.iloc[0]["valor"] == -100     # gana la preferente


def test_el_proveedor_suma_el_capex(tmp_path):
    """De extremo a extremo: el marco de flujos debe traer el capex completo."""
    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1")], [
            _num("a-1", "PaymentsToAcquirePropertyPlantAndEquipment", 371),
            _num("a-1", "PaymentsToAcquireSoftware", 717),
            _num("a-1", "NetCashProvidedByUsedInOperatingActivities", 5000),
        ]),
    ])
    est = SecProvider(raiz).statements("AAPL")
    assert est.cashflow_a.loc["Capital Expenditure"].iloc[0] == -1088
    # Y el FCF reconstruido sale bien: OCF + capex, con el capex en negativo.
    from screener.metrics import _financials as fin
    fcf = fin.free_cash_flow(est.cashflow_a)
    assert fcf.iloc[0] == 5000 - 1088


def test_operating_income_se_etiqueta_como_gaap(tmp_path):
    """Yahoo distingue el operativo normalizado del declarado; la SEC da el GAAP.

    Para Intel en 2025, Yahoo dice -23 M en "Operating Income" y -2.214 M en
    "Total Operating Income As Reported". La SEC publica el segundo. Etiquetarlo
    como el primero haría que B5 y B6 midieran cosas distintas según la fuente.
    """
    z = _zip(tmp_path, "2024q1.zip", [_sub("a-1")],
             [_num("a-1", "OperatingIncomeLoss", -2214)])
    salida = sec_parse.parsear(z)
    assert salida.iloc[0]["etiqueta"] == "Total Operating Income As Reported"

    # Las métricas lo encuentran igual: está en la cadena de alternativas.
    from screener.metrics import _financials as fin
    marco = pd.DataFrame({pd.Timestamp("2023-12-31"): [-2214.0]},
                         index=["Total Operating Income As Reported"])
    assert fin.line(marco, fin.OPERATING_INCOME) is not None


def test_ebitda_se_compone(tmp_path):
    """No es un concepto XBRL y B9 lo necesita: operativo + amortizaciones."""
    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1")], [
            _num("a-1", "OperatingIncomeLoss", 1000),
            _num("a-1", "DepreciationDepletionAndAmortization", 250),
        ]),
    ])
    est = SecProvider(raiz).statements("AAPL")
    assert "EBITDA" in est.income_a.index
    assert est.income_a.loc["EBITDA"].iloc[0] == 1250

    from screener.metrics import _financials as fin
    assert fin.line(est.income_a, fin.EBITDA) is not None


def test_sin_amortizaciones_no_se_inventa_ebitda(tmp_path):
    """Mejor sin EBITDA que con uno igual al resultado operativo."""
    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1")], [_num("a-1", "OperatingIncomeLoss", 1000)]),
    ])
    est = SecProvider(raiz).statements("AAPL")
    assert "EBITDA" not in est.income_a.index


def test_capex_no_incluye_intangibles(tmp_path):
    """Probado con datos y descartado: Yahoo no los trata igual entre empresas.

    Los incluye en el capex de Johnson & Johnson pero no en el de Philip Morris
    ni el de Verizon. Añadirlos arreglaba un caso y rompía dos. El criterio que
    queda —capacidad productiva sí, compra de intangibles no— es defendible por
    sí mismo y da mejor concordancia.
    """
    z = _zip(tmp_path, "2024q1.zip", [_sub("a-1")], [
        _num("a-1", "PaymentsToAcquirePropertyPlantAndEquipment", 100),
        _num("a-1", "PaymentsToAcquireIntangibleAssets", 900),
    ])
    salida = sec_parse.parsear(z)
    capex = salida[salida["etiqueta"] == "Capital Expenditure"]
    assert len(capex) == 1
    assert capex.iloc[0]["valor"] == -100


def test_capex_reconoce_other_productive_assets(tmp_path):
    """Verizon declara casi todo su capex ahí: sin esta etiqueta salían 450 M
    en lugar de 17.500 M."""
    z = _zip(tmp_path, "2024q1.zip", [_sub("a-1")],
             [_num("a-1", "PaymentsToAcquireOtherProductiveAssets", 18767)])
    salida = sec_parse.parsear(z)
    capex = salida[salida["etiqueta"] == "Capital Expenditure"]
    assert capex.iloc[0]["valor"] == -18767


def test_compone_la_deuda_total_para_b9(tmp_path):
    """`net_debt` busca "Total Debt"; la SEC la publica partida en dos tramos.

    Sin componerla, B9 (net debt/EBITDA) sale con cobertura 0% y el motor la
    desactiva sin que se note más que en una línea de log.
    """
    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1")], [
            _num("a-1", "LongTermDebtNoncurrent", 8000, qtrs=0),
            _num("a-1", "LongTermDebtCurrent", 2000, qtrs=0),
            _num("a-1", "CashAndCashEquivalentsAtCarryingValue", 1500, qtrs=0),
            _num("a-1", "ShortTermInvestments", 500, qtrs=0),
        ]),
    ])
    est = SecProvider(raiz).statements("AAPL")
    assert est.balance_q.loc["Total Debt"].iloc[0] == 10000
    assert est.balance_q.loc[
        "Cash Cash Equivalents And Short Term Investments"].iloc[0] == 2000

    from screener.metrics import _financials as fin
    # 10.000 de deuda menos 2.000 de liquidez: la deuda neta que espera B9.
    assert fin.net_debt(est.balance_q).iloc[0] == 8000


def test_sin_deuda_declarada_no_se_inventa_un_cero(tmp_path):
    """Un cero se leería como "sin deuda", que es una afirmación, no un hueco."""
    raiz = _monta(tmp_path, [
        ("2024q1.zip", [_sub("a-1")],
         [_num("a-1", "Assets", 5000, qtrs=0)]),
    ])
    est = SecProvider(raiz).statements("AAPL")
    assert "Total Debt" not in est.balance_q.index
