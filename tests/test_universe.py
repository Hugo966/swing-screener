"""Gates y conversión de divisa."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.data.provider import DataProvider
from screener.models import Candidate
from screener.universe import (
    _company_key,
    deduplicate_listings,
    drop_excluded_exchanges,
    fx_market_cap_to_usd,
    is_subunit,
    apply_cheap_gates,
    apply_trend_gate,
    fx_to_usd,
    passes_country,
    passes_industry,
    summarize,
)
from tests.conftest import make_prices


class FxPrices:
    """Sólo sirve pares de divisa: EURUSD=X a 1.10, GBPUSD=X a 1.25."""

    RATES = {"EURUSD=X": 1.10, "GBPUSD=X": 1.25}

    def __init__(self):
        self.calls = []

    def history(self, symbols, *, period):
        return {}

    def close_series(self, symbol, *, period):
        self.calls.append(symbol)
        rate = self.RATES.get(symbol)
        return None if rate is None else pd.Series([rate])


@pytest.fixture
def provider():
    prices = FxPrices()
    return DataProvider(prices=prices, fundamentals=None, universe=None)


# ---------------------------------------------------------------------------
# FX
# ---------------------------------------------------------------------------
def test_usd_needs_no_lookup(provider):
    assert fx_to_usd("USD", provider) == 1.0
    assert provider.prices.calls == []


def test_pence_and_pounds_do_not_share_a_cache_entry(provider):
    """GBp son peniques: mezclarlo con GBP daría un market cap 100x mal."""
    assert fx_to_usd("GBP", provider) == pytest.approx(1.25)
    assert fx_to_usd("GBp", provider) == pytest.approx(0.0125)


def test_fx_is_cached_per_currency(provider):
    fx_to_usd("EUR", provider)
    fx_to_usd("EUR", provider)
    assert provider.prices.calls.count("EURUSD=X") == 1


def test_unknown_currency_returns_none(provider):
    assert fx_to_usd("XYZ", provider) is None


# ---------------------------------------------------------------------------
# Gates baratos
# ---------------------------------------------------------------------------
def candidate(symbol, **kwargs):
    defaults = dict(
        symbol=symbol, sector="Technology", market_cap=5e9, avg_volume=1e6,
        price=100.0, currency="USD",
    )
    defaults.update(kwargs)
    return Candidate(**defaults)


def test_cheap_gates_reject_small_illiquid_and_excluded(cfg, provider):
    candidates = [
        candidate("OK"),
        candidate("PEQUENA", market_cap=1e9),                    # < 2.000 M
        candidate("ILIQUIDA", avg_volume=100.0),                 # < 5 M$/día
        candidate("BANCO", sector="Financial Services"),
        candidate("INMO", sector="Real Estate"),
    ]
    survivors, log = apply_cheap_gates(candidates, cfg, provider)

    assert [c.symbol for c in survivors] == ["OK"]
    failures = {g.symbol: g.failed_gate for g in log if not g.passed}
    assert failures == {
        "PEQUENA": "tamaño", "ILIQUIDA": "liquidez",
        "BANCO": "sector", "INMO": "sector",
    }
    assert "tamaño: 1" in summarize(log)


def test_size_gate_is_applied_in_usd(cfg, provider):
    """Una europea de 1.900 M€ vale 2.090 M$: pasa el gate, que es en dólares."""
    euro = candidate("EUR1", market_cap=1.9e9, currency="EUR", price=100.0)
    survivors, _ = apply_cheap_gates([euro], cfg, provider)

    assert [c.symbol for c in survivors] == ["EUR1"]
    assert euro.market_cap_usd == pytest.approx(1.9e9 * 1.10)


def test_liquidity_gate_uses_dollar_volume_not_share_volume(cfg, provider):
    """Un millón de acciones a 1$ no es lo mismo que a 100$."""
    cheap = candidate("PENNY", price=1.0, avg_volume=1e6)     # 1 M$/día
    pricey = candidate("CARA", price=100.0, avg_volume=1e6)   # 100 M$/día

    survivors, _ = apply_cheap_gates([cheap, pricey], cfg, provider)
    assert [c.symbol for c in survivors] == ["CARA"]


# ---------------------------------------------------------------------------
# Gate de tendencia
# ---------------------------------------------------------------------------
def test_trend_gate_requires_price_and_ma_above_the_slow_ma(cfg):
    days = 400
    rising = make_prices(100.0 * np.cumprod(np.full(days, 1.002)))
    falling = make_prices(100.0 * np.cumprod(np.full(days, 0.998)))
    short = make_prices(100.0 * np.cumprod(np.full(120, 1.002)))

    candidates = [candidate("SUBE"), candidate("BAJA"), candidate("CORTA")]
    prices = {"SUBE": rising, "BAJA": falling, "CORTA": short}

    survivors, log = apply_trend_gate(candidates, prices, cfg)

    assert [c.symbol for c in survivors] == ["SUBE"]
    failures = {g.symbol: g.failed_gate for g in log if not g.passed}
    assert failures == {"BAJA": "tendencia", "CORTA": "historia"}


def test_trend_gate_reports_missing_prices(cfg):
    survivors, log = apply_trend_gate([candidate("FANTASMA")], {}, cfg)
    assert survivors == []
    assert log[0].failed_gate == "sin precios"


# ---------------------------------------------------------------------------
# Industria
# ---------------------------------------------------------------------------
def test_industry_keywords_catch_what_the_sector_filter_misses(cfg):
    """El screener excluye sectores; bancos y REITs se cuelan por la industria."""
    assert not passes_industry("Banks - Regional", cfg)
    assert not passes_industry("REIT - Industrial", cfg)
    assert not passes_industry("Insurance - Life", cfg)
    assert passes_industry("Software - Infrastructure", cfg)
    assert passes_industry(None, cfg)


# ---------------------------------------------------------------------------
# Europa: divisas en subunidad, bolsas replicadas y cotizaciones duales
# ---------------------------------------------------------------------------
def test_market_cap_uses_the_major_unit_even_when_the_price_is_in_pence(provider):
    """Yahoo cotiza las británicas en peniques pero publica el marketCap en libras.

    Aplicarle el tipo de los peniques lo dividiría por 100 y todo el Reino Unido
    caería en el gate de tamaño.
    """
    assert fx_market_cap_to_usd("GBp", provider) == pytest.approx(1.25)
    assert fx_to_usd("GBp", provider) == pytest.approx(0.0125)
    # en una divisa sin subunidad ambos coinciden
    assert fx_market_cap_to_usd("EUR", provider) == fx_to_usd("EUR", provider)


def test_is_subunit_detects_the_lowercase_tail():
    assert is_subunit("GBp") and is_subunit("ZAc")
    assert not is_subunit("GBP") and not is_subunit("EUR") and not is_subunit("")


def test_uk_stock_passes_the_size_gate(cfg, provider):
    """3.000 M£ de capitalización son 4.030 M$: pasa. Con el tipo de peniques no pasaría."""
    uk = candidate("HSBA.L", market_cap=3e9, currency="GBp", price=800.0, avg_volume=1e6)
    survivors, _ = apply_cheap_gates([uk], cfg, provider)

    assert [c.symbol for c in survivors] == ["HSBA.L"]
    assert uk.market_cap_usd == pytest.approx(3e9 * 1.25)


def test_uk_liquidity_uses_the_quoted_unit(cfg, provider):
    """800 peniques × 1 M acciones = 8 M£ = 10 M$/día, no 1.000 M$."""
    uk = candidate("HSBA.L", market_cap=3e9, currency="GBp", price=800.0, avg_volume=1e6)
    apply_cheap_gates([uk], cfg, provider)
    assert uk.avg_dollar_volume * fx_to_usd("GBp", provider) == pytest.approx(10e6)


def test_replicated_exchanges_are_dropped(cfg):
    candidates = [
        candidate("HSBA.L", exchange="LSE"),
        candidate("HSBAL.XC", exchange="Cboe UK"),
        candidate("0M69.IL", exchange="IOB"),
    ]
    kept = drop_excluded_exchanges(candidates, cfg)
    assert [c.symbol for c in kept] == ["HSBA.L"]


def test_dual_listings_collapse_to_the_most_liquid_line():
    """BBVA en Madrid y Londres es una empresa, no dos: contaría doble el percentil."""
    madrid = candidate("BBVA.MC", name="Banco Bilbao Vizcaya Argentaria SA", avg_volume=5e6)
    london = candidate("BVA.L", name="Banco Bilbao Vizcaya Argentaria", avg_volume=1e5)

    kept = deduplicate_listings([london, madrid])
    assert [c.symbol for c in kept] == ["BBVA.MC"]


def test_dedup_keeps_the_watchlist_line_even_if_less_liquid():
    watched = candidate("AZN.L", name="AstraZeneca PLC", avg_volume=1e5)
    watched.is_watchlist = True
    other = candidate("AZN.ST", name="AstraZeneca", avg_volume=9e6)

    kept = deduplicate_listings([other, watched])
    assert [c.symbol for c in kept] == ["AZN.L"]


def test_dedup_does_not_merge_different_companies():
    kept = deduplicate_listings([
        candidate("SAP.DE", name="SAP SE"),
        candidate("SIE.DE", name="Siemens AG"),
        candidate("SAN.MC", name="Banco Santander SA"),
    ])
    assert len(kept) == 3


def test_dedup_falls_back_to_the_symbol_when_there_is_no_name():
    kept = deduplicate_listings([candidate("A", name=""), candidate("B", name="")])
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# Fase 2: DR de empresas extranjeras
# ---------------------------------------------------------------------------
def test_country_gate_rejects_foreign_depositary_receipts(cfg):
    """En emergentes la cabeza del universo son DR de megacaps americanas.

    NVDC34.SA es Nvidia cotizando en Brasil y NVDA80.BK es Nvidia en Tailandia:
    sin este filtro, "emergentes" screenearía empresas de EEUU.
    """
    emerging = cfg.region("emerging")

    assert passes_country("Brazil", emerging)        # PETR4.SA
    assert passes_country("Thailand", emerging)      # PTT.BK
    assert passes_country("South Africa", emerging)  # SOL.JO

    assert not passes_country("United States", emerging)   # NVDC34.SA, NVDA80.BK
    assert not passes_country("United Kingdom", emerging)  # BTI.JO
    assert not passes_country("Netherlands", emerging)     # PRX.JO


def test_country_gate_is_case_and_space_insensitive(cfg):
    assert passes_country("  south korea ", cfg.region("korea"))


def test_unknown_domicile_is_rejected_when_a_list_exists(cfg):
    """Sin domicilio no se puede afirmar que la empresa sea local."""
    assert not passes_country(None, cfg.region("emerging"))
    assert not passes_country("", cfg.region("emerging"))


def test_regions_without_a_country_list_do_not_filter(cfg):
    """EEUU y Europa se validaron sin este gate: no debe cambiarles nada."""
    assert cfg.region("us").countries == []
    assert cfg.region("europe_dev").countries == []
    assert passes_country("United States", cfg.region("us"))
    assert passes_country("Japan", cfg.region("us"))
    assert passes_country(None, cfg.region("europe_dev"))


def test_phase_two_regions_declare_their_domiciles(cfg):
    for key in ("korea", "emerging"):
        assert cfg.region(key).countries, key


# ---------------------------------------------------------------------------
# Normalización de nombres: los casos que fallaron con datos reales
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "left,right,motivo",
    [
        ("CUPID LIMITED", "CUPID LTD.", "el punto de LTD. rompía el token"),
        ("SamsungElec", "SamsungElec(1P)", "ordinaria y preferente coreanas"),
        ("PETROBRAS   PN  ATZ N2", "PETROBRAS   ON      NM", "clases y segmentos de B3"),
        ("ENEL SPA", "ENEL S.P.A.", "S.P.A. se parte en letras sueltas"),
        ("Banco Bilbao Vizcaya Argentaria SA", "Banco Bilbao Vizcaya Argentaria", "sufijo jurídico"),
        ("RELIANCE INDUSTRIES LTD", "Reliance Industries Limited", "LTD vs Limited"),
        ("AstraZeneca PLC", "AstraZeneca", "PLC"),
    ],
)
def test_same_company_two_lines_share_a_key(left, right, motivo):
    """Casos sacados de corridas reales: los dos alertaban por separado."""
    assert _company_key(left) == _company_key(right), motivo


@pytest.mark.parametrize(
    "left,right",
    [
        ("SAP SE", "Siemens AG"),
        ("Cisco Systems", "CIS Group"),
        ("Sasol Limited", "SA Corp"),
        ("TAIWAN SEMICONDUCTOR MANUFACTU", "Taiwan Cement"),
        ("Vale ON", "Valeo"),
    ],
)
def test_different_companies_keep_different_keys(left, right):
    """Recortar de más fusionaría empresas distintas y se perdería una."""
    assert _company_key(left) != _company_key(right)


def test_a_name_made_only_of_noise_is_not_emptied():
    """Si todo el nombre fuese forma jurídica, vaciarlo perdería el ticker."""
    assert _company_key("Holdings PLC") != ""
    assert _company_key("SA") != ""


def test_preferred_and_common_shares_collapse_to_the_most_liquid():
    """Comprar la ordinaria y la preferente del mismo emisor duplica exposición."""
    common = candidate("005930.KS", name="SamsungElec", avg_volume=9e6)
    preferred = candidate("005935.KS", name="SamsungElec(1P)", avg_volume=1e6)

    kept = deduplicate_listings([preferred, common])
    assert [c.symbol for c in kept] == ["005930.KS"]


def test_indian_dual_listing_collapses():
    """CUPID.NS y CUPID.BO alertaban las dos en la primera corrida de emergentes."""
    nse = candidate("CUPID.NS", name="CUPID LIMITED", avg_volume=5e5)
    bse = candidate("CUPID.BO", name="CUPID LTD.", avg_volume=1e4)

    kept = deduplicate_listings([nse, bse])
    assert [c.symbol for c in kept] == ["CUPID.NS"]
