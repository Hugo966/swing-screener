"""Construcción del universo por región y aplicación de los gates (§3).

Los gates se aplican en **dos tandas** y ese orden importa: primero lo que solo
necesita el screener y los precios (tamaño, liquidez, sector, tendencia), y solo
después se piden fundamentales de los supervivientes. Con el tier gratuito la
diferencia es entre ~1.800 tickers y unos pocos cientos de descargas pesadas.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import pandas as pd

from screener.data.provider import DataProvider
from screener.metrics._indicators import sma
from screener.models import Candidate, GateResult, Region

__all__ = [
    "apply_cheap_gates",
    "apply_trend_gate",
    "build_candidates",
    "deduplicate_listings",
    "drop_excluded_exchanges",
    "fx_market_cap_to_usd",
    "fx_to_usd",
    "is_subunit",
    "passes_country",
    "passes_industry",
    "summarize",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Divisa
# ---------------------------------------------------------------------------
def _major_rate(code: str, provider: DataProvider) -> float | None:
    """USD por unidad **mayor** de la divisa (GBP, EUR, SEK...)."""
    if code == "USD":
        return 1.0
    if code in provider.fx:
        return provider.fx[code]
    series = provider.prices.close_series(f"{code}USD=X", period="5d")
    rate = float(series.iloc[-1]) if series is not None and len(series) else None
    provider.fx[code] = rate
    if rate is None:
        log.warning("sin tipo de cambio para %s: los gates en USD no se pueden aplicar", code)
    return rate


def is_subunit(currency: str) -> bool:
    """`GBp`, `ZAc`, `ILA`: la minúscula final denota la subunidad (peniques...)."""
    return bool(currency) and currency[-1].islower()


def fx_to_usd(currency: str, provider: DataProvider) -> float | None:
    """USD por unidad **cotizada**. Es la que convierte precios y volumen en euros."""
    currency = currency or "USD"
    rate = _major_rate(currency.upper(), provider)
    if rate is None:
        return None
    return rate / 100.0 if is_subunit(currency) else rate


def fx_market_cap_to_usd(currency: str, provider: DataProvider) -> float | None:
    """USD por unidad de **capitalización**, que no siempre es la cotizada.

    Yahoo publica el precio de las británicas en peniques (GBp) pero su
    `marketCap` en libras: comprobado con `marketCap / (acciones × precio) = 0.01`
    en las 628 líneas GBp del universo europeo. Aplicar el tipo de los peniques
    a la capitalización la dividiría por 100 y todo el Reino Unido caería en el
    gate de tamaño.
    """
    return _major_rate((currency or "USD").upper(), provider)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def passes_country(country: str | None, region: Region) -> bool:
    """El domicilio de la empresa, no la bolsa donde cotiza.

    En emergentes la cabeza del universo por capitalización son **DR de empresas
    extranjeras**: NVDC34.SA es Nvidia en Brasil, NVDA80.BK es Nvidia en
    Tailandia, BTI.JO es British American Tobacco en Johannesburgo. Sin este
    filtro, "emergentes" screenearía megacaps americanas por su línea local.

    Se aplica solo si la región declara `countries`; sin lista, no filtra.
    """
    if not region.countries:
        return True
    if not country:
        return False  # sin domicilio conocido no se puede afirmar que sea local
    allowed = {c.lower() for c in region.countries}
    return country.strip().lower() in allowed


def passes_industry(industry: str | None, cfg) -> bool:
    """Bancos, seguros y REITs rompen el panel B aunque su sector no sea el excluido."""
    if not industry:
        return True
    lowered = industry.lower()
    keywords = cfg.gates.get("excluded_industry_keywords") or []
    return not any(str(k).lower() in lowered for k in keywords)


def _trend_ok(prices: pd.DataFrame, ma_fast: int, ma_slow: int) -> tuple[bool, str]:
    close = prices["Close"].dropna()
    if len(close) < ma_slow:
        return False, f"solo {len(close)} sesiones, se necesitan {ma_slow}"

    fast = sma(close, ma_fast).iloc[-1]
    slow = sma(close, ma_slow).iloc[-1]
    price = float(close.iloc[-1])
    if pd.isna(fast) or pd.isna(slow):
        return False, "medias móviles incompletas"
    if price <= float(slow):
        return False, f"precio {price:.2f} <= MM{ma_slow} {float(slow):.2f}"
    if float(fast) <= float(slow):
        return False, f"MM{ma_fast} {float(fast):.2f} <= MM{ma_slow} {float(slow):.2f}"
    return True, ""


def current_rates(currency: str, provider: DataProvider) -> tuple[float, float]:
    """(tipo de la unidad cotizada, tipo de la unidad de capitalización) de hoy."""
    return (
        fx_to_usd(currency, provider) or 1.0,
        fx_market_cap_to_usd(currency, provider) or 1.0,
    )


def apply_cheap_gates(
    candidates: list[Candidate],
    cfg,
    provider: DataProvider,
    *,
    rates=None,
) -> tuple[list[Candidate], list[GateResult]]:
    """Gates que solo necesitan los metadatos del screener: tamaño y liquidez.

    `rates` permite inyectar los tipos de cambio **de una fecha pasada**: el
    backtest necesita FX histórico (§11) y esto evita duplicar los gates.
    """
    if rates is None:
        def rates(currency):  # noqa: E306
            return current_rates(currency, provider)

    gates = cfg.gates
    min_cap = float(gates["min_market_cap_usd"])
    min_dollar_volume = float(gates["min_dollar_volume_usd"])
    excluded_sectors = {s.lower() for s in gates.get("excluded_sectors", [])}

    survivors: list[Candidate] = []
    results: list[GateResult] = []

    for candidate in candidates:
        if candidate.sector and candidate.sector.lower() in excluded_sectors:
            results.append(GateResult(candidate.symbol, False, "sector", candidate.sector))
            continue

        # Dos tipos distintos a propósito: la capitalización va en la unidad
        # mayor y el precio en la cotizada, y en el Reino Unido no coinciden.
        quote_rate, cap_rate = rates(candidate.currency)

        candidate.market_cap_usd = (
            candidate.market_cap * cap_rate if candidate.market_cap is not None else None
        )

        if candidate.market_cap_usd is None or candidate.market_cap_usd < min_cap:
            results.append(
                GateResult(candidate.symbol, False, "tamaño", f"{candidate.market_cap_usd}")
            )
            continue

        dollar_volume = candidate.avg_dollar_volume
        if dollar_volume is not None:
            dollar_volume *= quote_rate
        if dollar_volume is None or dollar_volume < min_dollar_volume:
            results.append(
                GateResult(candidate.symbol, False, "liquidez", f"{dollar_volume}")
            )
            continue

        survivors.append(candidate)
        results.append(GateResult(candidate.symbol, True))

    return survivors, results


def apply_trend_gate(
    candidates: list[Candidate], prices: dict[str, pd.DataFrame], cfg
) -> tuple[list[Candidate], list[GateResult]]:
    """Gate de tendencia: precio > MM200 y MM50 > MM200."""
    trend = cfg.gates["trend"]
    ma_fast, ma_slow = int(trend["ma_fast"]), int(trend["ma_slow"])
    min_history = int(cfg.gates.get("min_price_history_days", ma_slow))

    survivors: list[Candidate] = []
    results: list[GateResult] = []

    for candidate in candidates:
        frame = prices.get(candidate.symbol)
        if frame is None or frame.empty:
            results.append(GateResult(candidate.symbol, False, "sin precios", ""))
            continue
        if len(frame) < min_history:
            results.append(
                GateResult(candidate.symbol, False, "historia", f"{len(frame)} sesiones")
            )
            continue

        ok, detail = _trend_ok(frame, ma_fast, ma_slow)
        results.append(GateResult(candidate.symbol, ok, None if ok else "tendencia", detail))
        if ok:
            survivors.append(candidate)

    return survivors, results


# ---------------------------------------------------------------------------
# Universo
# ---------------------------------------------------------------------------
# Palabras que no identifican a la empresa: forma jurídica y clase de acción.
# Las clases importan porque la ordinaria y la preferente del mismo emisor son
# dos líneas (SamsungElec / SamsungElec(1P), PETR3 / PETR4) y comprar las dos es
# duplicar exposición, no diversificar.
# Incluye además los códigos de segmento de la bolsa brasileña (NM = Novo
# Mercado, N1/N2 = Nível 1 y 2...), que van pegados al nombre y no identifican
# nada: "PETROBRAS PN ATZ N2" y "PETROBRAS ON NM" son el mismo emisor.
_NOISE_TOKENS = frozenset(
    """plc ag sa nv na ab asa as se spa oyj group holding holdings company co inc
    corp corporation ltd limited kgaa publ the of and pn on pna pnb pnc unt units
    ord ordinary shares share cl class nm n1 n2 ma mb m2 atz ed ex""".split()
)
# Sufijos de ≥3 letras que además se recortan del nombre ya unido, para que
# "ENEL SPA" y "ENEL S.P.A." coincidan. Los de 2 letras no se recortan aquí:
# "cisco" acabaría en "cis".
_JOINED_SUFFIXES = ("holdings", "holding", "limited", "company", "group", "corp",
                    "plc", "ltd", "spa", "inc", "asa", "oyj")

_PAREN = re.compile(r"\([^)]*\)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _company_key(name: str) -> str:
    """Nombre normalizado, para reconocer la misma empresa en dos líneas.

    Se tokeniza tras eliminar puntuación —si no, "CUPID LTD." no coincide con
    "CUPID LIMITED"— y se quitan los paréntesis, que es donde va la clase de
    acción coreana ("SamsungElec(1P)").
    """
    lowered = _PAREN.sub(" ", str(name).lower())
    tokens = [t for t in _NON_ALNUM.split(lowered) if t]
    # Los de una letra son casi siempre restos de puntuación ("S.P.A." -> s, p, a);
    # si el nombre entero fuese de una letra, el fallback de abajo lo conserva.
    meaningful = [t for t in tokens if len(t) > 1 and t not in _NOISE_TOKENS]
    joined = "".join(meaningful or tokens)

    for suffix in _JOINED_SUFFIXES:
        if joined.endswith(suffix) and len(joined) > len(suffix) + 2:
            joined = joined[: -len(suffix)]
            break
    return joined


def drop_excluded_exchanges(candidates: list[Candidate], cfg) -> list[Candidate]:
    """Fuera las bolsas que solo replican cotizaciones de otras.

    En la región europea, Cboe UK e IOB aportan 275 de 1.313 líneas y ninguna es
    una empresa nueva: Cboe UK duplica el LSE (HSBAL.XC / HSBA.L) e IOB son
    cross-listings de valores de fuera (un húngaro cotizando en Londres en HUF).
    """
    excluded = {str(e).lower() for e in (cfg.universe.get("excluded_exchanges") or [])}
    if not excluded:
        return candidates
    kept = [c for c in candidates if (c.exchange or "").lower() not in excluded]
    if len(kept) < len(candidates):
        log.info("descartadas %d líneas de bolsas replicadas", len(candidates) - len(kept))
    return kept


def deduplicate_listings(candidates: list[Candidate]) -> list[Candidate]:
    """Una empresa, una línea: la más líquida.

    Una cotización dual (BBVA en Madrid y Londres, AstraZeneca en Londres y
    Estocolmo) entraría dos veces al universo, contaría doble en el percentil y
    podría disparar dos alertas de la misma empresa.
    """
    best: dict[str, Candidate] = {}
    order: list[str] = []
    for candidate in candidates:
        key = _company_key(candidate.name) if candidate.name else ""
        if not key:
            key = f"__{candidate.symbol}"
        if key not in best:
            best[key] = candidate
            order.append(key)
            continue
        # La watchlist manda: es la línea que Hugo sigue.
        current = best[key]
        if candidate.is_watchlist and not current.is_watchlist:
            best[key] = candidate
        elif current.is_watchlist and not candidate.is_watchlist:
            continue
        elif (candidate.avg_dollar_volume or 0) > (current.avg_dollar_volume or 0):
            best[key] = candidate

    kept = [best[k] for k in order]
    if len(kept) < len(candidates):
        log.info("deduplicadas %d cotizaciones duales", len(candidates) - len(kept))
    return kept


def build_candidates(region: Region, provider: DataProvider, cfg) -> list[Candidate]:
    """Universo amplio del screener + la watchlist personal, deduplicado."""
    candidates = provider.universe.candidates(region)
    candidates = drop_excluded_exchanges(candidates, cfg)
    candidates = deduplicate_listings(candidates)

    max_symbols = cfg.universe.get("max_symbols")
    if max_symbols:
        candidates = candidates[: int(max_symbols)]
        log.info("universo %s recortado a %d por universe.max_symbols", region.key, len(candidates))

    by_symbol = {c.symbol: c for c in candidates}
    for symbol in set(cfg.watchlist):
        if symbol in by_symbol:
            by_symbol[symbol].is_watchlist = True
        else:
            # La watchlist se puntúa siempre, aunque no salga del screener; hay
            # que traerle los metadatos a mano para que pueda pasar los gates.
            by_symbol[symbol] = _watchlist_candidate(symbol, provider)

    return list(by_symbol.values())


def _watchlist_candidate(symbol: str, provider: DataProvider) -> Candidate:
    profile = provider.fundamentals.profile(symbol)

    # El screener es quien normalmente aporta precio y volumen medio; aquí se
    # derivan de los propios precios para que el gate de liquidez sea aplicable.
    frame = provider.prices.history([symbol], period="1y").get(symbol)
    price = avg_volume = None
    if frame is not None and not frame.empty:
        price = float(frame["Close"].iloc[-1])
        avg_volume = float(frame["Volume"].iloc[-63:].mean())

    return Candidate(
        symbol=symbol,
        name=str(profile.get("longName") or profile.get("shortName") or ""),
        sector=profile.get("sector"),
        industry=profile.get("industry"),
        market_cap=_as_float(profile.get("marketCap")),
        avg_volume=avg_volume,
        price=price,
        currency=str(profile.get("currency") or "USD"),
        exchange=str(profile.get("exchange") or ""),
        is_watchlist=True,
    )


def _as_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def summarize(results: list[GateResult]) -> str:
    failures = Counter(r.failed_gate for r in results if not r.passed)
    if not failures:
        return "sin descartes"
    return ", ".join(f"{gate}: {count}" for gate, count in failures.most_common())
