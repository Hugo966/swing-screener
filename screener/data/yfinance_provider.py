"""Proveedor yfinance: precios, fundamentales, estimaciones y screener.

Es el proveedor primario de la Fase 1. Cubre gratis lo que la spec asignaba a
FMP: estados trimestrales y anuales, `earnings_dates` (sorpresa histórica con
fecha real de publicación), `eps_revisions`/`eps_trend` y el screener por país.

Todo fallo de red o de dato se degrada a None/vacío: el motor lo trata como
cobertura ausente, nunca como excepción que tumbe la corrida.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from screener.data.cache import DiskCache, RateLimiter
from screener.data.provider import Estimates, Statements
from screener.models import Candidate, Region

log = logging.getLogger(__name__)

# Los 11 sectores de la taxonomía de Yahoo. El screener no devuelve el sector en
# el quote, pero sí lo acepta como filtro: consultando sector a sector cada
# ticker queda etiquetado gratis.
YAHOO_SECTORS = (
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
)

_PROFILE_KEYS = (
    "sector",
    "industry",
    "country",  # domicilio: distingue la empresa local del DR de una extranjera
    "marketCap",
    "enterpriseValue",
    "currency",
    "financialCurrency",
    "sharesOutstanding",
    "quoteType",
    "exchange",
    "longName",
    "shortName",
)

_HISTORY_CHUNK = 100


class YFinanceProvider:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.cache = DiskCache(cfg.data["cache_dir"])
        self.limiter = RateLimiter(float(cfg.data["requests_per_minute"]))
        self.accumulate = bool(cfg.data.get("accumulate_statements", True))

    # ------------------------------------------------------------------
    # Universo
    # ------------------------------------------------------------------
    def candidates(self, region: Region) -> list[Candidate]:
        ttl = self.cfg.cache_ttl_hours("universe")
        cached = self.cache.read_frame("universe", region.key, ttl)
        if cached is not None:
            return [_candidate_from_row(row) for _, row in cached.iterrows()]

        excluded = {s.lower() for s in self.cfg.gates.get("excluded_sectors", [])}
        sectors = [s for s in YAHOO_SECTORS if s.lower() not in excluded]

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for yahoo_region in region.yahoo_regions:
            for sector in sectors:
                for quote in self._screen_sector(yahoo_region, sector):
                    symbol = quote.get("symbol")
                    if not symbol or symbol in seen:
                        continue
                    if quote.get("quoteType") != "EQUITY":
                        continue
                    seen.add(symbol)
                    rows.append(
                        {
                            "symbol": symbol,
                            "name": quote.get("shortName") or quote.get("longName") or "",
                            # El sector no viene en el quote: es el del filtro usado.
                            "sector": sector,
                            "market_cap": quote.get("marketCap"),
                            "avg_volume": quote.get("averageDailyVolume3Month"),
                            "price": quote.get("regularMarketPrice"),
                            "currency": quote.get("currency") or "USD",
                            "exchange": quote.get("fullExchangeName") or "",
                        }
                    )

        frame = pd.DataFrame(rows)
        if not frame.empty:
            self.cache.write_frame("universe", region.key, frame)
        log.info("universo %s: %d candidatos del screener", region.key, len(frame))
        return [_candidate_from_row(row) for _, row in frame.iterrows()]

    def _screen_sector(self, yahoo_region: str, sector: str) -> list[dict]:
        page_size = int(self.cfg.universe["page_size"])
        max_pages = int(self.cfg.universe["max_pages_per_sector"])
        min_cap = float(self.cfg.gates["min_market_cap_usd"])
        min_volume = float(self.cfg.universe["screener_min_volume"])

        query = yf.EquityQuery(
            "and",
            [
                yf.EquityQuery("eq", ["region", yahoo_region]),
                yf.EquityQuery("eq", ["sector", sector]),
                yf.EquityQuery("gt", ["intradaymarketcap", min_cap]),
                yf.EquityQuery("gt", ["avgdailyvol3m", min_volume]),
            ],
        )

        quotes: list[dict] = []
        for page in range(max_pages):
            self.limiter.wait()
            try:
                response = yf.screen(
                    query,
                    offset=page * page_size,
                    size=page_size,
                    sortField="intradaymarketcap",
                    sortAsc=False,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("screener %s/%s página %d falló: %s", yahoo_region, sector, page, exc)
                break
            batch = response.get("quotes") or []
            quotes.extend(batch)
            if len(batch) < page_size:
                break
        return quotes

    # ------------------------------------------------------------------
    # Precios
    # ------------------------------------------------------------------
    def history(self, symbols: list[str], *, period: str) -> dict[str, pd.DataFrame]:
        ttl = self.cfg.cache_ttl_hours("prices")
        result: dict[str, pd.DataFrame] = {}
        pending: list[str] = []

        for symbol in symbols:
            cached = self.cache.read_frame("prices", f"{symbol}@{period}", ttl)
            if cached is not None and not cached.empty:
                result[symbol] = _ensure_datetime_index(cached)
            else:
                pending.append(symbol)

        for start in range(0, len(pending), _HISTORY_CHUNK):
            chunk = pending[start : start + _HISTORY_CHUNK]
            self.limiter.wait()
            downloaded = self._download(chunk, period)
            for symbol, frame in downloaded.items():
                if frame is None or frame.empty:
                    continue
                self.cache.write_frame("prices", f"{symbol}@{period}", frame)
                result[symbol] = frame
            log.info("precios: %d/%d descargados", min(start + _HISTORY_CHUNK, len(pending)), len(pending))

        return result

    def _download(self, symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
        try:
            raw = yf.download(
                symbols,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("descarga de precios falló para %d símbolos: %s", len(symbols), exc)
            return {}
        return _split_download(raw, symbols)

    def close_series(self, symbol: str, *, period: str) -> pd.Series | None:
        frames = self.history([symbol], period=period)
        frame = frames.get(symbol)
        if frame is None or frame.empty or "Close" not in frame:
            return None
        return frame["Close"].dropna()

    # ------------------------------------------------------------------
    # Fundamentales
    # ------------------------------------------------------------------
    def profile(self, symbol: str) -> dict:
        ttl = self.cfg.cache_ttl_hours("profile")
        cached = self.cache.read_json("profile", symbol, ttl)
        if cached is not None:
            return cached

        self.limiter.wait()
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception as exc:  # noqa: BLE001
            log.debug("info de %s falló: %s", symbol, exc)
            info = {}
        payload = {key: info.get(key) for key in _PROFILE_KEYS}
        self.cache.write_json("profile", symbol, payload)
        return payload

    def statements(self, symbol: str) -> Statements:
        ttl = self.cfg.cache_ttl_hours("fundamentals")
        specs = (
            ("income_q", "quarterly_income_stmt"),
            ("income_a", "income_stmt"),
            ("cashflow_q", "quarterly_cashflow"),
            ("cashflow_a", "cashflow"),
            ("balance_q", "quarterly_balance_sheet"),
            ("balance_a", "balance_sheet"),
        )

        frames: dict[str, pd.DataFrame | None] = {}
        missing = []
        for field_name, _ in specs:
            cached = self.cache.read_statement(f"fundamentals/{field_name}", symbol, ttl)
            frames[field_name] = cached
            if cached is None:
                missing.append(field_name)

        if missing:
            self.limiter.wait()
            ticker = self._ticker(symbol)
            for field_name, attribute in specs:
                if field_name not in missing:
                    continue
                frame = _safe_frame(ticker, attribute, symbol)
                if frame is None or frame.empty:
                    continue  # queda en None: el motor lo lee como dato ausente
                frames[field_name] = self.cache.write_statement(
                    f"fundamentals/{field_name}", symbol, frame, accumulate=self.accumulate
                )

        return Statements(**frames)

    def estimates(self, symbol: str) -> Estimates:
        ttl = self.cfg.cache_ttl_hours("estimates")

        earnings = self.cache.read_frame("estimates/earnings_dates", symbol, ttl)
        revisions = self.cache.read_frame("estimates/eps_revisions", symbol, ttl)
        trend = self.cache.read_frame("estimates/eps_trend", symbol, ttl)
        shares = self.cache.read_frame("estimates/shares_full", symbol, ttl)

        if earnings is None or revisions is None or trend is None or shares is None:
            self.limiter.wait()
            ticker = self._ticker(symbol)

            if earnings is None:
                earnings = _safe_earnings_dates(ticker, symbol)
                if earnings is not None and not earnings.empty:
                    self.cache.write_frame("estimates/earnings_dates", symbol, earnings)
            if revisions is None:
                revisions = _safe_frame(ticker, "eps_revisions", symbol)
                if revisions is not None and not revisions.empty:
                    self.cache.write_frame("estimates/eps_revisions", symbol, revisions)
            if trend is None:
                trend = _safe_frame(ticker, "eps_trend", symbol)
                if trend is not None and not trend.empty:
                    self.cache.write_frame("estimates/eps_trend", symbol, trend)
            if shares is None:
                shares = _safe_shares(ticker, symbol, int(self.cfg.metric_params("dilution_sbc").get("years", 3)))
                if shares is not None and not shares.empty:
                    self.cache.write_frame("estimates/shares_full", symbol, shares)

        return Estimates(
            earnings_dates=_ensure_datetime_index(earnings) if earnings is not None else None,
            eps_revisions=revisions,
            eps_trend=trend,
            shares_full=(
                _ensure_datetime_index(shares).iloc[:, 0] if shares is not None and not shares.empty else None
            ),
        )

    def _ticker(self, symbol: str) -> yf.Ticker:
        return yf.Ticker(symbol)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _candidate_from_row(row) -> Candidate:
    return Candidate(
        symbol=str(row["symbol"]),
        name=str(row.get("name") or ""),
        sector=row.get("sector"),
        market_cap=_as_float(row.get("market_cap")),
        avg_volume=_as_float(row.get("avg_volume")),
        price=_as_float(row.get("price")),
        currency=str(row.get("currency") or "USD"),
        exchange=str(row.get("exchange") or ""),
    )


def _as_float(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_datetime_index(frame):
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, utc=False)
        except (ValueError, TypeError):
            return out
    return out.sort_index()


def _split_download(raw: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """`yf.download` devuelve columnas MultiIndex con varios símbolos y planas con uno."""
    if raw is None or raw.empty:
        return {}
    wanted = ["Open", "High", "Low", "Close", "Volume"]

    if isinstance(raw.columns, pd.MultiIndex):
        out: dict[str, pd.DataFrame] = {}
        available = set(raw.columns.get_level_values(0))
        for symbol in symbols:
            if symbol not in available:
                continue
            frame = raw[symbol].dropna(how="all")
            columns = [c for c in wanted if c in frame.columns]
            if not columns or frame.empty:
                continue
            out[symbol] = frame[columns].dropna(subset=["Close"])
        return out

    columns = [c for c in wanted if c in raw.columns]
    if not columns or len(symbols) != 1:
        return {}
    frame = raw[columns].dropna(subset=["Close"])
    return {symbols[0]: frame} if not frame.empty else {}


def _safe_frame(ticker: yf.Ticker, attribute: str, symbol: str) -> pd.DataFrame | None:
    try:
        frame = getattr(ticker, attribute)
    except Exception as exc:  # noqa: BLE001
        log.debug("%s.%s falló: %s", symbol, attribute, exc)
        return None
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    return frame


def _safe_earnings_dates(ticker: yf.Ticker, symbol: str) -> pd.DataFrame | None:
    try:
        frame = ticker.get_earnings_dates(limit=24)
    except Exception as exc:  # noqa: BLE001
        log.debug("%s.get_earnings_dates falló: %s", symbol, exc)
        return None
    if frame is None or frame.empty:
        return None
    out = frame.copy()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out


def _safe_shares(ticker: yf.Ticker, symbol: str, years: int) -> pd.DataFrame | None:
    start = (pd.Timestamp.today() - pd.DateOffset(years=years + 1)).date().isoformat()
    try:
        series = ticker.get_shares_full(start=start)
    except Exception as exc:  # noqa: BLE001
        log.debug("%s.get_shares_full falló: %s", symbol, exc)
        return None
    if series is None or len(series) == 0:
        return None
    frame = series.to_frame(name="shares")
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    return frame
