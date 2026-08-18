"""Caché en disco (parquet) y limitador de peticiones.

Yahoo es una API no oficial: la caché agresiva y la concurrencia baja son lo que
evita los 429. Todo lo que se descarga pasa por aquí.

Los estados financieros se **acumulan**: Yahoo solo sirve ~5 trimestres, pero
al fusionar cada corrida con lo ya guardado el histórico local crece con el
tiempo. El sidecar `.seen.json` anota cuándo se vio por primera vez cada
periodo, que es la semilla del archivo point-in-time del backtest (Hito 3).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_key(key: str) -> str:
    """`^STOXX` o `BRK.B` no pueden ser nombres de fichero tal cual."""
    return _UNSAFE.sub("_", key)


class RateLimiter:
    """Espaciado mínimo entre peticiones, compartido entre hilos."""

    def __init__(self, per_minute: float) -> None:
        self._interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self._interval
        if sleep_for > 0:
            time.sleep(sleep_for)


class DiskCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- rutas ----------------------------------------------------------
    def _path(self, kind: str, key: str, suffix: str) -> Path:
        directory = self.root / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{safe_key(key)}{suffix}"

    def is_fresh(self, path: Path, ttl_hours: float) -> bool:
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        return age_hours < ttl_hours

    # -- DataFrames -----------------------------------------------------
    def read_frame(self, kind: str, key: str, ttl_hours: float) -> pd.DataFrame | None:
        path = self._path(kind, key, ".parquet")
        if not self.is_fresh(path, ttl_hours):
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 — un parquet corrupto no tumba la corrida
            log.warning("caché ilegible %s: %s", path, exc)
            path.unlink(missing_ok=True)
            return None

    def write_frame(self, kind: str, key: str, frame: pd.DataFrame) -> None:
        path = self._path(kind, key, ".parquet")
        try:
            frame.to_parquet(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("no se pudo cachear %s: %s", path, exc)

    # -- JSON -----------------------------------------------------------
    def read_json(self, kind: str, key: str, ttl_hours: float) -> dict | None:
        path = self._path(kind, key, ".json")
        if not self.is_fresh(path, ttl_hours):
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("caché ilegible %s: %s", path, exc)
            path.unlink(missing_ok=True)
            return None

    def write_json(self, kind: str, key: str, payload: dict) -> None:
        path = self._path(kind, key, ".json")
        try:
            path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.warning("no se pudo cachear %s: %s", path, exc)

    # -- estados financieros (periodos en columnas) ---------------------
    def read_statement(self, kind: str, key: str, ttl_hours: float) -> pd.DataFrame | None:
        frame = self.read_frame(kind, key, ttl_hours)
        return None if frame is None else _columns_to_timestamps(frame)

    def write_statement(self, kind: str, key: str, frame: pd.DataFrame, *, accumulate: bool) -> pd.DataFrame:
        """Guarda fusionando con lo ya cacheado. Devuelve la versión fusionada."""
        merged = frame
        if accumulate:
            stored = self.read_frame(kind, key, ttl_hours=float("inf"))
            if stored is not None and not stored.empty:
                merged = _merge_statements(_columns_to_timestamps(stored), frame)
            self._record_first_seen(kind, key, merged)

        self.write_frame(kind, key, _columns_to_strings(merged))
        return merged

    def _record_first_seen(self, kind: str, key: str, frame: pd.DataFrame) -> None:
        """Anota la primera vez que vimos cada periodo: base del point-in-time."""
        path = self._path(kind, key, ".seen.json")
        try:
            seen = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:  # noqa: BLE001
            seen = {}
        today = date.today().isoformat()
        changed = False
        for column in frame.columns:
            label = str(pd.Timestamp(column).date()) if not isinstance(column, str) else column
            if label not in seen:
                seen[label] = today
                changed = True
        if changed:
            try:
                path.write_text(json.dumps(seen, sort_keys=True), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                log.debug("no se pudo anotar first_seen de %s: %s", key, exc)


def _columns_to_strings(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        str(pd.Timestamp(c).date()) if not isinstance(c, str) else c for c in out.columns
    ]
    return out


def _columns_to_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    columns = []
    for c in out.columns:
        try:
            columns.append(pd.Timestamp(c))
        except (ValueError, TypeError):
            columns.append(c)
    out.columns = columns
    return out.sort_index(axis=1)


def _merge_statements(stored: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Unión de periodos; ante el mismo periodo gana el dato recién descargado."""
    combined = fresh.combine_first(stored)
    return combined.sort_index(axis=1)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
