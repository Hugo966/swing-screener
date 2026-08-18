"""Carga y validación de config.yaml + .env.

Toda la configuración vive en el YAML; aquí solo se lee, se valida y se resuelve
en objetos `Region` con los pesos ya filtrados por `active_metrics` y
renormalizados a 100.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from screener.metrics import REGISTRY, names_for_panel
from screener.models import Panel, Region

PANELS: tuple[Panel, ...] = ("momentum", "quality")

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


class ConfigError(ValueError):
    """Config mal formada. Se falla en el arranque, no a mitad de corrida."""


@dataclass
class Secrets:
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_chat_id_watchlist: str | None = None
    fmp_key: str | None = None
    eodhd_key: str | None = None

    @property
    def telegram_ready(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path
    regions: dict[str, Region] = field(default_factory=dict)
    secrets: Secrets = field(default_factory=Secrets)

    # -- accesos con nombre a los bloques del YAML -----------------------
    @property
    def gates(self) -> dict[str, Any]:
        return self.raw["gates"]

    @property
    def alerting(self) -> dict[str, Any]:
        return self.raw["alerting"]

    @property
    def regime(self) -> dict[str, Any]:
        return self.raw["regime"]

    @property
    def coverage(self) -> dict[str, Any]:
        return self.raw["coverage"]

    @property
    def universe(self) -> dict[str, Any]:
        return self.raw["universe"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def run(self) -> dict[str, Any]:
        return self.raw["run"]

    @property
    def watchlist(self) -> list[str]:
        # una watchlist vacía en YAML (`- # comentario`) parsea como [None, None]
        items = self.raw.get("watchlist") or []
        return [str(s).strip().upper() for s in items if s]

    def normalize_against(self, panel: Panel) -> str:
        return self.raw["panels"][panel].get("normalize_against", "universe")

    def metric_params(self, name: str) -> dict[str, Any]:
        return self.raw.get("metric_params", {}).get(name) or {}

    def enabled_regions(self) -> list[Region]:
        return [r for r in self.regions.values() if r.enabled]

    def region(self, key: str) -> Region:
        if key not in self.regions:
            known = ", ".join(sorted(self.regions))
            raise ConfigError(f"región desconocida: {key!r}. Conocidas: {known}")
        return self.regions[key]

    def cache_ttl_hours(self, kind: str) -> float:
        return float(self.data["cache_ttl_hours"][kind])


# ---------------------------------------------------------------------------
# Resolución de pesos
# ---------------------------------------------------------------------------
def resolve_weights(
    base_weights: dict[str, float],
    active: list[str] | None,
    *,
    context: str,
) -> dict[str, float]:
    """Filtra por `active` y renormaliza a 100.

    `active=None` significa "todas". Este es el mecanismo del panel B reducido:
    en Korea quedan 71 pts de peso y se reescalan a 100 sin tocar código.
    """
    if active is None:
        selected = dict(base_weights)
    else:
        unknown = [m for m in active if m not in base_weights]
        if unknown:
            raise ConfigError(
                f"{context}: active_metrics contiene métricas sin peso definido: {unknown}"
            )
        selected = {m: base_weights[m] for m in active}

    total = sum(selected.values())
    if total <= 0:
        raise ConfigError(f"{context}: los pesos activos suman {total}")
    return {name: w * 100.0 / total for name, w in selected.items()}


def _validate_panel_weights(raw: dict[str, Any]) -> None:
    for panel in PANELS:
        if panel not in raw.get("panels", {}):
            raise ConfigError(f"falta el panel {panel!r} en config.panels")
        weights = raw["panels"][panel].get("weights") or {}
        if not weights:
            raise ConfigError(f"panel {panel!r} sin pesos")

        total = sum(weights.values())
        if abs(total - 100.0) > 1e-6:
            raise ConfigError(f"los pesos del panel {panel!r} suman {total}, deben sumar 100")

        registered = set(names_for_panel(panel))
        unknown = sorted(set(weights) - registered)
        if unknown:
            raise ConfigError(
                f"panel {panel!r}: pesos para métricas no registradas: {unknown}. "
                f"Registradas: {sorted(registered)}"
            )
        missing = sorted(registered - set(weights))
        if missing:
            raise ConfigError(f"panel {panel!r}: métricas registradas sin peso: {missing}")


def _build_region(key: str, spec: dict[str, Any], raw: dict[str, Any]) -> Region:
    active = spec.get("active_metrics")
    if active is None:
        active_by_panel: dict[Panel, list[str] | None] = {p: None for p in PANELS}
    elif isinstance(active, dict):
        active_by_panel = {p: active.get(p) for p in PANELS}
    else:
        raise ConfigError(
            f"región {key!r}: active_metrics debe ser null o un mapa "
            f"{{momentum: [...], quality: [...]}}, no {type(active).__name__}"
        )

    weights = {
        panel: resolve_weights(
            raw["panels"][panel]["weights"],
            active_by_panel[panel],
            context=f"región {key!r}, panel {panel!r}",
        )
        for panel in PANELS
    }

    for required in ("benchmark", "provider", "price_provider"):
        if not spec.get(required):
            raise ConfigError(f"región {key!r}: falta {required!r}")

    # `no` (Noruega) es el booleano falso en YAML 1.1, igual que `on`/`off`/`y`/`n`.
    # Sin esta comprobación se cuela como el literal "False" y el screener
    # devuelve cero resultados para esa bolsa sin decir por qué.
    raw_regions = spec.get("yahoo_regions") or []
    non_strings = [r for r in raw_regions if not isinstance(r, str)]
    if non_strings:
        raise ConfigError(
            f"región {key!r}: yahoo_regions contiene valores que YAML no leyó como "
            f"texto: {non_strings}. Entrecomíllalos (\"no\", \"on\", \"off\")."
        )

    return Region(
        key=key,
        enabled=bool(spec.get("enabled", False)),
        phase=int(spec.get("phase", 1)),
        benchmark=str(spec["benchmark"]),
        currency=str(spec.get("currency", "USD")),
        provider=str(spec["provider"]),
        price_provider=str(spec["price_provider"]),
        yahoo_regions=[str(r) for r in (spec.get("yahoo_regions") or [])],
        close_time_utc=str(spec.get("close_time_utc", "21:00")),
        sector_index=str(spec.get("sector_index", "synthetic")),
        weights=weights,
        countries=[str(c) for c in (spec.get("countries") or [])],
    )


# ---------------------------------------------------------------------------
# Entrada pública
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None, *, load_env: bool = True) -> Config:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise ConfigError(f"no existe el fichero de configuración: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    for block in ("gates", "alerting", "regime", "coverage", "panels", "regions",
                  "universe", "data", "run"):
        if block not in raw:
            raise ConfigError(f"falta el bloque {block!r} en {cfg_path}")

    _validate_panel_weights(raw)

    unknown_params = sorted(set(raw.get("metric_params") or {}) - set(REGISTRY))
    if unknown_params:
        raise ConfigError(f"metric_params para métricas no registradas: {unknown_params}")

    regions = {key: _build_region(key, spec, raw) for key, spec in raw["regions"].items()}

    secrets = Secrets()
    if load_env:
        load_dotenv(cfg_path.parent / ".env")
        secrets = Secrets(
            telegram_token=os.getenv("TELEGRAM_TOKEN") or None,
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
            telegram_chat_id_watchlist=os.getenv("TELEGRAM_CHAT_ID_WATCHLIST") or None,
            fmp_key=os.getenv("FMP_KEY") or None,
            eodhd_key=os.getenv("EODHD_KEY") or None,
        )

    return Config(raw=raw, path=cfg_path, regions=regions, secrets=secrets)
