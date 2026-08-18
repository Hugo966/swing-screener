"""Alertas por Telegram con desglose por métrica (§9).

El desglose no es opcional: cuando salte una alerta hay que poder ver *por qué*,
y para tunear los pesos hace falta saber qué percentiló cada métrica y cuánto
aportó al panel.

Se usa `parse_mode=HTML` en vez de MarkdownV2: en Markdown hay que escapar
dieciséis caracteres que aparecen constantemente en cifras (`.`, `-`, `+`, `(`),
y un solo escape olvidado hace que Telegram rechace el mensaje entero.
"""

from __future__ import annotations

import html
import logging
import re

import requests

from screener.metrics import registry
from screener.models import PanelBreakdown, TickerResult

log = logging.getLogger(__name__)

TELEGRAM_LIMIT = 4096
_API = "https://api.telegram.org/bot{token}/sendMessage"
_TAGS = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------
def _panel_lines(breakdown: PanelBreakdown | None, title: str) -> list[str]:
    if breakdown is None:
        return [f"<b>{title}</b>: sin datos"]

    lines = [f"<b>{title}</b> — score {breakdown.raw_score:.1f}, cobertura {breakdown.coverage * 100:.0f}%"]
    for metric in sorted(breakdown.metrics, key=lambda m: m.contribution, reverse=True):
        label = html.escape(registry.get(metric.name).label)
        raw = "s/d" if metric.raw is None else f"{metric.raw:+.3g}"
        flag = " ·imp" if metric.imputed else ""
        lines.append(
            f"  <code>p{metric.percentile:3.0f}</code> ×{metric.weight:4.1f}"
            f" = {metric.contribution:5.1f}  {label} ({raw}){flag}"
        )
    return lines


def format_alert(result: TickerResult, regime_detail: str = "", decision=None) -> str:
    """Mensaje de una alerta, en HTML de Telegram."""
    name = html.escape(result.name or result.symbol)
    tag = " 👁" if result.is_watchlist else ""
    price = f"{result.price:,.2f}" if result.price else "s/d"

    # Un nombre nuevo en el corte y uno que reaparece por haber escalado mucho
    # piden atención distinta: el icono lo dice de un vistazo.
    icon = "🚀"
    if decision is not None and decision.kind == "mejora":
        icon = "📈"

    header = [
        f"{icon} <b>{html.escape(result.symbol)}</b>{tag} — {name}",
        f"{html.escape(result.region)} · {html.escape(result.sector or 's/d')} · {price}",
        "",
        f"A_pct <b>{result.a_pct:.0f}</b> · B_pct <b>{result.b_pct:.0f}</b>"
        f" · régimen <b>{result.regime:.2f}</b>",
        f"score final <b>{result.score_final:.1f}</b>",
    ]
    if decision is not None and decision.detail:
        header.append(f"<b>{html.escape(decision.kind)}</b> — {html.escape(decision.detail)}")
    if regime_detail:
        header.append(f"<i>{html.escape(regime_detail)}</i>")
    header.append("")

    body = _panel_lines(result.momentum, "Panel A · Momentum")
    body.append("")
    body.extend(_panel_lines(result.quality, "Panel B · Calidad"))

    return "\n".join(header + body)


def format_summary(region: str, run) -> str:
    """Resumen de la corrida, se manda aunque no haya ninguna alerta."""
    regime = run.regime
    lines = [
        f"📊 <b>{html.escape(region)}</b> — {run.asof.isoformat()}",
        f"universo {run.universe_size} · puntuados {run.scored} · alertas {len(run.alerts)}",
    ]
    if regime is not None:
        lines.append(f"régimen <b>{regime.multiplier:.2f}</b> — {html.escape(regime.detail)}")
    return "\n".join(lines)


def to_plain_text(message: str) -> str:
    """Misma información sin etiquetas, para el `--dry-run` por consola."""
    return html.unescape(_TAGS.sub("", message))


def split_message(message: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Trocea respetando saltos de línea; Telegram rechaza más de 4096 caracteres."""
    if len(message) <= limit:
        return [message]

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in message.split("\n"):
        # Una línea suelta más larga que el límite se parte en crudo.
        while len(line) > limit:
            if current:
                chunks.append("\n".join(current))
                current, length = [], 0
            chunks.append(line[:limit])
            line = line[limit:]
        if length + len(line) + 1 > limit and current:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Envío
# ---------------------------------------------------------------------------
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, watchlist_chat_id: str | None = None) -> None:
        self.token = token
        self.chat_id = chat_id
        self.watchlist_chat_id = watchlist_chat_id or chat_id

    def send(self, message: str, *, to_watchlist: bool = False, timeout: float = 20.0) -> bool:
        chat_id = self.watchlist_chat_id if to_watchlist else self.chat_id
        ok = True
        for chunk in split_message(message):
            try:
                response = requests.post(
                    _API.format(token=self.token),
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=timeout,
                )
                if response.status_code != 200:
                    log.error("Telegram %s: %s", response.status_code, response.text[:300])
                    ok = False
            except requests.RequestException as exc:
                log.error("Telegram no respondió: %s", exc)
                ok = False
        return ok


def build_notifier(cfg) -> TelegramNotifier | None:
    """None si faltan credenciales: el runner cae a consola en vez de fallar."""
    secrets = cfg.secrets
    if not secrets.telegram_ready:
        return None
    separate = bool(cfg.alerting["watchlist"].get("separate_channel"))
    watchlist_chat = secrets.telegram_chat_id_watchlist if separate else None
    return TelegramNotifier(secrets.telegram_token, secrets.telegram_chat_id, watchlist_chat)
