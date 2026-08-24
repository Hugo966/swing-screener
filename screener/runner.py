"""CLI / entrypoint EOD.

No es un demonio: se lanza por cron tras el cierre de cada región. `run.schedule`
y `close_time_utc` documentan cuándo debería correr; `--force` se salta la
comprobación para poder probar a cualquier hora.

    python -m screener.runner --region us --dry-run
    python -m screener.runner --region us --explain AMD
    python -m screener.runner --region us          # envía por Telegram
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, time, timezone
from pathlib import Path

import pandas as pd

from screener import alerts as alerts_mod
from screener.config import Config, load_config
from screener.engine import RegionRun, run_region
from screener.models import Region
from screener.state import AlertState

log = logging.getLogger("screener")


# ---------------------------------------------------------------------------
# Salida por consola
# ---------------------------------------------------------------------------
def print_ranking(run: RegionRun, limit: int = 25) -> None:
    regime = run.regime
    print(f"\n=== {run.region} · {run.asof.isoformat()} ===")
    print(f"universo {run.universe_size} · puntuados {run.scored} · alertas {len(run.alerts)}")
    if regime is not None:
        print(f"régimen {regime.multiplier:.3f} — {regime.detail}")
    if run.dropped_low_coverage:
        print(f"descartados por cobertura: {len(run.dropped_low_coverage)}")

    print(f"\n{'':2} {'SÍMBOLO':<8} {'A_pct':>6} {'B_pct':>6} {'SCORE':>7}  {'SECTOR':<24} MOTIVO")
    for result in run.ranked()[:limit]:
        mark = "🚀" if result.alert else ("👁" if result.is_watchlist else "  ")
        print(
            f"{mark:2} {result.symbol:<8} {result.a_pct:6.1f} {result.b_pct:6.1f}"
            f" {result.score_final:7.1f}  {(result.sector or '')[:24]:<24} {result.alert_reason}"
        )


def print_explain(run: RegionRun, symbol: str) -> int:
    match = next((r for r in run.results if r.symbol.upper() == symbol.upper()), None)
    if match is None:
        print(f"{symbol} no está entre los {run.scored} valores puntuados de {run.region}.")
        failed = [g for g in run.gate_log if g.symbol.upper() == symbol.upper() and not g.passed]
        if failed:
            gate = failed[-1]
            print(f"Descartado en el gate '{gate.failed_gate}': {gate.detail}")
        elif symbol.upper() in {s.upper() for s in run.dropped_low_coverage}:
            print("Descartado por cobertura de datos insuficiente.")
        return 1

    detail = run.regime.detail if run.regime else ""
    print(alerts_mod.to_plain_text(alerts_mod.format_alert(match, detail)))
    print(f"\nDecisión: {'ALERTA' if match.alert else 'sin alerta'} — {match.alert_reason}")
    return 0


def write_csv(run: RegionRun, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run.region}_{run.asof.isoformat()}.csv"

    rows = []
    for result in run.ranked():
        row = {
            "symbol": result.symbol,
            "name": result.name,
            "sector": result.sector,
            "watchlist": result.is_watchlist,
            "a_pct": round(result.a_pct, 2),
            "b_pct": round(result.b_pct, 2),
            "regime": round(result.regime, 4),
            "score_final": round(result.score_final, 2),
            "alert": result.alert,
            "reason": result.alert_reason,
        }
        for breakdown in (result.momentum, result.quality):
            if breakdown is None:
                continue
            for metric in breakdown.metrics:
                row[f"{metric.name}__pct"] = round(metric.percentile, 1)
                row[f"{metric.name}__raw"] = metric.raw
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Envío
# ---------------------------------------------------------------------------
def dispatch(run: RegionRun, cfg: Config, *, dry_run: bool, send_summary: bool) -> int:
    notifier = None if dry_run else alerts_mod.build_notifier(cfg)
    if not dry_run and notifier is None:
        log.warning("faltan TELEGRAM_TOKEN/TELEGRAM_CHAT_ID en .env: se imprime por consola")

    state = AlertState(cfg.run["state_db"])
    cooldown = int(cfg.alerting.get("cooldown_days", 0))
    resurface = cfg.alerting.get("resurface") or {}
    regime_detail = run.regime.detail if run.regime else ""

    # El puesto en el ranking de la región es lo que permite detectar "ha
    # escalado mucho" en una que ya se avisó.
    ranks = {result.symbol: rank for rank, result in enumerate(run.ranked(), start=1)}

    sent = 0
    silenced = 0
    for result in run.alerts:
        decision = state.classify(
            result,
            ranks.get(result.symbol, 0),
            cooldown_days=cooldown,
            score_delta=float(resurface.get("score_delta", 0.0)),
            rank_jump=int(resurface.get("rank_jump", 0)),
        )
        if not decision.should_send:
            log.debug("%s silenciada: %s", result.symbol, decision.detail)
            silenced += 1
            continue

        message = alerts_mod.format_alert(result, regime_detail, decision=decision)
        if notifier is None:
            print("\n" + alerts_mod.to_plain_text(message))
        else:
            if not notifier.send(message, to_watchlist=result.is_watchlist):
                continue
        # En dry-run no se anota la alerta: si se anotara, una prueba consumiría
        # el cooldown y silenciaría la corrida real de después.
        if not dry_run:
            state.record(result, rank=ranks.get(result.symbol), kind=decision.kind)
        sent += 1

    # El snapshot y los KPIs se guardan siempre, también en dry-run: son el
    # histórico del panel y la referencia contra la que se mide la mejora de mañana.
    state.record_snapshot(run)
    state.record_run(run)
    if silenced:
        log.info("%d alertas silenciadas (ya avisadas, sin mejora relevante)", silenced)

    if send_summary:
        summary = alerts_mod.format_summary(run.region, run)
        if notifier is None:
            print("\n" + alerts_mod.to_plain_text(summary))
        else:
            notifier.send(summary)

    return sent


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------
def market_closed(region: Region, now: datetime | None = None) -> bool:
    """¿Ya cerró la región hoy? Aproximación por hora UTC de cierre."""
    now = now or datetime.now(timezone.utc)
    hours, minutes = (int(x) for x in region.close_time_utc.split(":"))
    return now.timetz() >= time(hours, minutes, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="screener", description=__doc__)
    parser.add_argument("--region", action="append", help="clave de región; repetible. Por defecto, las habilitadas")
    parser.add_argument("--config", default=None, help="ruta a config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="no envía a Telegram: consola + CSV")
    parser.add_argument("--explain", metavar="TICKER", help="desglose completo de un valor")
    parser.add_argument("--force", action="store_true", help="ignora la hora de cierre")
    parser.add_argument("--summary", action="store_true", help="manda también el resumen de la corrida")
    parser.add_argument("--top", type=int, default=25, help="filas del ranking por consola")
    parser.add_argument("--no-csv", action="store_true", help="no escribir el CSV")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # yfinance emite como ERROR cosas rutinarias ("No earnings dates found" de un
    # ADR sin cobertura), que en un universo de mil valores tapan los errores de
    # verdad. Con -v vuelven a verse.
    if not args.verbose:
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    cfg = load_config(args.config)
    if args.region:
        regions = [cfg.region(key) for key in args.region]
    else:
        regions = cfg.enabled_regions()
    if not regions:
        log.error("no hay regiones habilitadas en %s", cfg.path)
        return 1

    exit_code = 0
    for region in regions:
        if not args.force and not market_closed(region):
            log.info("%s aún no ha cerrado (cierre %s UTC); usa --force", region.key, region.close_time_utc)
            continue

        log.info("=== región %s (fase %d, %d métricas de momentum, %d de calidad) ===",
                 region.key, region.phase,
                 len(region.weights["momentum"]), len(region.weights["quality"]))
        run = run_region(region, cfg)

        if args.explain:
            exit_code |= print_explain(run, args.explain)
            continue

        print_ranking(run, args.top)
        if not args.no_csv:
            path = write_csv(run, cfg.run["output_dir"])
            print(f"\nCSV: {path}")

        sent = dispatch(run, cfg, dry_run=args.dry_run, send_summary=args.summary)
        # En --dry-run `dispatch` solo imprime: decir "enviadas" hace que el log
        # mienta justo cuando es lo único que se mira en una corrida desatendida.
        log.info("región %s: %d alertas %s", region.key, sent,
                 "mostradas" if args.dry_run else "enviadas")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
