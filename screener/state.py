"""Histórico de alertas y del ranking, y clasificación de qué se reenvía.

El corte del §7 es estable día a día: si hoy pasan 51 nombres, mañana pasarán
unos 51 de los cuales ~45 son los mismos. Reenviarlos todos convierte el canal
en ruido. Aquí se decide qué llega:

- **nueva**: no había avisado de ella nunca, o salió del corte y ha vuelto a
  entrar tras estar fuera. `cooldown_days` mide cuánto ha de estar fuera.
- **mejora**: ya avisada, pero ha subido de forma significativa — el score gana
  `resurface.score_delta` puntos, o escala `resurface.rank_jump` puestos en el
  ranking de la región respecto a cuando se avisó.
- **silenciada**: sigue en el corte más o menos donde estaba. No se manda.

Además se guarda un snapshot del ranking completo de cada corrida. Sirve a la
comparación de arriba y es la fuente del histórico del panel de Streamlit.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

NEW = "nueva"
IMPROVED = "mejora"
SILENCED = "silenciada"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    symbol      TEXT NOT NULL,
    region      TEXT NOT NULL,
    alerted_on  TEXT NOT NULL,
    a_pct       REAL,
    b_pct       REAL,
    score       REAL,
    regime      REAL,
    rank        INTEGER,
    kind        TEXT,
    PRIMARY KEY (symbol, region, alerted_on)
);
CREATE INDEX IF NOT EXISTS alerts_by_symbol ON alerts (region, symbol, alerted_on);

CREATE TABLE IF NOT EXISTS snapshots (
    region      TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    snapshot_on TEXT NOT NULL,
    a_pct       REAL,
    b_pct       REAL,
    score       REAL,
    rank        INTEGER,
    passed      INTEGER,
    PRIMARY KEY (region, symbol, snapshot_on)
);
CREATE INDEX IF NOT EXISTS snapshots_by_date ON snapshots (region, snapshot_on);

CREATE TABLE IF NOT EXISTS runs (
    region        TEXT NOT NULL,
    run_on        TEXT NOT NULL,
    regime        REAL,
    regime_detail TEXT,
    universe_size INTEGER,
    gated         INTEGER,
    scored        INTEGER,
    alerts        INTEGER,
    run_at        TEXT,
    PRIMARY KEY (region, run_on)
);
"""


@dataclass
class AlertDecision:
    """Qué hacer con una alerta que ya pasó el corte del §7."""

    kind: str
    detail: str = ""

    @property
    def should_send(self) -> bool:
        return self.kind in (NEW, IMPROVED)


class AlertState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Añade columnas nuevas a una base preexistente sin perder el histórico."""
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(alerts)")}
        for column, ddl in (("rank", "INTEGER"), ("kind", "TEXT")):
            if column not in existing:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {column} {ddl}")

        # `run_on` es solo la fecha; `run_at` guarda el instante exacto para que
        # el panel pueda avisar de una región que dejó de actualizarse.
        en_runs = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
        if "run_at" not in en_runs:
            conn.execute("ALTER TABLE runs ADD COLUMN run_at TEXT")

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------
    def last_alert(self, symbol: str, region: str, since: date) -> sqlite3.Row | None:
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT alerted_on, score, rank FROM alerts "
                "WHERE region = ? AND symbol = ? AND alerted_on >= ? "
                "ORDER BY alerted_on DESC LIMIT 1",
                (region, symbol, since.isoformat()),
            ).fetchone()

    def in_cooldown(self, symbol: str, region: str, cooldown_days: int, today: date | None = None) -> bool:
        if cooldown_days <= 0:
            return False
        today = today or date.today()
        return self.last_alert(symbol, region, today - timedelta(days=cooldown_days)) is not None

    def _alert_still_standing(self, symbol: str, region: str) -> sqlite3.Row | None:
        """La última alerta, por vieja que sea, si el valor no ha salido del corte desde ella.

        Sin esto el cooldown es una ventana fija: pasados `cooldown_days` el
        registro se cae de `last_alert` y un nombre que lleva semanas plantado en
        el corte vuelve a contar como nuevo. Como la primera tanda entra junta,
        expira junta, y el canal recibe una ráfaga cada `cooldown_days + 1` días
        con valores que no se han movido. Medido en producción: 46 alertas el
        24-ago y 33 el 4-sep, con el corte estable en ~44 todo el intervalo.
        """
        previous = self.last_alert(symbol, region, date.min)
        if previous is None:
            return None
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT"
                " (SELECT COUNT(DISTINCT snapshot_on) FROM snapshots"
                "   WHERE region = ? AND snapshot_on > ?) AS corridas,"
                " (SELECT COUNT(*) FROM snapshots"
                "   WHERE region = ? AND symbol = ? AND snapshot_on > ? AND passed = 1) AS en_el_corte",
                (region, previous["alerted_on"], region, symbol, previous["alerted_on"]),
            ).fetchone()
        # Estar ausente del snapshot cuenta como haber salido: un valor que no
        # llega a puntuarse (gate fallado, datos que faltan) no sigue en el corte.
        # Y sin snapshots posteriores no hay prueba de continuidad, así que se
        # respeta el cooldown clásico en vez de silenciar a ciegas.
        if not row["corridas"]:
            return None
        return previous if row["en_el_corte"] == row["corridas"] else None

    # ------------------------------------------------------------------
    # Clasificación
    # ------------------------------------------------------------------
    def classify(
        self,
        result,
        rank: int,
        *,
        cooldown_days: int,
        score_delta: float,
        rank_jump: int,
        today: date | None = None,
    ) -> AlertDecision:
        today = today or date.today()
        if cooldown_days <= 0:
            return AlertDecision(NEW, "sin cooldown configurado")

        previous = self.last_alert(result.symbol, result.region, today - timedelta(days=cooldown_days))
        if previous is None:
            # Fuera de la ventana, pero puede llevar ahí desde el aviso: entonces
            # no es una entrada nueva, es el mismo nombre de siempre.
            previous = self._alert_still_standing(result.symbol, result.region)
        if previous is None:
            return AlertDecision(NEW, f"nueva en el corte (puesto {rank})")

        gained = result.score_final - (previous["score"] or 0.0)
        if gained >= score_delta:
            return AlertDecision(
                IMPROVED,
                f"score {gained:+.1f} desde el aviso del {previous['alerted_on']}",
            )

        previous_rank = previous["rank"]
        if previous_rank is not None and previous_rank - rank >= rank_jump:
            return AlertDecision(
                IMPROVED,
                f"sube {previous_rank - rank} puestos (del {previous_rank} al {rank})",
            )

        return AlertDecision(
            SILENCED,
            f"ya avisada el {previous['alerted_on']}, sin mejora relevante ({gained:+.1f})",
        )

    # ------------------------------------------------------------------
    # Escritura
    # ------------------------------------------------------------------
    def record(self, result, today: date | None = None, *, rank: int | None = None, kind: str = NEW) -> None:
        today = today or date.today()
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO alerts "
                "(symbol, region, alerted_on, a_pct, b_pct, score, regime, rank, kind) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.symbol,
                    result.region,
                    today.isoformat(),
                    result.a_pct,
                    result.b_pct,
                    result.score_final,
                    result.regime,
                    rank,
                    kind,
                ),
            )
            conn.commit()

    def record_snapshot(self, run, today: date | None = None) -> int:
        """Guarda el ranking completo de una corrida. Base del histórico del panel."""
        today = today or run.asof
        rows = [
            (
                run.region,
                result.symbol,
                today.isoformat(),
                result.a_pct,
                result.b_pct,
                result.score_final,
                rank,
                int(result.alert),
            )
            for rank, result in enumerate(run.ranked(), start=1)
        ]
        if not rows:
            return 0
        with closing(self._connect()) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO snapshots "
                "(region, symbol, snapshot_on, a_pct, b_pct, score, rank, passed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        return len(rows)

    def record_run(self, run, today: date | None = None) -> None:
        """Metadatos de la corrida: KPIs y régimen del día, para el panel."""
        today = today or run.asof
        regime = getattr(run, "regime", None)
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(region, run_on, regime, regime_detail, universe_size, gated, scored, alerts, run_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.region,
                    today.isoformat(),
                    regime.multiplier if regime else None,
                    regime.detail if regime else None,
                    run.universe_size,
                    len(run.results) + len(run.dropped_low_coverage),
                    run.scored,
                    len(run.alerts),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Lectura para el panel
    # ------------------------------------------------------------------
    def runs(self, region: str) -> list[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT * FROM runs WHERE region = ? ORDER BY run_on", (region,)
            ).fetchall()

    def regions(self) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT DISTINCT region FROM runs ORDER BY region").fetchall()
        return [row["region"] for row in rows]

    def recent(self, region: str, days: int = 30, today: date | None = None) -> list[sqlite3.Row]:
        cutoff = ((today or date.today()) - timedelta(days=days)).isoformat()
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT symbol, alerted_on, a_pct, b_pct, score, rank, kind FROM alerts "
                "WHERE region = ? AND alerted_on >= ? ORDER BY alerted_on DESC, score DESC",
                (region, cutoff),
            ).fetchall()

    def history(self, region: str, symbol: str | None = None) -> list[sqlite3.Row]:
        query = (
            "SELECT symbol, snapshot_on, a_pct, b_pct, score, rank, passed FROM snapshots "
            "WHERE region = ?"
        )
        params: list = [region]
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY snapshot_on"
        with closing(self._connect()) as conn:
            return conn.execute(query, params).fetchall()

    def snapshot_dates(self, region: str) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT DISTINCT snapshot_on FROM snapshots WHERE region = ? ORDER BY snapshot_on DESC",
                (region,),
            ).fetchall()
        return [row["snapshot_on"] for row in rows]
