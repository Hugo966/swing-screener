"""Qué alertas se reenvían: solo las nuevas y las que mejoran de verdad."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from screener.models import TickerResult
from screener.state import IMPROVED, NEW, SILENCED, AlertState

TODAY = date(2025, 3, 10)
RULES = dict(cooldown_days=10, score_delta=8.0, rank_jump=20)


def result(symbol="AMD", score=70.0, a_pct=85.0, b_pct=85.0):
    r = TickerResult(symbol=symbol, region="us", a_pct=a_pct, b_pct=b_pct, regime=0.9)
    r.score_final = score
    r.alert = True
    return r


@pytest.fixture
def state(tmp_path):
    return AlertState(tmp_path / "state.sqlite")


def test_first_time_in_the_cut_is_new(state):
    decision = state.classify(result(), rank=5, today=TODAY, **RULES)
    assert decision.kind == NEW
    assert decision.should_send


def test_same_name_barely_moving_is_silenced(state):
    """El caso del día siguiente: 45 de los 51 son los mismos y no se han movido."""
    state.record(result(score=70.0), TODAY, rank=5, kind=NEW)

    decision = state.classify(result(score=72.0), rank=6, today=TODAY + timedelta(days=1), **RULES)
    assert decision.kind == SILENCED
    assert not decision.should_send
    assert "sin mejora relevante" in decision.detail


def test_a_big_score_gain_resurfaces_it(state):
    state.record(result(score=70.0), TODAY, rank=30, kind=NEW)

    decision = state.classify(result(score=79.0), rank=28, today=TODAY + timedelta(days=2), **RULES)
    assert decision.kind == IMPROVED
    assert decision.should_send
    assert "+9.0" in decision.detail


def test_a_big_rank_jump_resurfaces_it_even_without_score_gain(state):
    """Si el mercado entero baja, escalar 40 puestos importa aunque el score no suba."""
    state.record(result(score=70.0), TODAY, rank=48, kind=NEW)

    decision = state.classify(result(score=72.0), rank=8, today=TODAY + timedelta(days=3), **RULES)
    assert decision.kind == IMPROVED
    assert "sube 40 puestos" in decision.detail


def test_dropping_in_the_ranking_never_resurfaces(state):
    state.record(result(score=75.0), TODAY, rank=5, kind=NEW)

    decision = state.classify(result(score=71.0), rank=40, today=TODAY + timedelta(days=1), **RULES)
    assert decision.kind == SILENCED


def test_it_becomes_new_again_after_the_cooldown(state):
    state.record(result(score=70.0), TODAY, rank=5, kind=NEW)

    decision = state.classify(result(score=70.0), rank=5, today=TODAY + timedelta(days=11), **RULES)
    assert decision.kind == NEW


def test_zero_cooldown_sends_everything(state):
    state.record(result(score=70.0), TODAY, rank=5, kind=NEW)
    decision = state.classify(
        result(score=70.0), rank=5, today=TODAY, cooldown_days=0, score_delta=8.0, rank_jump=20
    )
    assert decision.kind == NEW


def test_classification_is_per_region(state):
    state.record(result(score=70.0), TODAY, rank=5, kind=NEW)

    other = result(score=70.0)
    other.region = "europe_dev"
    assert state.classify(other, rank=5, today=TODAY, **RULES).kind == NEW


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------
class FakeRun:
    def __init__(self, results, region="us", asof=TODAY):
        self.results = results
        self.region = region
        self.asof = asof

    def ranked(self):
        return sorted(self.results, key=lambda r: r.score_final, reverse=True)


def test_snapshot_stores_the_whole_ranking_with_positions(state):
    run = FakeRun([result("A", 90.0), result("B", 50.0), result("C", 70.0)])
    assert state.record_snapshot(run) == 3

    rows = {row["symbol"]: row["rank"] for row in state.history("us")}
    assert rows == {"A": 1, "C": 2, "B": 3}


def test_snapshot_is_idempotent_within_a_day(state):
    """Varias corridas el mismo día no duplican el histórico."""
    run = FakeRun([result("A", 90.0)])
    state.record_snapshot(run)
    state.record_snapshot(run)
    assert len(state.history("us")) == 1


def test_history_can_be_filtered_by_symbol(state):
    state.record_snapshot(FakeRun([result("A", 90.0), result("B", 50.0)]))
    state.record_snapshot(FakeRun([result("A", 95.0), result("B", 55.0)], asof=TODAY + timedelta(days=1)))

    history = state.history("us", "A")
    assert [row["score"] for row in history] == [90.0, 95.0]
    assert state.snapshot_dates("us") == ["2025-03-11", "2025-03-10"]


# ---------------------------------------------------------------------------
# El eco del cooldown: expirar la ventana no basta para volver a avisar
# ---------------------------------------------------------------------------
def snapshots_in_the_cut(state, days, score=70.0):
    for offset in days:
        state.record_snapshot(FakeRun([result(score=score)], asof=TODAY + timedelta(days=offset)))


def test_a_name_that_never_left_the_cut_stays_silenced_past_the_cooldown(state):
    """Sin esto, la tanda inicial expira junta y el canal recibe una ráfaga entera."""
    state.record(result(score=70.0), TODAY, rank=5, kind=NEW)
    snapshots_in_the_cut(state, range(1, 12))

    decision = state.classify(result(score=70.0), rank=5, today=TODAY + timedelta(days=11), **RULES)
    assert decision.kind == SILENCED
    assert not decision.should_send


def test_it_is_new_again_if_it_dropped_out_and_came_back(state):
    """Salir del corte y volver sí es noticia: es una entrada nueva de verdad."""
    state.record(result(score=70.0), TODAY, rank=5, kind=NEW)
    fuera = result(score=40.0)
    fuera.alert = False
    for offset in range(1, 11):
        state.record_snapshot(FakeRun([fuera], asof=TODAY + timedelta(days=offset)))
    snapshots_in_the_cut(state, [11])

    decision = state.classify(result(score=70.0), rank=5, today=TODAY + timedelta(days=11), **RULES)
    assert decision.kind == NEW


def test_falling_out_of_the_ranking_entirely_also_counts_as_leaving(state):
    """Un valor que ni se puntúa (gate fallado) no sigue en el corte."""
    state.record(result(score=70.0), TODAY, rank=5, kind=NEW)
    for offset in range(1, 12):
        state.record_snapshot(FakeRun([result("OTRO", 90.0)], asof=TODAY + timedelta(days=offset)))

    decision = state.classify(result(score=70.0), rank=5, today=TODAY + timedelta(days=11), **RULES)
    assert decision.kind == NEW


def test_a_real_gain_still_resurfaces_a_long_standing_name(state):
    """Silenciar por continuidad no puede tapar una mejora de verdad."""
    state.record(result(score=70.0), TODAY, rank=30, kind=NEW)
    snapshots_in_the_cut(state, range(1, 12))

    decision = state.classify(result(score=80.0), rank=28, today=TODAY + timedelta(days=11), **RULES)
    assert decision.kind == IMPROVED
    assert "+10.0" in decision.detail
