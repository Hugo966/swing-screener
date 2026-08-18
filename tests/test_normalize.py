from __future__ import annotations

import pandas as pd
import pytest

from screener.normalize import NEUTRAL, percentile_by_group, percentile_rank


def test_percentile_rank_orders_ascending():
    ranked = percentile_rank(pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}))
    assert ranked["d"] == 100.0
    assert ranked["a"] == 25.0
    assert ranked["a"] < ranked["b"] < ranked["c"] < ranked["d"]


def test_percentile_rank_inverts_when_lower_is_better():
    """Deuda, dilución y valoración: la métrica devuelve el mal, aquí se invierte."""
    values = pd.Series({"sin_deuda": 0.0, "poca": 1.0, "mucha": 5.0})
    ranked = percentile_rank(values, higher_is_better=False)
    assert ranked["sin_deuda"] > ranked["poca"] > ranked["mucha"]
    assert ranked["sin_deuda"] == 100.0


def test_percentile_rank_keeps_missing_as_nan():
    ranked = percentile_rank(pd.Series({"a": 1.0, "b": None, "c": 3.0}))
    assert pd.isna(ranked["b"])
    # el ausente no participa: a y c se reparten el rango completo
    assert ranked["a"] == 50.0
    assert ranked["c"] == 100.0


def test_percentile_by_group_ranks_within_sector():
    """Un margen mediocre en software puede ser excelente en distribución."""
    values = pd.Series({f"tech{i}": float(i) for i in range(10)})
    values = pd.concat([values, pd.Series({f"ind{i}": float(i) * 100 for i in range(10)})])
    groups = pd.Series(
        {**{f"tech{i}": "Technology" for i in range(10)},
         **{f"ind{i}": "Industrials" for i in range(10)}}
    )

    ranked = percentile_by_group(values, groups, min_group_size=8)
    # el mejor de cada sector percentila 100 aunque sus escalas no se parezcan
    assert ranked["tech9"] == 100.0
    assert ranked["ind9"] == 100.0
    assert ranked["tech0"] == ranked["ind0"]


def test_percentile_by_group_falls_back_for_small_sectors():
    """Un sector con 2 nombres no da percentil informativo: se usa el pool regional."""
    values = pd.Series({f"tech{i}": float(i) for i in range(10)})
    values = pd.concat([values, pd.Series({"tiny1": 100.0, "tiny2": 200.0})])
    groups = pd.Series(
        {**{f"tech{i}": "Technology" for i in range(10)}, "tiny1": "Energy", "tiny2": "Energy"}
    )

    ranked = percentile_by_group(values, groups, min_group_size=8)
    # tiny2 es el valor más alto de TODO el pool, no solo de su sector de 2
    assert ranked["tiny2"] == 100.0
    assert ranked["tiny1"] == pytest.approx(100.0 * 11 / 12)


def test_neutral_is_the_midpoint():
    assert NEUTRAL == 50.0
