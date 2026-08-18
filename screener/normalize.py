"""Normalización a percentil cross-sectional (§4).

En absoluto casi todo cae entre 40 y 70 y se pierde discriminación. En percentil
la pregunta que responde el score es "¿está en el top 5% del mercado en esto?",
que es la que importa en momentum.

- Momentum -> percentil contra **todo el universo de la región**.
- Fundamentales -> percentil contra **su sector dentro de la región**: un margen
  del 12% es excelente en distribución y mediocre en software.

Consecuencia: no se puede puntuar un ticker aislado. Estas funciones trabajan
sobre la región entera.
"""

from __future__ import annotations

import pandas as pd

NEUTRAL = 50.0


def percentile_rank(values: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    """Percentil [0,100] dentro de la serie, ignorando ausentes.

    `higher_is_better=False` invierte: la métrica devuelve la magnitud de algo
    malo (deuda, dilución, valoración) y aquí se convierte en "cuanto más alto,
    mejor" como el resto.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    ranked = numeric.rank(pct=True, ascending=higher_is_better, na_option="keep") * 100.0
    return ranked


def percentile_by_group(
    values: pd.Series,
    groups: pd.Series,
    *,
    higher_is_better: bool = True,
    min_group_size: int = 8,
) -> pd.Series:
    """Percentil dentro del grupo (sector).

    Un sector con menos de `min_group_size` nombres no da un percentil
    informativo: esos valores caen al pool completo de la región.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(index=numeric.index, dtype="float64")

    counts = numeric.groupby(groups).transform("count")
    big_enough = counts >= min_group_size

    if big_enough.any():
        subset = numeric[big_enough]
        result.loc[subset.index] = (
            subset.groupby(groups[big_enough])
            .rank(pct=True, ascending=higher_is_better, na_option="keep")
            * 100.0
        )

    fallback = ~big_enough
    if fallback.any():
        result.loc[numeric.index[fallback]] = percentile_rank(
            numeric, higher_is_better=higher_is_better
        )[fallback]

    return result


def fill_neutral(percentiles: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Imputa los ausentes al percentil neutro y marca cuáles se imputaron."""
    imputed = percentiles.isna()
    return percentiles.fillna(NEUTRAL), imputed
