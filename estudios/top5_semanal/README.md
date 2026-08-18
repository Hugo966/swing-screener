# Study: weekly top N, exit rules, and comparison against the index

Point-in-time simulation of a simple strategy on top of the screener: every
Monday the universe is scored, **€100 is bought of each stock in the top N by
final score**, and nothing already held is bought again.

Five parts, in the order they were done:

1. Five **exit rules** with top 5.
2. Three portfolio sizes (**top 3 / 5 / 10**) without stops, against the index.
3. **What separates the winners**: return by rank and information coefficient of
   each metric.
4. **Effect of changing the weights** off the back of part 3. *(Reverted — see
   part 5.)*
5. **Validation across four regions** and the final decision on the weights.

Parts 1-3 on 2026-08-12 (region `us`), part 4 on 2026-08-13, part 5 on
2026-08-14/18 across `us`, `europe_dev`, `emerging` and `korea`. 135-136 weeks
from Jan-2024 to Aug-2026 in each region.

> **Current state: v1 weights, frozen.** Every file in this directory is generated
> with the v1 weights, which are the ones in `config.yaml`. The v2 experiment from
> part 4 was reverted on 2026-08-14 after it failed to replicate outside the US.
> The `pesos_v1/` folder is the snapshot of that comparison and is kept purely as
> a historical record: it is no longer "the other version", because v1 *is* the
> version.

---

## Part 5 · Validation across four regions and the final decision

The question was whether the weight change from part 4 was a general improvement
or an artifact of the data it was derived from. The full cross-sectional panel was
built in the other three regions to find out.

### The weight change was overfitting

Per-metric IC **does not depend on the weights** — it is computed on each metric's
percentile, and the weights only affect `score` and `puesto` (rank) — so the IC
tables from all four regions are directly comparable.

Combining the four with inverse-variance weighting and measuring disagreement with
**Cochran's Q** (`scripts/metaanalisis_regiones.py`):

| Metric | Pooled IC | t | Q | Verdict |
|---|---|---|---|---|
| A6 momentum / ATR% | +0.036 | +2.91 | 0.72 | **replicates** |
| A1 multi-window RS | +0.035 | +2.13 | 0.21 | **replicates** |
| B10 EV valuation | +0.025 | +2.06 | 3.52 | borderline (t=2.00 without Korea) |
| A4 sector momentum | −0.033 | −2.27 | **11.57** | contradicts itself across regions |
| B5 ROIC | −0.016 | −1.83 | **11.61** | contradicts |
| A8 earnings reaction | +0.016 | +1.60 | **10.30** | contradicts |

Critical Q at 5% is 7.81 with four regions. **A4, which part 4 had cut from 10 to
5, is the most self-contradictory metric of all**: it measures −0.060 in the US but
+0.079 in Europe. Its pooled t of −2.27 looks significant and is not, because it
averages values of opposite sign. A3, the other one that had been changed, comes
out homogeneous (Q=5.80) but insignificant (t=1.64).

### Why the weights are not tuned

The dispersion of IC across metrics decomposes as:

```
observed        sd = 0.0193
measurement     sd = 0.0133
TRUE            sd = 0.0140   ->  52% of the dispersion is real
```

Almost half the apparent difference between metrics is measurement error. The
corresponding empirical-Bayes shrinkage factor is **k=0.55**: each estimate belongs
halfway to the grand mean. Shrunk, the best metric sits at **+0.023** and the worst
at **−0.013**, with an individual standard error of 0.013. There are not two
tiers of metrics — there is a narrow continuum measured with blunt instruments.

Hence the decision: **neither prune metrics nor retune weights.** Keeping only the
two that replicate would repeat part 4's mistake at a larger scale, and would also
give up the diversification across weak signals that is precisely what protects
you when none of them is individually reliable.

### Performance by region (v1 weights)

Top 5 portfolio, each region against **its own index** and against SPY:

| Region | Strategy | Own index | Excess | t | Sharpe strat. / index |
|---|---|---|---|---|---|
| Emerging | 29.67% | EEM 22.71% | +6.96pp | +0.34 | **1.62** / 1.08 |
| US | 26.16% | SPY 20.01% | +6.15pp | +0.84 | 1.12 / **1.17** |
| Europe | 17.52% | STOXX 13.49% | +4.03pp | +0.59 | 1.16 / 1.07 |
| Korea | 39.57% | KOSPI 40.46% | −0.89pp | −0.20 | 1.18 / 1.03 |

It beats its own index in 3 of 4 regions by 4-7 points a year, but **no excess
reaches significance**: the highest t across the twelve region×size combinations
is 0.84. At the observed information ratios (0.52 in the US, 0.21 in emerging) it
would take **15 years in the US and 91 in emerging markets** to reach t=2. The sign
test, 3 of 4 positive, gives p=0.31.

Two distinctions that do separate regions:

- **The US is the only one where Sharpe gets worse** than its index (1.12 vs
  1.17): its 6 points of excess are a risk premium (vol 23.4% vs 17.0%), not skill.
  And the US is the region everything was tuned on.
- **Emerging markets is the only unambiguous improvement**: more return with
  *less* volatility than EEM. Not significant, but the only place where the sign
  points to selection rather than disguised leverage.

### Korea does not work as a validation region

86 unique stocks, median of 20 per date. 60% of its dates fall below the minimum of
30 needed to compute an IC, and **the 54 that survive are all after 2025Q3** — six
consecutive quarters without a single valid date. Its IC would measure one market
regime. It is reported in the tables but does not count as an independent vote.

### Decision

**Weights frozen at v1.** The reasoning is also written into `config.yaml`, next to
the weights themselves, which is where anyone would go to change them. From here
the rational move is not to retune on these 2.6 years but to accumulate
out-of-sample data: each additional year of live operation is worth more than any
re-optimization over the period already seen.

### Two bugs fixed along the way

- **Currency conversion in the benchmark.** `comparar_top_n.py` assumed every
  index is quoted in USD. True for SPY and EEM, false for STOXX (EUR) and KOSPI
  (KRW). It would have made the strategy look like it beat its index when what it
  actually beat was the exchange rate.
- **Standard error of the IC.** `analizar_patrones` reports `ic` as the mean over
  dates but `t` from the mean over quarters, which hold different numbers of dates.
  Deriving `se = ic/t` produced nonsense whenever the IC was near zero (ROIC in the
  US: 0.0002 instead of 0.0146, a factor of 70). `ic_trim` and `se` are now stored
  explicitly and the meta-analysis uses the coherent pair.

---

## Part 4 · Effect of the weight change (v1 → v2) — REVERTED

> **This change was reverted on 2026-08-14.** The analysis is kept because the
> lesson is the most useful result in the study: a 6-point annual improvement in
> the backtest turned out to be noise (t=1.41) and replicated in no other region.
> What follows describes the experiment as it was run; the verdict is in part 5.

After the information-coefficient analysis, five weights were changed:

| Metric | v1 | v2 | Rationale at the time |
|---|---|---|---|
| A3 RS vs sector | 12 | **17** | Best IC of the 19 (+0.060), rising with horizon |
| A4 sector momentum | 10 | **5** | Only consistently negative short-horizon IC (−0.049) |
| B1 revenue growth | 14 | **17** | IC +0.048, rising |
| B4 cash quality | 16 | **10** | Was the largest weight in panel B and measures −0.008 |
| B6 margin expansion | 9 | **12** | IC +0.050, rising |

Result over the same period:

| Portfolio | Total % v1 → v2 | Annual TWR v1 → v2 | Sharpe v1 → v2 |
|---|---|---|---|
| top 3 | +38.28% → **+47.41%** | 25.61% → **31.45%** | 1.08 → **1.24** |
| top 5 | +39.77% → **+46.50%** | 25.51% → **28.73%** | 1.10 → **1.18** |
| top 10 | +38.04% → **+44.26%** | 24.76% → **26.85%** | 1.12 → **1.17** |
| SPY | ~29% | ~19.5% | ~1.15 |

**This does NOT validate the new weights.** They were derived from looking at this
very period, so it was nearly impossible for them not to improve it: the figure is
an optimistic upper bound.

What did seem informative at the time were two things that were not obvious:

- **The improvement is large** (almost 6 points a year at top 3), not marginal. If
  moving five weights out of twenty shifted things by two tenths, the conclusion
  would be that the metrics do not matter and all the performance comes from the
  trend gate.
- **It scales with concentration** (+5.8 at top 3, +2.1 at top 10) and the Sharpe
  crosses above the index, where v1 trailed it in all three variants. A purely
  spurious fit would have no particular reason to produce that structure.

**Outcome (2026-08-14).** Neither signal held. The v2-over-v1 advantage has
**t=1.41 at top 3, 0.81 at top 5 and 0.31 at top 10**: indistinguishable from zero,
and the fact that it evaporates as the portfolio widens — from +10.4pp to +1.6pp —
is the signature of luck with few names, not of better ordering. The two metrics
whose weights were moved turned out to be among the least stable across regions.
See part 5.

---

## Part 2 · Top 3 / 5 / 10 without stops, against the index (v1 weights)

| Portfolio | Pos. | Invested | Value | Total % | **Annual TWR** | Annual IRR | Max DD | Volatility | Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| **top 3** | 82 | €8,200 | €11,338.58 | +38.28% | **25.61%** | 23.39% | −27.8% | 23.7% | 1.08 |
| SPY same flows | 82 | €8,200 | €10,534.98 | +28.48% | 19.59% | 17.84% | −22.9% | 17.0% | **1.15** |
| **top 5** | 116 | €11,600 | €16,213.65 | +39.77% | **25.51%** | 23.92% | −27.5% | 23.3% | 1.10 |
| SPY same flows | 116 | €11,600 | €15,021.10 | +29.49% | 19.52% | 18.21% | −23.0% | 17.0% | **1.15** |
| **top 10** | 185 | €18,500 | €25,538.16 | +38.04% | **24.76%** | 23.17% | −26.0% | 22.2% | 1.12 |
| SPY same flows | 185 | €18,500 | €23,936.04 | +29.38% | 19.42% | 18.31% | −22.9% | 17.0% | **1.15** |

TWR by calendar year:

| Year | top 5 | SPY | Difference |
|---|---|---|---|
| 2024 | +36.83% | +31.36% | +5.5 |
| 2025 | +12.58% | +4.34% | **+8.2** |
| 2026 | +16.72% | +15.60% | +1.1 |

**It beats the index by about 6 points a year**, and does so in all three years.
The most telling is 2025: SPY returned +4.3% and the strategy +12.6%; in 2024 and
2026 everything was rising.

**But the Sharpe is worse in all three variants** (1.08-1.12 against 1.15). It
returns more at the cost of more volatility (23% vs 17%) and deeper drawdowns.
Risk-adjusted, it lands where a slightly leveraged index would leave you: €1.35 in
SPY for every euro of the strategy gives the same volatility and marginally more
return.

**N is irrelevant**: 38.28% / 39.77% / 38.04% are the same figure within noise. The
screener repeats the same names so often that widening from 3 to 10 does not
diversify. What does change is the capital required (€8,200 vs €18,500) and,
slightly, the max drawdown (−26.0% at top 10 vs −27.8% at top 3). Choose N by
capital and drawdown, not by return.

Two return measures, because with staggered contributions they do not coincide:
**TWR** chains daily returns net of contributions and is the one comparable to an
index; **IRR** is what the money actually earned.

> Part 5 revisits this across all four regions and adds the significance test that
> is missing here: none of these excesses clears |t| = 1.

---

## Part 1 · Exit rules (top 5, v1 weights)

136 weeks, Jan-2024 → Aug-2026, €100 ticket:

| Rule | Positions | Invested | Value | P&L | % | Winners | Best | Worst |
|---|---|---|---|---|---|---|---|---|
| **No stop** | 116 | €11,600 | €16,213.65 | **+€4,613.65** | **+39.77%** | 70% | +460% | −53.7% |
| Fixed +20% | 208 | €20,800 | €22,971.07 | +€2,171.07 | +10.44% | 79% | +38% | −53.9% |
| Hybrid −8%/ATR | 172 | €17,200 | €18,003.26 | +€803.26 | +4.67% | 47% | +144% | −28.7% |
| Trailing 15% | 167 | €16,700 | €17,459.24 | +€759.24 | +4.55% | 44% | +153% | −28.7% |
| Trailing ATR k=3 | 167 | €16,700 | €17,334.62 | +€634.62 | +3.80% | 46% | +144% | −47.0% |

**Conclusion: any stop cuts the tail of winners, and the tail is where the return
comes from.** 90th percentile of per-position return: +127% with no stop, +25% to
+34% with any rule. The five best positions (FIX +460%, STRL +288%, VRT +283%,
AGX +246%, KGC +220%) contribute 32% of all profit.

Two nuances that read wrong if you only look at the total %:

- The **fixed +20%** has the highest hit rate (79%) and still returns a quarter of
  doing nothing: it caps gains at +38% and lets losses run to −54%. Being right
  often is not the same as making money.
- **ATR added nothing**, contrary to expectations when it was designed (spec §6
  suggests reusing ATR% for the stop): it trails the simple percentage trailing
  stop and has a far worse worst case (−47% vs −28.7%). At k=3, a volatile name
  like SNDK needs 41% of headroom, and by the time the stop triggers almost
  everything has been given back.

Stops do cap the worst case: −28.7% with the hybrid against −53.7% with nothing.
The price of that protection was 35 points of return.

---

## Part 3 · What separates the winners (v1 weights)

Built on `panel_transversal.parquet`: the ~700 stocks of each Monday with their
rank and the percentile of all 20 metrics.

**Rank within the top carries no information.** Mean 63-session return by band:

| Band | v1 | v2 |
|---|---|---|
| 1-3 | +7.75% | +8.66% |
| 6-10 | +8.40% | +10.09% |
| 21-50 | +6.64% | +7.42% |
| 51-100 | +5.02% | +5.10% |
| >100 | +4.66% | +4.60% |

The score separates the top ~50 from the rest, but **within the top 20 it does not
order**: rank-return correlation of −0.012 (t = −0.72) at 63 sessions, and only
half the dates with the expected sign. That is why top 3, 5 and 10 return almost
the same.

**Four metrics carry signal and one runs backwards** (IC = rank correlation within
each date, then averaged):

| Metric | 21d | 63d | 126d |
|---|---|---|---|
| A3 RS vs sector | +0.037 | +0.060 | **+0.076** |
| A6 momentum / ATR% | +0.011 | +0.037 | **+0.080** |
| B6 margin expansion | +0.028 | +0.050 | +0.071 |
| B1 revenue growth | +0.026 | +0.048 | +0.069 |
| **A4 sector momentum** | **−0.045** | **−0.049** | +0.008 |

What is convincing here is not any isolated t-statistic but **the shape**: the
positive ones grow monotonically with the horizon, which is what a genuine slow
signal does, and four metrics do it at once.

**A3 versus A4 was the actionable finding.** They are cousins: A3 rewards *beating
your sector*, A4 rewards *being in a hot sector*. The first was the best of the 19;
the second the worst. Pick the stock that beats its sector, do not chase the
sector.

**It is not a sector effect**: recomputing the IC only among stocks in the same
sector, it holds (A3 +0.054 against +0.060 overall; A6 rises to +0.053).

**B4, the heaviest metric in panel B (16 pts), measures −0.008**: indistinguishable
from zero. Same for B5, B9 and B7. The ~39 points spread across those four bought
no measurable information. **A10 does not even appear**: it returns 1.0 for almost
everything and does not discriminate.

Caveats: results in the `fwd_hoy` column are contaminated (it measures from the
signal to today, so the horizon varies with the calendar); an IC of 0.06 explains
0.4% of variance — real but weak; and these metrics *construct* the score, so they
are not independent of it. 76 tests, ~10 independent quarterly blocks.

> **Part 5 overturned the actionable finding.** A3 and A4 turned out to be the two
> least stable metrics across regions, and the weight change built on them was
> reverted. What survived validation was A6 and A1, not A3.

---

## Why the period starts in January 2024

**You cannot go further back with Yahoo data.** Mean panel-B coverage when
reconstructing the past with `pointintime.as_of`, measured across 10 stocks:

| Date | Panel B coverage |
|---|---|
| 2021 | 10% |
| 2022 | 10% |
| 2023 | 58% |
| 2024 | 78% |
| 2025-2026 | 90% |

Yahoo serves only 4-5 annual fiscal years and ~5 quarters, so on a 2021 date there
is no already-published financial statement inside that window and the only live
metric is B3, which comes from `earnings_dates` (with ~6 years of depth). With
`coverage.min_panel_coverage` at 70%, before 2023 **every stock would be
discarded**. In the real run, 53 of ~800 per date were discarded (~6%).

B8 (revisions) always comes out at 0 by design: `eps_revisions` is a snapshot of
today without vintages, so `as_of` nulls it and its 10 points redistribute.

## Files

| File | What it is |
|---|---|
| `senales_top20_<region>.json` | **The expensive one, and the one to reuse.** Top **20** with scores, regime and number scored for each of the 135 Mondays. ~50 minutes of computation each; with it, any N ≤ 20 and any new exit rule evaluate in seconds. |
| `comparativa_top_n*.csv` | The part 2 tables, one per region. |
| `curvas_valor*.csv` | Daily euro value of each portfolio and its benchmark equivalent. Also carries the `__aporte` (contribution) column, which is required to compute returns net of contributions. |
| `resumen_reglas.csv` | The part 1 table. |
| `posiciones/*.csv` | Position-by-position detail for each rule: entry, value, return and close reason. |
| `pesos_v1/` | Snapshot of the v1/v2 comparison from part 4. Historical record: the files in this directory are already v1. |
| `panel_transversal.parquet` | Full US cross-section: 83,275 rows with the ~660 stocks of each Monday, their rank, score and the percentile of all 20 metrics, plus forward returns. Basis for parts 3 and 5. |
| `panel_transversal_{europe_dev,emerging,korea}.parquet` | The same for the other three regions (13,847 / 21,610 / 4,248 rows). Basis for part 5. |
| `information_coefficient*.csv` / `retorno_por_puesto*.csv` | Output of `analizar_patrones`, one per region. They include `ic_trim_*` and `se_*`, which are the coherent pair for pooling across regions. |
| `metaanalisis_regiones.csv` | The decisive table from part 5: pooled IC, t and Cochran's Q per metric. |
| `comparativa_regiones_ic63.csv` | 63-day IC of every metric in every region, side by side. |
| `scripts/exportar_transversal.py` | Generates the parquet and the signals for one region. 20-60 min depending on universe. |
| `scripts/analizar_patrones.py` | Generates part 3 from the parquet. Seconds. |
| `scripts/metaanalisis_regiones.py` | **Part 5.** Pools IC across regions and measures heterogeneity. Consult before touching any weight. |
| `scripts/comparar_regiones.py` | Side-by-side IC across regions plus the against-chance test. |
| `scripts/comparar_top_n.py` | Generates part 2 for one region. Reuses `senales_top20_<region>.json` if present. |
| `scripts/comparar_reglas.py` | Generates part 1. Scores once and simulates all five rules. |
| `scripts/simular_una_regla.py` | Single-rule version, with a configurable window (`MESES`). |

> The `.parquet` files are gitignored: ~28 MB of derived data, regenerable with
> `exportar_transversal.py`. The findings live in the committed CSVs.

Scripts expect to be run from the project root with the venv, and use the
`./.cache` directory. If the fundamentals cache has expired (7-day TTL) they will
re-download ~1,600 profiles, which takes about 12 minutes.

## Assumptions and limits

- **Survivorship bias, uncorrected**: the universe is whatever trades today. Yahoo
  does not serve historical constituents. Returns are biased upward.
- **The period is almost entirely bullish with no bear market.** Not using a stop
  always wins in a rising market; in a 2022 the table would flip, and that is
  precisely the year that cannot be simulated for the reason above. This is the
  study's most serious limitation: a momentum strategy without stops in a bear
  market is exactly the scenario the regime multiplier was meant to cover, and it
  has not been tested here.
- **Statistical power is insufficient for comparing variants.** No excess over the
  index reaches significance in any region (|t| < 0.85), and at the observed
  information ratios it would take 15-90 years. Any comparison between weight sets,
  exit rules or portfolio sizes falls inside the noise **by construction**: it is
  not that the test needs refining, it is that the period does not support it.
  Always present these differences with their t or IR alongside.
- The 6 points of annual excess are **pre-tax**. Selling an index once every many
  years and rotating 185 positions are not taxed the same way.
- 136 weekly dates are heavily overlapping: they amount to roughly **30
  independent observations**, not 136.
- Buys at the same Monday's close using that close's signal. In reality you would
  buy on Tuesday.
- Stops evaluated **on closing prices**, not intraday highs and lows. This favours
  the stop variants: a gap down would fill worse.
- No commissions or slippage. The stop variants make 156-162 closes against zero;
  at real prices their disadvantage would grow.
- Fractional shares. Conversion to euros at each date's EUR/USD rate, so the result
  includes the currency effect.
