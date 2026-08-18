# Stock Screener & Buy Alerts — Technical specification

> **Design note.**
> The project is built **in 2 phases**. The design (panels, normalization, regime,
> combination) is identical across both; what changes between phases is the **universe**
> and the **density of available data**. Phase 1 is built complete, with the **region
> abstraction** already in place so that Phase 2 is a matter of adding configuration, not
> rewriting the engine.
>
> - **Phase 1 — US + developed Europe.** Full design, all 20 metrics active. Dense data.
> - **Phase 2 — Emerging markets + Korea.** Reduced B panel: metrics that depend on analyst
>   revisions, earnings surprise and structured SBC/dilution are **disabled or down-weighted**
>   where the data is absent, and the remaining weights are **renormalized**. Each region has
>   its own regime benchmark and percentile pool.
>
> Weights, thresholds, universes and sources are never hardcoded: everything lives in
> `config.yaml`. Each metric is a **pure function** `(data) -> raw_value`. Percentile
> normalization and weighting happen outside the metric.

---

## 1. Objective

A program that, at each market close, walks a per-region stock universe, scores every stock on
two independent panels (momentum and quality), and **sends a Telegram alert** when a stock
clears the threshold on **both** panels at once. Underlying strategy: momentum/growth swing
trading (weeks to months of holding period).

## 2. Overall architecture

Flow per region and session:

```
1. Build the region's UNIVERSE (index/exchange lists + watchlist)
2. Apply GATES (binary filters; anything that fails is discarded)
3. Compute raw METRICS for each survivor (pure functions)
4. NORMALIZE each metric to a cross-sectional PERCENTILE:
      - momentum metrics      -> percentile vs the region's whole universe
      - fundamental metrics   -> percentile vs its sector within the region
5. PANELS: weighted sum of percentiles -> panel_raw (A and B)
      -> percentile of panel_raw within the region -> panel_pct (A_pct, B_pct)
6. Region MARKET REGIME -> continuous 0.5..1.0 multiplier
7. Alert DECISION (see §7)
8. Telegram ALERTS
```

**Key consequence:** percentiles are cross-sectional, so **a single ticker cannot be scored in
isolation**. The engine works **in batch per region**: it computes the whole universe first,
then ranks. `engine.py` orchestrates the entire universe, not one symbol.

**Region abstraction** (`models.py`): a `Region` defines its universe list, its regime
benchmark, its percentile pool, its `DataProvider` and its **set of active metrics** (the last
of which is what enables Phase 2's reduced B panel without touching code).

## 3. Universe and gates

Each region's universe combines a **broad universe** (index/exchange constituents) with a
**personal watchlist** (a tagged subset that is always scored; see §7 for its own threshold).

**Gates (binary, all mandatory; if one fails, discard without scoring):**

| Gate | Rule |
|---|---|
| Trend | price > MA200 **and** MA50 > MA200 |
| Size | market cap ≥ USD 2,000 M (convert currency to USD) |
| Liquidity | average daily **dollar** volume ≥ threshold (config); keeps illiquids out |
| Sector | **not** in the excluded list: banks, insurers, REITs (they break panel B: no meaningful FCF, net debt/EBITDA does not apply, ROE reads backwards) |

## 4. Percentile normalization (cross-sectional)

Each metric returns a raw value; `normalize.py` converts it to a `[0,100]` percentile:

- **Momentum → percentile vs the region's whole universe** (momentum is relative to the market).
- **Fundamentals → percentile vs the sector within the region** (a 12% margin is excellent in
  distribution and mediocre in software; 20% growth is weak in SaaS and spectacular in
  industrials). Crossing sectors or accounting regimes destroys the signal.

This replaces green/red flags and absolute scales: in absolute terms almost everything would
land between 40 and 70 and you would lose discriminating power. In percentile terms the score
answers "is this in the market's top 5% on this dimension", which is the question that matters
in momentum.

## 5. Market regime (global multiplier, per region)

`regime.py` computes a **continuous** `0.5..1.0` multiplier per region (not a step function: an
abrupt halving on crossing the MA200 creates a cliff). It combines:

- the distance from the region's benchmark to its MA200 (SPY/US, STOXX 600/Europe, KOSPI/Korea,
  MSCI EM proxy/emerging), and
- breadth: % of the region's stocks above their MA50.

`final_score = combined_panel_pct * regime`. In a real bear market nothing reaches the
threshold; in a transition it tightens progressively. This is the mechanism that protects
against *momentum crashes* (2009, 03/2020).

## 6. The two panels

Each panel's weights sum to 100. The weighted sum of percentiles produces `panel_raw`; that
`panel_raw` is then **percentiled within the region** to obtain `A_pct` / `B_pct` (§2, step 5).

### Panel A — Momentum (10 metrics)

| # | Metric | Weight | How | Source | Phase 2 |
|---|---|---|---|---|---|
| A1 | Multi-window relative strength | 18 | weighted 3m/6m/**12-1**; penalizes excessive +1m (reversal) | OHLCV | OK |
| A2 | Distance to highs / base breakout | 14 | % below ATH + bonus for breakout on volume | OHLCV | OK |
| A3 | Relative strength **vs its sector** | 12 | rising 20% in a sector up 25% is weakness in disguise | OHLCV | OK |
| A4 | Sector momentum | 10 | the region's sector ETF/index in an uptrend | OHLCV | OK |
| A5 | Consistency (frog-in-the-pan) | 11 | % positive days / smoothness; drip momentum persists, gap momentum does not | OHLCV | OK |
| A6 | Volatility-adjusted momentum | 11 | return / ATR%; avoids favouring volatile small caps. Reuse ATR% for stop/sizing | OHLCV | OK |
| A7 | Accumulation (OBV + breakout volume) | 9 | single block of volume-based demand | OHLCV | OK |
| A8 | Reaction to last earnings (PEAD) | 8 | gapped on volume after reporting and held → reliable continuation | OHLCV + earnings dates | degraded |
| A9 | Price above MA (20/50) | 3 | redundant with A1/A2; token weight | OHLCV | OK |
| A10 | Event proximity (risk) | 4 | 1.0 with no event in sight; drops as earnings approach. **Penalizes, does not veto** | Earnings calendar | degraded |

*Everything OHLCV comes from the price source (yfinance in Phase 1 US, EODHD globally).*

### Panel B — Quality (10 metrics)

| # | Metric | Weight | How | Source | Phase 2 |
|---|---|---|---|---|---|
| B1 | Revenue growth Y/Y (level) | 14 | percentile vs sector | Fundamentals | OK |
| B2 | Growth trend **+ duration** | 13 | 4Q slope + number of quarters growing >20% (persistence beats recent slope). Tolerant rule: "2 of the last 3 growing" so a single blip does not veto | Fundamentals | OK |
| B3 | **4Q mean** surprise + reaction | 9 | mean of the last 4 surprises (not just 1Q) + price reaction in the following days | Estimates | **disable/reduce in EM** |
| B4 | Cash quality (FCF/NI + FCF↑) | 16 | accruals anomaly (Sloan): accounting profit without cash behind it underperforms. FCF/NI < 0.8 is a red flag. Merges FCF, net income and conversion | Fundamentals | OK (conversion yes; see SBC note) |
| B5 | ROIC vs sector | 11 | replaces ROE (which inflates with leverage) | Fundamentals | OK |
| B6 | Expanding margins | 9 | percentile vs sector | Fundamentals | OK |
| B7 | Dilution / SBC | 10 | growing 30% while diluting 8% ≠ growing 25% while buying back; `weightedAverageShsOutDil` 3y trend; SBC > 15% of revenue penalizes (reported FCF does not deduct SBC) | Fundamentals | **disable/reduce in EM** |
| B8 | Estimate revisions | 10 | best proxy for forward guidance; leads price momentum | Estimates | **disable/reduce in EM** |
| B9 | Balance-sheet health | 5 | net debt/EBITDA | Fundamentals | OK |
| B10 | Valuation | 3 | **EV/Sales or EV/EBITDA as a sector percentile**, NOT raw PEG (PEG explodes at growth ≈0 or negative). Deliberately decorative: in momentum/growth you almost never reject on price | Fundamentals | OK |

**Phase 2 — reduced B panel:** disable B3, B7, B8 (≈29 pts) wherever there is no dense
revisions/surprise/SBC data (emerging and Korean small/mid caps). Renormalize the remaining 71
pts to 100. This is driven from the `Region`'s `active_metrics`, not by touching code.

## 7. Panel combination and alert decision

**Do not average the two scores** (`mean(A,B)` fills the alerts with mediocre-but-uniform
companies) **and do not use `max`** (which rewards excellence in one dimension — useful for a
factor portfolio, not for a buy alert where you want a good company **and** good timing). The
correct rule is an **AND of two high percentile thresholds**:

```
alert  ⟺  gates OK
       ∧  A_pct ≥ threshold_A    (good timing;  default 80th percentile)
       ∧  B_pct ≥ threshold_B    (good company; default 80th percentile)
       ∧  (A_pct * B_pct/100 * regime) ≥ final_cut
```

- An excellent company with terrible timing does not fire; a momentum rocket with garbage
  fundamentals does not either. Consistent with the two-panel philosophy: **B decides *whether*
  the company is worth owning, A decides *whether now*.**
- **Personal watchlist:** same gates, but its own looser threshold (e.g. 70/70) and/or a
  separate Telegram channel, since these are names already being tracked. Configurable.
- Starting thresholds **80/80**, to be **tuned against the backtest**.

## 8. Data sources by phase

Program against a **`DataProvider` interface** (`data/provider.py`) and implement providers that
are swappable by config, so nothing has to be rewritten when moving between phases:

- **Phase 1 (US + Europe):** `yfinance` (OHLCV, free) + `FMP` (fundamentals, estimates,
  surprise, earnings calendar). Sufficient and cheap to start with.
- **Phase 2 (global):** `EODHD` as the primary source (global market and fundamentals coverage
  in a single API; paid tier). `yfinance`/`FMP` remain as US fallbacks.

Cross-cutting requirements for global coverage: **FX** conversion to USD (size and liquidity
gates), exchange suffixes (`.KS`/`.KQ` Korea, `.DE`, `.PA`, `.L`, `.HK`…), and per-exchange
calendars/timezones for "market close".

API keys in a `.env` file (`FMP_KEY`, `EODHD_KEY`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`); never
in code and never in `config.yaml`.

> **What was actually built:** yfinance covers all four regions. FMP's free tier is 250
> requests/day and US-only; EODHD is paid. Yahoo provides quarterly and annual statements,
> dated earnings surprises, revisions and a country-aware screener for free. The
> `DataProvider` interface is unchanged, so switching remains a config change. See the README.

## 9. Alerts (Telegram)

Telegram bot (`alerts.py`). Each alert includes: ticker, region, `A_pct`, `B_pct`,
`final_score`, the region's regime, and the **per-metric breakdown** (what percentiled where).
The breakdown is not optional: when an alert fires you will want to see *why*, and tuning
weights requires it.

## 10. Project structure

```
screener/
├── config.yaml
├── .env                      # API keys (not versioned)
├── data/
│   ├── provider.py           # DataProvider interface (abstraction)
│   ├── yfinance_provider.py
│   ├── fmp_provider.py
│   ├── eodhd_provider.py     # Phase 2
│   └── cache.py              # on-disk cache (parquet); respects rate limits
├── models.py                 # dataclasses: Region, TickerResult, PanelBreakdown…
├── universe.py               # builds the per-region universe + applies gates
├── metrics/
│   ├── momentum.py           # one pure function per metric -> raw value
│   └── quality.py
├── normalize.py              # cross-sectional percentiles (universe / sector, per region)
├── regime.py                 # continuous per-region multiplier
├── panels.py                 # momentum_score(), quality_score() -> panel_raw + breakdown
├── engine.py                 # per-region batch: universe -> metrics -> percentiles -> panels -> decision
├── alerts.py                 # Telegram bot
├── runner.py                 # per-region EOD scheduler (respects closes/timezones)
└── backtest.py               # point-in-time, publication lag
```

## 11. Backtest — traps that invalidate results

Before trusting a single live alert, backtest. Watch out for:

- **Look-ahead / point-in-time.** FMP/EODHD financial statements are dated **by period, not by
  publication date**. If you use Q3 from Sep 30 instead of from its late-October publication,
  the backtest looks glorious and fails in production. Lag every fundamental datum to its **real
  publication date** (use the provider's filing date, not a fixed offset). Same for revisions
  (B8): use the estimate prevailing *at date t*, not the one revised afterwards.
- **Survivorship bias.** Include delisted/bankrupt stocks in the historical universe; if you
  only look at what exists today, the backtest lies.
- **Percentiles without look-ahead.** When normalizing at date t, the percentile pool may only
  use data available at t.
- **Historical FX.** Convert at the rate as of the date, not today's.

## 12. Phased build plan (checklist)

**Phase 1 — US + developed Europe**
- [x] `DataProvider` + yfinance implementation (OHLCV, fundamentals and estimates) with cache.
- [x] `models.Region` with benchmark, percentile pool, provider and `active_metrics`.
- [x] Universe (index constituents + watchlist) and gates (§3).
- [x] Metrics A1–A10 and B1–B10 as pure functions.
- [x] `normalize.py` (universe/sector percentile) and `regime.py` (continuous multiplier).
- [x] `panels.py` + `engine.py` batched per region.
- [x] Alert decision (§7) + Telegram bot + per-metric breakdown.
- [x] `runner.py` EOD.
- [x] `backtest.py` with point-in-time; survivorship is **not** solvable with Yahoo data.
- [ ] Tune thresholds (starting at 80/80) — pending; see the statistical-power limits in
      `estudios/top5_semanal/`.

**Phase 2 — Emerging markets + Korea**
- [x] FX + exchange suffixes + per-exchange calendars.
- [x] New regions with their own benchmark (KOSPI, MSCI EM proxy) and reduced `active_metrics`
      (without B3/B7/B8 where data is missing), weights renormalized.
- [x] Validate data coverage per region before enabling alerts.
- [ ] `eodhd_provider.py` — not needed so far; yfinance covers all four regions.
- [ ] Phase 2 backtest — not run yet.
