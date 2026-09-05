# Stock Screener — momentum + quality, Telegram alerts

After each market close it walks a region's universe, scores every stock on two
independent panels (momentum and quality), applies a market-regime multiplier,
and sends a Telegram alert when a stock clears the threshold on **both** panels
at once.

The full functional specification lives in [`spec.md`](spec.md). This README
covers how to run it and the implementation decisions worth knowing about.

> Code comments and identifiers are in Spanish; all documentation is in English.

## Status

**All four regions live**, plus the point-in-time backtest and the dashboard.

| Region | Phase | Universe | Scored | Alerts | Regime |
|---|---|---|---|---|---|
| `us` | 1 | 1,627 | 816 | 51 | 0.891 |
| `europe_dev` | 1 | 747 | 227 | 10 | 0.929 |
| `emerging` | 2 | 2,996 | 314 | 19 | 0.968 |
| `korea` | 2 | 671 | 29 | 3 | 0.800 |

Two caveats about Phase 2 worth holding in mind when reading those numbers:

- **Korea only scores 29 stocks.** A percentile over 29 names is barely more than
  a rank: the 80th percentile means "top 6 of 29". The §7 rule still works, but it
  discriminates far less than in the US with 816. And with `min_sector_size` at 8,
  nearly every sector falls back to the full regional pool.
- **`emerging` is really Taiwan and India**: 258 of the 314 scored. Indonesia
  contributes 2 and Brazil 13, mostly because the dollar-liquidity gate removes
  the rest.

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then fill in TELEGRAM_TOKEN / TELEGRAM_CHAT_ID
```

Without Telegram credentials the program does not fail: it prints alerts to the
console and tells you they are missing.

## Usage

```bash
# Console ranking + CSV, sends nothing
python -m screener.runner --region us --dry-run --force

# Why a given stock alerts or not: metric-by-metric breakdown
python -m screener.runner --region us --explain AMD --force

# Real run (sends to Telegram)
python -m screener.runner --region us
```

And the dashboard:

```bash
streamlit run screener/panel.py
```

`--force` skips the market-close time check. Without it the runner does nothing
before `close_time_utc`, which is what you want under cron:

```cron
30 21 * * 1-5  cd /path/to/swing-screener && .venv/bin/python -m screener.runner --region us
```

The first run downloads everything and is slow; later ones hit the `./.cache`
directory (prices 12 h, fundamentals 7 days, profile 30 days).

## How an alert is decided

```
alert  ⟺  gates OK
       ∧  A_pct ≥ 80        (good timing)
       ∧  B_pct ≥ 80        (good company)
       ∧  (A_pct · B_pct/100 · regime) ≥ 55
```

The panels are neither averaged nor combined with `max`: **B decides whether the
company is worth owning, A decides whether now is the moment.** The starting
thresholds are 80/80, to be tuned against the backtest. A personal watchlist uses
its own looser thresholds (70/70) and can route to a separate Telegram channel.

### What gets sent and what stays quiet

Clearing the threshold is not the same as deserving a message. The threshold is
stable day to day — if 51 names clear it today, roughly 51 will clear it tomorrow
and most will be the same ones — so only two things are sent:

- **new**: not alerted within `cooldown_days`.
- **improvement** (📈): already alerted, but gained `resurface.score_delta` score
  points or climbed `resurface.rank_jump` places since the alert.

Everything else stays quiet while it sits in the cut without moving. Verified on
the real US universe: first run 51 alerts, second run 0.

In `--dry-run` alerts are deliberately **not** recorded: if they were, a test run
would burn the cooldown and silence the real run afterwards. Ranking snapshots are
always saved, because they are the dashboard's history.

## Dashboard

```bash
streamlit run screener/panel.py
```

Three tabs: **Ranking** (A_pct vs B_pct scatter — the §7 rule is an AND of two
thresholds, so the alert zone is literally the upper-right quadrant — plus the
full table), **Stock detail** (metric-by-metric contribution across both panels
and the score history) and **Alert history** (with the type, new or improvement,
and how the regime evolved).

It reads from two places, both written by the runner: `state.sqlite` for KPIs,
daily rankings and alerts; and `out/<region>_<date>.csv` for the per-metric
breakdown. History accumulates on its own, run after run.

## Configuration

Everything lives in [`config.yaml`](config.yaml): weights, thresholds, universes,
per-metric parameters, cache TTLs. **None of it is in the code.** API keys go in
`.env`.

To disable metrics in a region, list them in its `active_metrics`; the remaining
weights renormalize to 100 on their own. That is the mechanism behind the reduced
B panel in Phase 2, and it requires no code changes.

## Architecture

```
universe → gates → metrics → percentiles → panels → regime → decision → alert
```

Percentiles are cross-sectional, so **a single ticker cannot be scored in
isolation**: the engine works in batch per region and then ranks.

| Module | Responsibility |
|---|---|
| `data/provider.py` | `PriceProvider` / `FundamentalsProvider` / `UniverseProvider` protocols |
| `data/yfinance_provider.py` | Yahoo implementation of all three |
| `data/cache.py` | On-disk parquet cache, TTL, rate limiting |
| `universe.py` | Per-region screener + §3 gates |
| `metrics/` | The 20 metrics as pure functions + registry |
| `normalize.py` | Percentile vs universe (momentum) or vs sector (quality) |
| `regime.py` | Continuous 0.5–1.0 multiplier |
| `panels.py` | Weighted sum of percentiles + breakdown |
| `engine.py` | Per-region batch orchestration |
| `alerts.py` | Telegram with per-metric breakdown |
| `state.py` | Alert and ranking history; decides what gets re-sent |
| `runner.py` | CLI / EOD entrypoint |
| `panel.py` | Streamlit dashboard |
| `pointintime.py` | Trims a `TickerData` to what was known on a past date |
| `backtest.py` | Point-in-time backtest + threshold sweep |

## Implementation notes

Things you cannot infer from `spec.md` and should know before touching the code.

**The provider is yfinance across all four regions, not FMP or EODHD.** FMP's free
tier is 250 requests a day and covers the US only; EODHD is paid. Yahoo covers,
for free, almost everything the spec assigned to both — quarterly and annual
statements, `earnings_dates` with the real report date and the surprise,
`eps_revisions`/`eps_trend`, and a country-aware screener that reaches Korea and
emerging markets. The `DataProvider` interface is untouched: when an EODHD key
appears, switching is two lines of `config.yaml` per region.

**The universe comes from the screener queried sector by sector.** Yahoo's
screener quotes do not carry the sector, but the screener accepts it as a filter:
querying sector by sector labels every ticker at no extra cost. Watch out — the
screener's `exchange` field holds country codes, not MICs.

**Gates run in two passes and the order matters.** First everything that needs
only the screener and prices (size, liquidity, sector, trend), and only then are
fundamentals requested for the survivors. On the free tier that is the difference
between downloading ~1,800 profiles and a few hundred. The industry exclusion
(banks, insurers, REITs) is applied later, once the profile has been downloaded,
because the screener does not expose it.

**Sector indices are synthetic.** A3 and A4 use an equal-weighted index built from
the universe's own constituents rather than ETFs: there is no clean XLK/XLV
equivalent in Europe or in Phase 2. It is built from everything that passed the
cheap gates, not only what passed the trend gate — otherwise the sector index
would measure nothing but rising stocks.

**A1 shifts all three windows, not just the 12-month one.** If the last month
counted inside the 3m and 6m windows, a recent vertical move would raise the score
through that channel more than the reversal penalty lowers it — precisely what the
metric is meant to avoid. The last month is already captured by A2 and A9.

**Yahoo only serves ~5 quarters of income statement**, not enough to measure four
year-over-year quarterly growth rates. B1 and B2 use the annuals (4-5 fiscal
years) and step up to quarterly resolution as soon as there is depth. The cache
**accumulates** statements run after run (`data.accumulate_statements`), so the
local history grows over time; the `.seen.json` sidecar records when each period
was first observed, which is the seed for the point-in-time archive the backtest
needs.

**Europe: three multi-country traps.** Yahoo quotes UK stocks in pence (`GBp`) but
reports their `marketCap` in pounds — verified with
`marketCap / (shares × price) = 0.0100` across the 628 GBp lines in the universe.
Hence **two** exchange rates: `fx_to_usd` for the quoted unit (price and volume)
and `fx_market_cap_to_usd` for the major unit (market cap). With only one,
AstraZeneca comes out at $2.7 bn instead of $265.5 bn and the entire UK fails the
size gate. On top of that, `universe.excluded_exchanges` drops Cboe UK and IOB,
which mirror the LSE or cross-list non-European stocks (174 lines out of 925
without a single new company), and `deduplicate_listings` collapses dual
listings — BBVA in Madrid and London is one company, not two, and would otherwise
count twice in the percentile.

**Emerging markets: the top of the universe is foreign companies' DRs.** Sort
Brazil or Thailand by market cap and the first names are `NVDC34.SA` (Nvidia in
Brazil), `MSFT34.SA` (Microsoft), `NVDA80.BK` (Nvidia in Thailand) — depositary
receipts, not local companies. Johannesburg has the same problem with `BTI.JO`
(British American Tobacco) and `PRX.JO` (Prosus). Without filtering them,
"emerging markets" would be screening US megacaps through their local line. The
filter is `regions.*.countries`, which keys on the company's **domicile**
(`info['country']`), not on the exchange it trades on: it is the only field that
tells them apart reliably (`financialCurrency` does not work — Shell reports in
USD and is a legitimate European company). It only applies where the region
declares `countries`, so the US and Europe behave exactly as validated.

**Careful with country codes in YAML.** `no` is the boolean false in YAML 1.1, not
Norway — same for `on`, `off`, `y` and `n`. They are quoted, and `config.py`
rejects at startup any `yahoo_regions` entry that is not a string; without that it
slips through as the literal `"False"` and Norway disappears with no warning at
all.

**European coverage is better than the spec predicted.** B3 (surprise) covers 76%,
B8 (revisions) 97% and B7 (dilution) 100%: none needs disabling. The only one that
falls over in Europe is **A8** (earnings reaction, 48%), because many European
companies report semiannually and Yahoo's calendar is thinner outside the US. The
`spec.md` prediction about B3/B7/B8 applies to Phase 2 (emerging and Korea), not
to developed Europe.

**Missing data: neutral percentile and coverage discount.** A metric that cannot
be computed returns `None`, is imputed at the 50th percentile, and is discounted
from the ticker's coverage. Below `coverage.min_panel_coverage` the stock is
dropped **before** percentiling, so a ticker with junk data cannot shift everyone
else's percentile. A metric that falls below `coverage.min_metric_coverage` across
the whole region is disabled for that run and the weights renormalize.

## Tests

```bash
.venv/bin/python -m pytest -q
```

198 tests, none of which touch the network: synthetic price series and financial
statements, plus a full pipeline against a fake provider.

First real US run (2026-07-31): 1,627 screener candidates → 1,598 after
size/liquidity/sector → 849 after trend → 816 scored → 51 alerts. Lowest per-metric
coverage 82% (B4); none was disabled. Cold run ~23 min, warm 51 s.

## Backtest

```bash
python -m screener.backtest --region us
python -m screener.backtest --region us --start 2023-01-01 --sweep-horizon 63
```

At each rebalance date it rebuilds the universe as it looked that day and scores
it with **the same** `engine.score_universe` the live run uses. Sharing that
function is not a style choice: a backtest that reimplements scoring measures
something else and drifts apart at the first metric change.

What it does about the four traps in §11:

| Trap | Status |
|---|---|
| Look-ahead / point-in-time | **Solved.** Each statement is lagged to its real report date (`earnings_dates`), not the period-end date. |
| Percentiles without look-ahead | **Solved by construction.** The pool at date t is computed only from data trimmed to t. |
| Historical FX | **Solved.** Gates use the rate as of the date, injected into `apply_cheap_gates`. |
| Survivorship bias | **NOT solved, and not solvable with Yahoo.** |

The two limitations the report repeats on every run, because without them the
numbers mislead:

- **Survivorship.** The universe starts from today's screener, so it contains no
  delisted or bankrupt names. Yahoo does not serve historical constituents.
  Returns are biased upward.
- **B8 is not backtestable.** `eps_revisions` is a snapshot of today, not a
  vintaged series: on a past date there is no way to know what the prevailing
  estimate was. It is disabled and its 10 points redistribute automatically. B3
  and B7 **are** backtestable, contrary to what I assumed at the start:
  `earnings_dates` carries a dated surprise and `get_shares_full` is a time series.

The report compares alert returns against the **mean of the scored universe on the
same dates**, not against 0%: that is the bar to clear. The threshold sweep
(`threshold_grid`) re-evaluates the cut without re-running the backtest, which is
what the spec asked for when tuning the 80/80.

## Studies

`estudios/top5_semanal/` holds a point-in-time simulation of buying €100 of the
top N every Monday, across all four regions. It is the most methodologically
interesting part of the repository, and its headline result is a negative one.

**What holds up:**

- **Any stop-loss cuts the tail of winners** and costs 29 to 36 points, though it
  does cap the worst case from −54% to −29%.
- Two of twenty metrics replicate across all four regions: **A6 (momentum/ATR%)**
  and **A1 (multi-window RS)**, with Cochran's Q of 0.72 and 0.21 against a 7.81
  critical value.
- The ranking orders returns within every region: in emerging markets, ranks 1-3
  return +15.0% over 63 days against +4.9% for ranks 51-100.

**What does not hold up, and this is the point:**

An earlier weight change, derived from the US information coefficient, improved
the US backtest by roughly 6 points a year. It was noise — t=1.41, and the
advantage evaporated as the portfolio grew, from +10.4pp at top 3 to +1.6pp at
top 10. Validation across the other three regions showed the two metrics it had
reweighted were the **least stable of the twenty**: sector momentum measures
−0.060 in the US and +0.079 in Europe. The change was reverted and the weights are
now frozen, with the rationale written next to them in `config.yaml`.

More broadly: the strategy beats its own index in 3 of 4 regions by 4-7 points a
year, but **no excess return reaches significance in any region** (highest |t| =
0.84 across twelve region×size combinations). At the observed information ratios
it would take 15 years of data in the US and 91 in emerging markets to reach t=2.
There are 2.6 available. Any comparison between weight sets, exit rules or
portfolio sizes falls inside the noise **by construction** — which is why the
weights are frozen rather than tuned.

That directory also explains why no backtest can start before January 2024: the B
panel runs out of data going backwards (10% coverage in 2021-2022). Read it before
proposing a longer window.

The expensive artifact is `senales_top20_<region>.json`, holding the top 20 and
their scores for all 135 weeks: ~50 minutes of computation each, and with them any
N ≤ 20 or any new exit rule evaluates in seconds.

## Deployment

The screener is a cron job, not a service: it runs for a few minutes a day per
region and needs no inbound traffic. A small VM with `cron` and local disk is
enough — measured peak is **0.76 GB of RAM at 27% CPU**, so it is bound by
network, not compute. Anything with 2 GB of usable memory and 10 GB of disk fits.

`run.sh` wraps the runner for cron. It resolves its own location, so the same
file works on any machine without editing:

```bash
./run.sh us
```

It holds a per-region lock (emerging takes ~45 min cold, and without the lock the
next day's cron would start on top of it), caps the log, and sends a Telegram
message if a run fails — a cron that dies silently goes unnoticed until you miss
the alerts weeks later.

### Setup on a fresh VM

```bash
sudo apt install -y python3-venv python3-pip git curl cron
sudo systemctl enable --now cron

git clone <repo> /opt/swing-screener && cd /opt/swing-screener
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # 198 tests, no network: verifies the platform

cp .env.example .env                   # optional: without it, alerts print to console
```

On a 1 GB instance, add swap before installing — the peak would otherwise hit the
ceiling mid-run:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Cron

Market closes are in UTC, which is what `close_time_utc` already uses. Korea and
Europe are far enough apart to stand alone. US and emerging both close at 21:00,
so they share one entry and run back to back:

```cron
45  6 * * 1-5  /opt/swing-screener/run.sh korea
50 16 * * 1-5  /opt/swing-screener/run.sh europe_dev
15 21 * * 1-5  /opt/swing-screener/run.sh us; /opt/swing-screener/run.sh emerging
```

They are chained rather than staggered by a fixed offset because the US runtime is
not stable: measured over six sessions it ranged from 19 minutes to 2h24, so any
fixed gap eventually gets overtaken. The previous 45-minute stagger (`emerging` at
22:00) overlapped on four of those six days. Overlap matters here because the box
has 1 GB of RAM and one region alone already peaks at 0.76 GB — two at once pages
to swap. The `;` runs `emerging` even if `us` exits non-zero; the per-region lock
in `run.sh` does not help, since it only guards a region against itself.

The first run per region downloads everything cold: ~23 minutes for the US, longer
for emerging. Later runs hit the cache, but "cached" is not "fast" — daily US runs
in production still take 20 minutes on a warm cache and occasionally hours.

### One machine only

`state.sqlite` decides which alerts have already been sent. Running the screener
on two machines gives each its own copy, the cooldowns drift apart, and you get
duplicate alerts for the same stocks. Pick one machine for real runs and use
`--dry-run` elsewhere — it deliberately records nothing.

## Disclaimer

**This is a research project, not investment advice.** It produces alerts from a
quantitative model with known and documented limitations: the backtest carries
uncorrected survivorship bias, covers a period with no bear market, and — as the
study in `estudios/top5_semanal/` shows in detail — no excess return it produces
reaches statistical significance in any of the four regions. Nothing here is a
recommendation to buy or sell any security. Anyone acting on its output does so
at their own risk and should do their own research.

## Data source

Market data comes from Yahoo Finance through the [`yfinance`](https://github.com/ranaroussi/yfinance)
library, which reads Yahoo's public but undocumented endpoints. That data is
provided by Yahoo under its own terms of use, which restrict commercial
redistribution; this repository uses it for personal research only and ships no
market data of its own. The cache directory is gitignored for that reason as much
as for its size.

If the project ever needed a commercial footing, the `DataProvider` interface in
`data/provider.py` exists precisely so the source can be swapped for a licensed
one (EODHD, FMP paid tier) without touching the engine.

## License

MIT — see [`LICENSE`](LICENSE).

## Open items / decisions

- **Tune the thresholds** against the backtest, bearing in mind survivorship bias
  and that the effective sample is ~15 independent observations.
- **Korea's reduced B panel may be unnecessary.** The config disables B3, B7 and B8
  in Korea and emerging markets following spec §6, but Yahoo does have the data: in
  the Korean samples `eps_revisions` returned all four rows and `get_shares_full`
  between 170 and 762 points. Since `coverage.min_metric_coverage` already disables
  any data-less metric on its own, hard-disabling them is probably throwing away
  signal. Test it by setting `quality: null` in their `active_metrics`.
- **EODHD** once there is a paid key: two lines per region in the config, plus
  implementing `eodhd_provider.py` against the already-defined protocols.
- **Phase 2 backtest**: not run yet. With 29 stocks in Korea the historical
  percentile will be even coarser than the current one.
