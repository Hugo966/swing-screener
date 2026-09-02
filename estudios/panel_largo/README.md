# Study: extending the backtest to 2013 with SEC data

The backtest cannot start before January 2024, and that is not a tuning
problem — it is the data. Yahoo serves 4-5 annual fiscal years and ~5 quarters,
so reconstructing a 2021 date leaves the B panel at 10% coverage and every stock
gets dropped by `min_panel_coverage`.

The SEC publishes the XBRL of every US filer quarterly, going back to 2009, and
each record carries the date it was actually filed. That is strictly better than
what this project does today, which approximates the publication date from
Yahoo's earnings calendar.

## Why it matters

With 2.6 years of data, no excess return over the benchmark reaches significance
in any region — the highest t across twelve region×size combinations is 0.84.
At the observed information ratio of 0.52 in the US, reaching t=2 needs about 15
years. The study in `../top5_semanal/` documents this in detail.

```
                   years    t = IR x sqrt(years)
Yahoo today          2.6     0.84   insufficient
SEC from 2013       13.0     1.87   on the threshold
```

So this does not answer the question on its own. It gets close enough that two or
three more years of live out-of-sample data would cross the line, instead of
fifteen.

## What the data actually supports

XBRL was phased in, so the early years are nearly empty. Measured, not assumed:

| Quarter | Filers |
|---|---|
| 2009q2 | 22 |
| 2010q2 | 522 |
| 2011q2 | 1,705 |
| 2012q2 | 9,193 |
| 2013q2 | 8,954 |

Full coverage starts in **2012**, not 2009. And the trend gate needs 280 days of
prior price history, plus A1 looks back 252 sessions ending 21 days ago — so the
usable backtest start is **2013**, with 2012 consumed by the trailing window.

## The filing lag, which is the whole point

Measured over 5,174 filings in one quarter:

```
10-K: median 62 days between period end and filing
10-Q: median 41 days
43% of filings take more than 60 days
```

Using period end instead of the filing date would hand the strategy a **58-day
head start on the market** — precisely the look-ahead trap `spec.md` §11 warns
about. The current backtest avoids it by approximating with Yahoo's earnings
calendar; the SEC gives the real date.

One nuance in the conservative direction: `filed` is when the report was
submitted, which is a few days after the earnings press release. So this
provider assumes information became public slightly later than it did. For a
backtest, erring late is the safe side.

## Cross-checking the tag mapping

Before trusting a panel that reaches 2013, the XBRL mapping was checked against
Yahoo over 2024-2026, where both sources overlap. 39 large caps, 1,153
period-by-period comparisons, matched on exact period-end dates rather than by
position — fiscal years do not line up with calendar years.

The first run agreed on 92.2% of comparisons within 1%, and the disagreements
pointed at two real bugs rather than at noise.

### CapEx is a sum, not a tag

Mastercard reported $371M of property and equipment and $717M of software.
Yahoo's figure is $1,088M — the sum. Caterpillar adds equipment leased to
others. Taking the first matching tag understated capital expenditure by roughly
half in those cases, and since free cash flow is reconstructed as `OCF - capex`,
it silently **inflated FCF** — the input to B4, the heaviest metric in the panel
at weight 16.

The fix models capex as components that sum, each with its own fallback chain:
within a component the alternatives compete (they are different names for the
same thing), across components they add. The components are property and
equipment, capitalised software, and equipment leased to others.

Two further iterations, both driven by the data rather than by taste:

**Verizon needed a tag that was missing entirely.** It books almost all of its
capital spending under `PaymentsToAcquireOtherProductiveAssets`, so without that
alternative it showed $450M instead of $17.5B.

**Intangibles were tried and dropped.** Adding acquired intangibles and
in-process R&D fixed Johnson & Johnson — $4,424M + $1,783M is exactly Yahoo's
$6,207M — but broke Philip Morris and Verizon, where Yahoo excludes them.
Overall agreement fell from 97% to 95%, so they are out. Yahoo's own definition
is not consistent across companies, so the rule kept is the one that stands on
its own terms: productive capacity yes, purchases of intangible assets no. JNJ
remains a known and documented difference.

### Yahoo has two different operating income rows

For Intel in 2025:

```
Total Operating Income As Reported   -2,214,000,000   ← matches the SEC exactly
Operating Income                        -23,000,000   ← normalised figure
```

The SEC value was right; the label was wrong. Calling GAAP operating income
"Operating Income" would make B5 and B6 measure different things depending on
which provider was used — the kind of inconsistency that makes two backtests
incomparable without either of them looking broken. It is now labelled
`Total Operating Income As Reported`, which `_financials.OPERATING_INCOME`
already looks for as a fallback.

### EBITDA has to be composed

`EBITDA` is not a GAAP concept and barely exists in XBRL — 21 rows across a
quarter with thousands of filers. Yahoo computes it; B9 (net debt / EBITDA)
expects it. The provider composes it the same way: reported operating income
plus depreciation and amortisation, and omits the row entirely when D&A is
missing rather than emitting a figure that would silently equal operating income.

### Where they still disagree, the SEC is usually right

After the fixes, agreement went from 92.2% to **97.8%**, and the residue is
informative rather than noisy.

A caveat on reading that number: the headline percentage can fall while the data
improves. Adding Verizon's capex tag dropped the capex agreement from 97.1% to
94.4% — not because anything broke, but because Verizon previously had **no capex
row at all** and contributed no comparisons. The same 135 comparisons still
agree; four new ones appeared, all of them for a company whose free cash flow was
previously equal to its operating cash flow, which for a telecom is nonsense.
Coverage and agreement are different things, and coverage matters more.

Johnson & Johnson's 2022 revenue is $94.9B in the SEC data and $80.0B in Yahoo's.
Danaher shows the same pattern. Both spun off a division in 2023 — Kenvue and
Veralto — and **Yahoo restates the historical figures to exclude the discontinued
business, while the SEC keeps what was actually reported at the time**.

For a backtest, the SEC figure is the correct one: in 2022 the market saw $94.9B.
Scoring a 2022 date against a number that only exists because of a 2023 corporate
action is look-ahead, and it is invisible — the restated figure looks perfectly
clean.

This is the same reasoning behind keeping the earliest `filed` during
consolidation, and it is an argument for the SEC beyond simply having more years.

The remaining capex gaps are the same story. Verizon reported about $18.8B of
capital spending in 2023, which is what the SEC data gives; Yahoo's $24.6B
includes spectrum licences. Johnson & Johnson's gap is acquired in-process R&D.
In both cases the as-reported figure is the defensible one for a screener that
needs the same definition applied to every company.

### Rows the metrics expect already aggregated

Two metrics went to 0% coverage on the first real run and the engine disabled
them, which shows up as one line in the log and nothing else:

- **B9 (net debt / EBITDA)** looks for a `Total Debt` row. The SEC publishes debt
  split into current and non-current tranches, so it found nothing. The provider
  now composes `Total Debt`, and also the combined
  `Cash Cash Equivalents And Short Term Investments` row that `net_debt` prefers —
  using cash alone would overstate net debt.
- **EBITDA**, for the same reason described above.

Both are composed with `min_count=1`, so a period with no data stays NaN instead
of becoming zero. A zero would read as "no debt", which is a claim, not a gap.

### Dates that cannot be real

0.067% of rows carry period end dates like 1927 or 2923 — filer typos. A period
cannot close after it was filed, and nothing before 2005 belongs in a panel that
starts in 2012. Both filters are applied during consolidation.

## What this does not fix

- **Survivorship bias, still unsolved.** The universe comes from Yahoo's
  screener, so delisted and bankrupt companies are absent even though the SEC
  has their filings. Returns remain biased upward. Fixing this needs a paid
  source (Norgate, Sharadar) with historical constituents and delisted prices.
- **Market cap is today's**, not the simulated date's. That limitation predates
  this work.
- **The early cross-section is thin.** In 2013 only about 200 of the 1,561
  current listings pass the gates, because the universe is what trades *today*
  and most of the rest had not yet IPO'd — they fail the 280-day price history
  requirement. That is correct behaviour, but percentile normalisation over 200
  names discriminates less than over the ~800 available now, so the early years
  carry less weight than their length suggests.
- **B3 and B8 have no data.** Earnings surprise and estimate revisions come from
  analysts, not from XBRL filings. `min_metric_coverage` disables them and
  renormalises the remaining weights — the same mechanism as the reduced B panel
  in Phase 2.

## How it is put together

```
sec_download.py   ~58 quarterly ZIPs (5.4 GB) + the official CIK→ticker map
sec_parse.py      strips each ZIP to the ~25 tags the B panel needs (500 MB → 5 MB)
sec_provider.py   assembles Statements per company, keeping first publication
hybrid_provider.py  SEC fundamentals + Yahoo prices, sector and universe
```

Three things that produce plausible but wrong numbers if done carelessly, all
covered by tests in `tests/test_sec.py`:

- **CapEx sign.** Yahoo reports `Capital Expenditure` negative and the code does
  `ocf + capex`. The SEC reports the payment positive. Without inverting it, free
  cash flow comes out inflated and nothing warns you.
- **Consolidated only.** Rows with `segments` or `coreg` are divisional or
  co-registrant breakdowns. Adding them to the total double-counts.
- **First publication wins.** A 2025 10-K restates 2023 figures. The market back
  then saw the original number, so the earliest `filed` is what counts — the same
  rule as the `.seen.json` sidecar in the Yahoo cache.

## Running it

```bash
# One-off: download and parse (about 20 minutes, 5.4 GB)
SEC_CONTACT_EMAIL=you@example.com .venv/bin/python -m screener.data.sec_download
.venv/bin/python -m screener.data.sec_parse

# Cross-check the tag mapping against Yahoo where they overlap
.venv/bin/python estudios/panel_largo/scripts/validar_contra_yahoo.py 40

# The long backtest
.venv/bin/python -m screener.backtest --region us --start 2013-01-01 --fundamentals sec
```

`--fundamentals` is a command-line flag rather than a config setting on purpose:
`config.yaml` also governs the daily production run, which lives on a 1 GB VM
that would not survive loading the consolidated panel. Opting into the SEC has to
be a deliberate act, not a state the config can drift into.

The SEC requires a real contact email in the User-Agent and returns 403 without
one. It goes in `.env` as `SEC_CONTACT_EMAIL`, not in the code.

Loading the consolidated panel takes about 1 GB of RAM, so this is for studies on
a workstation — not for the 1 GB production VM, which only runs the daily screen.
