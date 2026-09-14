# Merton Structural Credit Risk

A from-scratch implementation of the Merton (1974) structural credit risk model,
estimated on **132,503 firm-months across 1,804 US non-financial companies**
(2016–2026) and tested against bankruptcies read directly out of SEC filings.

The model is textbook. The point of the repository is everything around it: a
point-in-time panel with no look-ahead, default labels taken from primary sources,
delisted firms deliberately kept in, and a validation suite that is capable of
showing the model losing — which, on one of the two outcomes tested, it does.

Full output: **[results.md](results.md)**.

## Headline results

| | |
|---|---|
| Panel | 132,503 firm-months, 1,804 firms, 120 months (2016–2026) |
| Bankruptcies observable with pre-filing equity data | 22 firms, 214 firm-months |
| Solver convergence | 99.99% |
| **AUC, predicting a bankruptcy filing 12 months out** | **0.82** (95% CI 0.71–0.91) |
| AUC, equity volatility / Altman Z / book leverage, same task | 0.77 / 0.76 / 0.62 |
| Out-of-sample AUC: DD alone / accounting ratios / both | 0.82 / 0.76 / 0.83 |
| Correlation, Δ median distance-to-default vs Δ Baa spread | −0.67 |
| Rank correlation, distance-to-default vs S&P rating (20 names) | 0.94 |

Three findings are worth more than the headline number.

**Distance-to-default predicts bankruptcy, and it is not just leverage in disguise.**
AUC 0.82, against 0.62 for book leverage and 0.76 for the Altman Z-score. Out of
sample it beats a full set of accounting ratios on its own (0.82 against 0.76), and
adding the ratios to it buys almost nothing (0.83) — the structural measure is doing
the work. Firms that went on to file sat at a median distance-to-default of 0.8 in
the quarter of filing, against 5.6 for firms that never filed, and the decline is
already visible three years out.

**On a broader distress definition, the model is beaten by the simplest possible
benchmark.** If the event is "the equity lost 90% of its value," trailing equity
volatility alone scores 0.86 and distance-to-default only 0.63. That is not a bug —
it is what the comparison is for. An equity wipeout is substantially a volatility
event, and the structural machinery adds nothing to predicting it beyond the
volatility that goes into it. The model earns its keep on insolvency, not on drawdown.

**The probabilities themselves are unusable, and the failure is quantified rather
than asserted.** Above about three standard deviations of distance-to-default the
model puts the one-year default probability at zero to machine precision, for firms
that went on to file anyway; in the riskiest decile it is too pessimistic by
roughly a factor of ten. This is the credit spread puzzle, and it is expected behaviour for a
single-factor diffusion at a one-year horizon. The rankings are the output worth
reading; the levels are not.

![Distance-to-default approaching bankruptcy](figures/event_study.png)

![Model-implied distress against traded credit spreads](figures/credit_cycle.png)

## What this is

Equity is a European call on the firm's assets, struck at the face value of debt:

```
E          = V·N(d1) − D·e^(−rT)·N(d2)
σ_E·E      = N(d1)·σ_V·V
```

Asset value `V` and asset volatility `σ_V` are unobservable, so both are backed out
jointly from observed equity value and equity volatility. The solved pair gives
distance-to-default (`d2`), the risk-neutral default probability `N(−d2)`, the
model-implied credit spread, and asset leverage.

That is textbook. What makes it a study rather than a demo is the data underneath it
and the fact that the output is checked against what actually happened.

## The data

Everything comes from free primary sources, and every fetch is cached, so a rebuild
costs nothing and the committed panel reproduces offline.

| | Source | Used for |
|---|---|---|
| Fundamentals | SEC XBRL `companyfacts` | debt, shares, balance sheet, income statement |
| Default events | SEC 8-K **Item 1.03** filings | bankruptcy dates |
| Prices | stockanalysis.com daily history | market cap, realized volatility |
| Rates | FRED `DGS1`, `BAA10Y`, `AAA10Y` | risk-free rate, market credit spreads |

Three things about the construction are worth more than the model code.

**No look-ahead.** An accounting figure enters a month only if the SEC had already
received it by that month-end. Every XBRL fact carries a `filed` date alongside the
period it covers, and the join keys on the former. December's balance sheet is not
knowable in January — it becomes usable in February, when the 10-K is filed. Market
data enters only from the trailing window. Nothing in a row could have been unknown
to an observer standing at that month-end. There is a test for exactly this
(`test_as_of_uses_filing_date_not_period_end`).

**Default labels are primary-source.** A US issuer entering bankruptcy or
receivership must file an 8-K under Item 1.03 within four business days. Those
filings are dated and machine-readable in EDGAR's submissions index, so default
dates are read off the filings themselves rather than transcribed from a list
someone else compiled. Nothing here depends on my recollection of who went bankrupt
and when.

**Delisted firms are kept.** Firms that leave SEC's ticker file are
disproportionately the ones that failed, and dropping them would rebuild the exact
survivorship bias the panel exists to avoid. Tickers missing from that file are
resolved by matching the EDGAR company name against the price provider, with a
conservative similarity threshold — a wrong ticker would attach one firm's prices to
another firm's balance sheet, which is far worse than a miss, so borderline matches
are rejected and the firm is dropped.

## The honest limitation

The clean test — does distance-to-default predict a bankruptcy filing? — is
underpowered here, and the reason is worth stating plainly.

Firms that reorganize under Chapter 11 and relist keep filing with the SEC, so their
Item 1.03 date is recorded. But their *pre-filing share price* is not available from
any free source: the surviving ticker's history begins at emergence. Hertz, Chesapeake,
Frontier, Valaris, Noble, California Resources, Weatherford, iHeartMedia and Clear
Channel all appear in this panel with a verified bankruptcy date and no equity data
in front of it. What survives with usable price history is mostly firms that were
liquidated under their final ticker, or that filed and kept trading.

So the panel scores two outcomes:

- **Bankruptcy within 12 months** — the clean event, but scarce, and reported with
  wide confidence intervals that reflect how few firms carry it.
- **Severe distress within 12 months** — the firm either filed, or its equity lost
  90% of its value. Observable for every firm in the panel, and so far better
  powered: 638 events against 214.

The second label was added for power, and testing it taught something the first
could not. Predicting large equity falls with equity-derived measures is partly
mechanical — a volatile stock is more likely to lose 90% whatever its leverage — so
raw equity volatility is carried through every comparison as a competing score,
precisely to see whether the structural machinery earns its keep. It does not, on
that label: volatility alone scores 0.86 and distance-to-default 0.63. On the
bankruptcy label the ordering reverses, and distance-to-default beats volatility
0.82 to 0.77.

The reasonable conclusion is that the wipeout label measures something closer to
"how volatile is this equity" than "is this firm insolvent," and that the scarce,
clean label is the one that speaks to the model's actual claim. A version of this
project with a paid default database — Moody's DRD, or CRSP delisting codes — would
not have to make that trade.

## Running it

```bash
pip install -e ".[dev]"

python -m merton.cli crosssection            # the 20-name rating cross-section
python -m merton.cli crosssection --stress 30 --markdown
python -m merton.cli panel                   # fit the committed panel, riskiest firms

python scripts/run_analysis.py               # regenerate results.md and figures/
python scripts/build_panel.py --firms 1800   # rebuild the panel from source (slow)
pytest -q
```

`build_panel.py` is resumable and writes one shard per firm, so an interrupted run
picks up where it stopped. `run_analysis.py` reads the committed panel and needs no
network.

## Layout

```
merton/
  core.py            the model: solve, distance-to-default, PD, spread
  crosssection.py    the hand-built rating cross-section
  cli.py             command line entry points
  data/
    http.py          cached, rate-limited, retrying fetcher
    sec.py           EDGAR: universe, point-in-time facts, Item 1.03 labels
    prices.py        daily prices, market cap, realized volatility
    symbols.py       name-to-ticker resolution for delisted firms
    rates.py         FRED series, cached to data/fred
    panel.py         panel assembly, labels, accounting ratios
  analysis/
    fit.py           run the solve across the panel
    validation.py    AUC, bootstrap, deciles, calibration, logistic benchmarks
    timeseries.py    monthly aggregates against market spreads
    figures.py       the charts
scripts/
  build_panel.py     build the panel from source
  run_analysis.py    fit, validate, write results.md and figures/
tests/               68 offline tests
data/                committed panel, universe, defaults, FRED series
```

## Implementation notes

The two-equation system is solved two ways. `solve_assets` uses `scipy.optimize.fsolve`
over `(log V, log σ_V)`, which keeps both unknowns positive without a penalty branch,
with residuals scaled by their own magnitudes so a $4T firm and a $90M one converge to
the same relative tolerance. `solve_panel` runs a vectorized Newton iteration with an
analytic Jacobian instead, falling back to `fsolve` only on rows it cannot settle —
100,000 firm-months solve in about 0.15 seconds rather than two minutes. The Jacobian
simplifies sharply because of the Black–Scholes identity `V·φ(d1) = D·e^(−rT)·φ(d2)`:
the two vega terms in the first equation cancel, leaving `∂f₁/∂(log V) = V·N(d1)`
exactly. A test pins the fast path against `fsolve` across the full range of leverage
and volatility in the panel, because a hand-rolled Newton solver is exactly the kind
of optimization that can be subtly wrong and still look plausible.

The tests are deliberately offline — the model's mathematics, the point-in-time joins
and the validation machinery all run against synthetic inputs with known answers — so
CI never depends on a third-party endpoint being up.

## Modelling caveats

- **Probabilities are risk-neutral, not physical.** They embed a risk premium and sit
  above actual expected default frequencies. Section 3 of `results.md` measures the
  gap rather than asserting it.
- **The barrier is the face value of total debt at a single horizon.** Real capital
  structures have a maturity schedule; commercial implementations place the barrier
  nearer short-term debt plus half of long-term debt.
- **Captive finance arms distort leverage.** Ford's and GM's debt includes their
  credit subsidiaries, which fund receivables rather than industrial operations.
  Treating that as a default barrier overstates leverage.
- **Equity volatility is a trailing 12-month estimate.** It is backward-looking by
  construction, which is what keeps it out-of-sample, but it also means the model
  normalizes slowly after a crash: volatility stays elevated for a year after the
  event, which is visible in the 2020–21 plateau in the credit-cycle figure.
- **Banks and insurers are excluded** (SIC 6000–6799). Their leverage is a regulatory
  construct and deposits are not debt in the sense the model means.
- **The cross-section inputs are approximate.** `companies.csv` holds rounded
  snapshot figures for 20 names, good enough to rank a cross-section and not good
  enough to quote any single issuer's default probability.
