# Merton Structural Credit Risk Model

From-scratch implementation of the Merton (1974) structural credit risk model, applied
to a 20-company cross-section spanning AAA to B-.

Equity is treated as a European call on the firm's assets struck at the face value of
debt. Asset value and asset volatility are unobservable, so both are backed out jointly
from observed equity value and equity volatility:

```
E         = V·N(d1) - D·e^(-rT)·N(d2)
sigma_E·E = N(d1)·sigma_V·V
```

The solved pair gives distance-to-default (d2), the risk-neutral 1-year default
probability N(-d2), the model-implied credit spread, and asset leverage.

Full output and commentary: **[results.md](results.md)**.

## Run

```
pip install -r requirements.txt
python merton.py                        # ranked table + rating-bucket summary
python merton.py --markdown             # markdown tables
python merton.py --stress 30            # also report a 30% equity drawdown
python merton.py --rate 0.045 --horizon 5
python merton.py --data my_companies.csv
```

## Headline results

| | |
|---|---|
| Rank correlation, model DD vs S&P rating | 0.94 |
| Safest (DD) | Microsoft 20.41 |
| Riskiest (DD) | American Airlines 1.74 |
| Rank correlation, DD vs leverage | -0.79 |
| Rank correlation, DD vs asset volatility | +0.11 |

Distance-to-default tracks agency ratings closely (0.94), and what drives the ordering
is leverage rather than volatility. NVIDIA carries the highest asset volatility in the
sample (44.9%) and still ranks seventh safest, because its default barrier is 0.2% of
asset value. The two BBB- names Carnival (2.88) and Ford (2.94) reach nearly identical
distance-to-default by opposite routes — moderate leverage with high volatility, versus
very high leverage with very low volatility.

The model's investment-grade *levels* are not usable: everything rated A+ or better
prints a 1-year PD indistinguishable from zero and a spread under 1bp, against real
market spreads of tens of basis points. That is the credit spread puzzle, and it is
expected behaviour for a single-factor Merton model at a one-year horizon. The rankings
are the output worth reading, not the probabilities.

## Data

Inputs live in [`companies.csv`](companies.csv): market value of equity, annualized
equity volatility, face value of total debt, and the S&P issuer rating, plus a 4.03%
1-year Treasury yield as the risk-free rate.

**These are approximate snapshot figures for Aug 2026, rounded to the nearest round
number and drawn from public market-data aggregators rather than raw 10-K filings.**
They are good to roughly an order of magnitude on debt and within a few percent on the
large-cap equity values, which is enough to rank a cross-section but not enough to
quote a specific issuer's default probability. Refresh the CSV from a real data source
before using any single name's output. Boeing, Carnival and J&J carry the same inputs
as the original three-company version, and reproduce its results exactly.

## Modelling caveats

Beyond the near-zero investment-grade PDs described above:

- **Captive finance arms distort leverage.** Ford's and GM's debt totals include Ford
  Credit and GM Financial borrowings, which fund receivables rather than industrial
  operations. Treating that as a default barrier on the consolidated firm overstates
  leverage, so their DD figures (2.94, 3.36) understate credit quality. The same
  applies to secured aircraft and loyalty-program financing at the airlines.
- **The barrier is face value of total debt at a single horizon.** Real capital
  structures have a maturity schedule; commercial implementations (e.g. KMV) place the
  barrier nearer short-term debt plus half of long-term debt.
- **Equity volatility is a single point estimate** and is held fixed. In practice it is
  both time-varying and, for levered names, strongly negatively correlated with the
  equity price — which is why the `--stress` results should be read as an artifact of
  the shock design rather than an economic conclusion (see results.md).
- **PDs are risk-neutral**, not physical. They embed a risk premium and sit above
  actual expected default frequencies.

## Implementation notes

The two-equation system is solved with `scipy.optimize.fsolve` over `(log V, log
sigma_V)` rather than `(V, sigma_V)`, which keeps both unknowns positive without a
penalty branch. Residuals are scaled by their own magnitudes so that a $4T firm and a
$9B one converge to the same relative tolerance. Convergence is asserted per company —
a non-converging solve raises rather than silently returning a plausible-looking number.
