# Results

Output of `python merton.py --markdown --stress 30`, 20 companies, r = 4.03%, T = 1 year.

**Inputs are approximate snapshot figures, not audited filing data** — see the Data
section of the README. Treat the rankings as the result and the decimals as noise.

## Full cross-section

Ranked by distance-to-default. `Lev` is the discounted default barrier as a share of
implied asset value, `Spread` the credit spread implied by the model's value of the
firm's debt.

| Company | Rating | Equity | Debt | Asset Val | Asset Vol | Lev | DD | PD (1yr) | Spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Microsoft (MSFT) | AAA | 3,700B | 45.0B | 3,743B | 21.7% | 1.2% | 20.41 | ~0 | <1bp |
| Alphabet (GOOGL) | AA+ | 2,600B | 28.0B | 2,627B | 25.7% | 1.0% | 17.68 | ~0 | <1bp |
| Johnson & Johnson (JNJ) | AAA | 616.0B | 50.8B | 664.8B | 15.8% | 7.3% | 16.50 | ~0 | <1bp |
| Coca-Cola (KO) | A+ | 300.0B | 45.0B | 343.2B | 14.0% | 12.6% | 14.75 | ~0 | <1bp |
| Apple (AAPL) | AA+ | 3,300B | 98.0B | 3,394B | 24.3% | 2.8% | 14.63 | ~0 | <1bp |
| Walmart (WMT) | AA | 780.0B | 60.0B | 837.6B | 19.6% | 6.9% | 13.59 | ~0 | <1bp |
| NVIDIA (NVDA) | A+ | 4,000B | 10.0B | 4,010B | 44.9% | 0.2% | 13.22 | ~0 | <1bp |
| Exxon Mobil (XOM) | AA- | 480.0B | 40.0B | 518.4B | 22.2% | 7.4% | 11.60 | ~0 | <1bp |
| Meta Platforms (META) | AA- | 1,700B | 50.0B | 1,748B | 31.1% | 2.7% | 11.39 | ~0 | <1bp |
| Amazon (AMZN) | AA | 2,400B | 130.0B | 2,525B | 26.6% | 4.9% | 11.16 | ~0 | <1bp |
| Tesla (TSLA) | BBB | 1,100B | 13.0B | 1,112B | 54.4% | 1.1% | 7.98 | 7.1e-14% | <1bp |
| AT&T (T) | BBB | 200.0B | 135.0B | 329.7B | 12.1% | 39.3% | 7.63 | 1.2e-12% | <1bp |
| Verizon (VZ) | BBB+ | 180.0B | 150.0B | 324.1B | 10.6% | 44.5% | 7.63 | 1.2e-12% | <1bp |
| Boeing (BA) | BBB- | 166.0B | 54.4B | 218.3B | 25.1% | 23.9% | 5.57 | 1.3e-06% | <1bp |
| Delta Air Lines (DAL) | BBB | 35.0B | 18.0B | 52.3B | 26.8% | 33.1% | 4.00 | 3.2e-03% | <1bp |
| Intel (INTC) | BBB | 100.0B | 50.0B | 148.0B | 30.4% | 32.4% | 3.55 | 0.02% | <1bp |
| General Motors (GM) | BBB | 55.0B | 130.0B | 179.9B | 10.7% | 69.4% | 3.36 | 0.04% | <1bp |
| Ford Motor (F) | BBB- | 45.0B | 160.0B | 198.7B | 8.6% | 77.4% | 2.94 | 0.17% | <1bp |
| Carnival (CCL) | BBB- | 38.0B | 27.0B | 63.9B | 29.7% | 40.6% | 2.88 | 0.20% | 2bp |
| American Airlines (AAL) | B- | 9.0B | 33.0B | 40.6B | 13.7% | 78.0% | 1.74 | 4.06% | 21bp |

## By rating bucket

| Rating | n | Median DD | Median Spread | Names |
|---|---:|---:|---:|---|
| AAA | 2 | 18.45 | <1bp | MSFT, JNJ |
| AA+ | 2 | 16.15 | <1bp | GOOGL, AAPL |
| AA | 2 | 12.38 | <1bp | WMT, AMZN |
| AA- | 2 | 11.50 | <1bp | XOM, META |
| A+ | 2 | 13.98 | <1bp | KO, NVDA |
| BBB+ | 1 | 7.63 | <1bp | VZ |
| BBB | 5 | 4.00 | <1bp | TSLA, T, DAL, INTC, GM |
| BBB- | 3 | 2.94 | <1bp | BA, F, CCL |
| B- | 1 | 1.74 | 21bp | AAL |

Spearman rank correlation, model DD vs S&P rating: **0.94**.

## Findings

**The model orders names roughly the way the agencies do, with one systematic
exception.** Rank correlation is 0.94, and median DD falls monotonically from AAA
through BBB- to B- with a single inversion: A+ (13.98) sits above AA and AA-. That
inversion is entirely NVIDIA. It carries the highest asset volatility in the sample
(44.9%) and still lands at DD 13.22, because its default barrier is 0.2% of asset
value — there is almost no debt for the volatility to act on. Tesla does the same
thing inside the BBB bucket: 54.4% asset vol, 1.1% leverage, DD 7.98 against a BBB
median of 4.00.

**Leverage, not volatility, drives the ranking.** Rank correlation between DD and
leverage is **-0.79**; between DD and *asset* volatility it is **+0.11**, i.e. none.
The two lowest-volatility firms in the sample, Ford (8.6%) and GM (10.7%), sit near the
bottom of the table on leverage of 77% and 69%. Volatility only matters once there is
enough debt for the asset value to fall through — which is why the model's asset vol,
taken alone, carries no signal here, while the *equity* volatility that goes in as an
input correlates -0.67 with DD purely because levered firms have volatile equity.

**Two BBB- names arrive at the same distance-to-default by opposite routes.** Carnival
(2.88) gets there with moderate leverage and high volatility — 40.6% and 29.7%. Ford
(2.94) gets there with very high leverage and very low volatility — 77.4% and 8.6%.
The original three-company finding, that Carnival's DD is roughly half Boeing's (5.57)
despite a shared BBB- rating, survives in the larger sample: the whole BBB- bucket
clusters at a median of 2.94, and Boeing is the outlier within it, not Carnival.

**The model's investment-grade probabilities and spreads are not usable as levels.**
Every name rated A+ or better prints a 1-year PD indistinguishable from zero and a
spread under 1bp, while these issuers trade at tens of basis points in the cash market.
Even B- rated American Airlines implies only 21bp against a market spread in the
hundreds. This is the credit spread puzzle, and it is the expected behaviour of a
single-factor Merton model with a fixed barrier at a 1-year horizon: with lognormal
assets and no jumps, a firm 10 standard deviations from its barrier essentially cannot
reach it within a year. The DD ordering is informative; the PD and spread levels are
not. Extending the horizon, adding jumps, or calibrating a barrier below face value all
raise the levels — `--horizon 5` alone moves American Airlines from 21bp to 299bp and
the BBB- bucket from under 1bp to 35bp, which is in the right neighbourhood for cash
spreads.

## Stress: 30% equity drawdown

Debt and equity volatility held fixed, equity value shocked down 30%.

| Ticker | DD | Stressed DD | Change | Stressed PD |
|---|---:|---:|---:|---:|
| MSFT | 20.41 | 18.88 | -1.52 | ~0 |
| GOOGL | 17.68 | 16.38 | -1.30 | ~0 |
| JNJ | 16.50 | 14.89 | -1.61 | ~0 |
| KO | 14.75 | 13.26 | -1.49 | ~0 |
| AAPL | 14.63 | 13.37 | -1.26 | ~0 |
| WMT | 13.59 | 12.27 | -1.32 | ~0 |
| NVDA | 13.22 | 12.44 | -0.78 | ~0 |
| XOM | 11.60 | 10.46 | -1.14 | ~0 |
| META | 11.39 | 10.41 | -0.98 | ~0 |
| AMZN | 11.16 | 10.12 | -1.05 | ~0 |
| TSLA | 7.98 | 7.37 | -0.61 | 8.2e-12% |
| T | 7.63 | 7.00 | -0.63 | 1.3e-10% |
| VZ | 7.63 | 7.04 | -0.58 | 9.3e-11% |
| BA | 5.57 | 5.03 | -0.54 | 2.5e-05% |
| DAL | 4.00 | 3.65 | -0.35 | 0.01% |
| INTC | 3.55 | 3.23 | -0.32 | 0.06% |
| GM | 3.36 | 3.22 | -0.14 | 0.07% |
| F | 2.94 | 2.85 | -0.09 | 0.22% |
| CCL | 2.88 | 2.66 | -0.23 | 0.40% |
| AAL | 1.74 | 1.69 | -0.06 | 4.59% |

The levered names move *least* in absolute DD terms — Ford loses 0.09 and American
Airlines 0.06, against 1.5 or more for Microsoft and J&J. That is an artifact of the
shock design, not a statement that leveraged issuers are robust. Holding equity
volatility fixed while equity value falls implies a *lower* asset volatility for a
levered firm, and the smaller denominator offsets much of the fall in asset value. In a
real drawdown, equity volatility rises sharply for exactly these names. A shock applied
to asset value with asset volatility held fixed, or one that raises equity volatility
alongside the price fall, is the more informative experiment. What the table does show
is where the *level* ends up: after the shock only AAL clears 1% PD, and the
investment-grade names remain numerically unreachable.
