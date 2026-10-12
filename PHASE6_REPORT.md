# Phase 6 — Execution-aware economic validation

**PREDICTIVE SIGNAL DOES NOT SURVIVE PRIMARY EXECUTION ASSUMPTIONS**

**PHASE 6: PASS.** Methodological validity and economic success are different
questions. The frozen predictive result survives as a forecasting result; the
predeclared aggressive execution strategy loses after crossing the spread.
**MSFT, 2003-01-03; single historical trading day; aggressive top-of-book
simulation; one-share price-taking assumption.** All dollar figures below are
aggregate USD PnL from this one-share experiment, not per-share price forecasts.

## Upstream integrity

Phase 2 SHA-256:
`a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96`.
Phase 3 grid SHA-256:
`1781e33df0ff416cdf9d377fd3a66eed3fed1718f21efec9f465b292276f9e5b`.
Phase 5 regression predictions SHA-256:
`1344dbde0623b6a2af7c91a7fb5d382e18a363d5ad9825b692d78e34b0e8912f`.
Paths were discovered from authoritative manifests, not substituted. All
146 Phase 4/5 evidence entries and 55 saved Phase 5
model artifacts matched. Earlier log versions are verified as append-only prefixes.
Predictive source, reports, feature definitions, matrices, models and scores remain
unchanged. Phase 6 adds code/artifacts and appends the research log/gitignore only.

Saved genuine OOS regression scores are reused: Ridge rho 0.050746889, RF
0.097539968, LightGBM 0.098636590. RF and LightGBM beat Ridge in 5/5 folds;
LightGBM's paired gain is 0.047889701, 60s CI [0.030392513,0.068317717].
Frozen classification Macro F1 remains Logistic 0.302183044, RF 0.345043280,
LightGBM 0.384455023. The prior verdicts RELIABLE NONLINEAR IMPROVEMENT and
RELIABLE NONLINEAR DIRECTIONAL IMPROVEMENT are preserved, not reinterpreted as profit.

Exactly the same five folds and 11,695 test scores per model are used, 12:45–16:00
Eastern. The full OOS clock is 11,700 seconds. The purged final score cell in each
fold receives no fresh signal; existing inventory is held until the declared
terminal close. No train/validation prediction is traded and no model is refit.

## Execution model

Signal time is g. Execution request is g+L, primary L=10ms. Use the latest Phase 2
state at or before request, last physical source order resolving ties. Book state
persists until the next event; never first-after, nearest, interpolated or future
quotes. Validate the latest state rather than skipping an invalid segment. Signal
features remain the original causal source-state features; no future regeneration.

Buy at ask; sell at bid. Inventory is -1, 0 or +1 share. Trade only target minus
current; a flip trades two shares and pays two half-spreads. Each fold starts flat,
ends with a mandatory risk liquidation request at test_end-1ms, and has no capital
reset that erases a loss. These boundary risk orders add no further signal latency.
No inventory crosses a fold/session boundary and no next-day quote is used.

Insufficient displayed top-level depth or an invalid/missing quote produces a
missed instruction and retains inventory until the next scheduled target. No
deeper-book or partial fill is invented. Primary: **0 missing/depth failures**.
The 30% tail stress has 23 depth-limited requests at 10ms and 24 at 0ms; these are
two-share flips against one displayed share. All other tested base scenarios have
zero failures. Every actual fill passes the size check. These failed paths remain
in the stress results. Terminal inventory is zero in every scenario.

Primary filled-quote age: median 102.000ms,
p99 864.000ms, maximum 984.000ms.
Age is time since the last observed source state, not a measured feed delay.
Source accuracy is 1ms; original transport/receive times and completeness are unknown.

Cash changes by -delta*side_quote-extra_fee. Marked equity is cash+inventory*mid.
For drawdown, marks include every source event and execution request; figures
display the 1s sampled path. No passive order, maker rebate, queue-position fill,
or fill-probability model is used. No institutional size or market impact claim.

## Primary signal

**Frozen LightGBM regression; validation-derived symmetric 10% tails; one-second
rebalancing.** Each fold has its own q10/q90 from its selected saved model's
validation predictions. Score>=q90 targets +1, score<=q10 targets -1, otherwise 0.
Validation predictions are recovered with frozen training medians and saved tree
objects; Ridge uses saved coefficients and training preprocessing. Reconstructed
test predictions agree within 1e-12, but test scores never estimate thresholds.
Validation predictions and thresholds were hash-frozen before any real PnL.
No ensemble or classifier execution experiment was added.

## Midprice vs executable result

At 10ms, **MIDPRICE GROSS PNL (NON-EXECUTABLE UPPER BOUND)** is
**$2.4475**. **QUOTE-EXECUTABLE PNL before extra fees is
$-45.6460**. This optimistic quote simulation already assumes our
infinitesimal order can take the observed quote without impact or competing orders.

Fixed normalization capital is **$53.60**, one share at the first valid regular
state, 09:30:00.060 Eastern; the exact session-open grid has no state. This same
denominator is used for all folds/scenarios and never compounded. Primary normalized
PnL is **-85.1604%**, PnL/traded-notional
**-2.691472bps**, and maximum marked drawdown
**$45.6460**. The large normalized loss reflects repeated spread
payments relative to one share of capital, not a percentage move in MSFT.

This denominator is a fixed exposure normalization, not a broker funding model:
margin, short borrow, financing and capital-exhaustion liquidation are not simulated.
Some controls/cost stresses exceed a 100% normalized loss. Their continued fixed
one-share paths require funding and must not be described as feasible funded accounts.
Position size never adapts to equity or results. There are no annualized headlines.

## Spread impact

At actual 10ms fills, spread drag is **$48.0935**, about
19.65 times gross midprice PnL.
Average execution spread is $0.030197; median
$0.029000; mean half-spread
$0.015099. The observed gross price movement is
far too small to pay for entering and exiting through visible quotes.

| Population | Rows | Mean spread $ | Median spread $ | Total paid $ | Mean paid/request $ |
|---|---|---|---|---|---|
| all_requests | 11700 | 0.028719402 | 0.029 | 48.0935 | 0.0041105556 |
| fills | 2989 | 0.030197056 | 0.029 | 48.0935 | 0.016090164 |
| entries | 1398 | 0.030061516 | 0.029 | 21.013 | 0.015030758 |
| exits | 1398 | 0.030261087 | 0.029 | 21.1525 | 0.015130544 |
| flips | 193 | 0.030715026 | 0.029 | 5.928 | 0.030715026 |

Two equivalent exact decompositions prevent double-counting:

- Same-latency: 2.4475 - 48.0935 = -45.6460.
- Zero-latency reference: 2.4355 - 48.1525
  - (-0.0710) - 0 extra fees = -45.6460.

Latency effect is Quote_0-Quote_L. Its mid component is
-0.0120, spread component
-0.0590; the signed total is
**-0.0710**, a small benefit at 10ms in this sample.
Never subtract this again from already-delayed gross PnL. All 126 model/tail/latency/
cost attribution rows reconcile; maximum residual is
3.13e-13 USD. Attribution with depth failures
also includes changed inventory paths and is not a causal latency estimate.

## Latency

| Latency ms | Quote PnL $ | PnL/notional bps | Trades | Changed BBO at fill | BBO changes during delay | Drawdown $ |
|---|---|---|---|---|---|---|
| 0 | -45.717 | -2.695658 | 2989 | 0 | 0 | 45.717 |
| 10 | -45.646 | -2.6914725 | 2989 | 47 | 50 | 45.646 |
| 50 | -45.612 | -2.6894678 | 2989 | 180 | 210 | 45.612 |
| 100 | -45.787 | -2.6997847 | 2989 | 352 | 437 | 45.787 |
| 250 | -45.55 | -2.6858125 | 2989 | 689 | 1050 | 45.55 |

| Latency ms | PnL minus 0ms $ | Delta PnL/trade $ | Delta win rate | Mean signed price change $ |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 10 | 0.071 | 2.3753764e-05 | 0 | -1.1709602e-05 |
| 50 | 0.105 | 3.5128806e-05 | -0.001257071 | -1.1375042e-05 |
| 100 | -0.07 | -2.3419204e-05 | -0.001257071 | 4.0816327e-05 |
| 250 | 0.167 | 5.5871529e-05 | 0 | -2.7768484e-05 |

10/50/100/250ms are 1/5/10/25% of the frozen 1s prediction horizon. Zero latency
is an **IDEALIZED UPPER BOUND assumption**, not realistic receipt/processing time;
it need not have the highest realized PnL when quotes change favorably by chance.
Latency recomputes every quote, rather than subtracting a chosen penalty.
The observed PnL effect is small and nonmonotonic; H3 is not supported as a monotonic
statement on this day. Spread dominates at every tested latency.

## Additional costs

| Extra one-way bps | Extra cost $ | Net PnL $ | PnL/notional bps | Drawdown $ |
|---|---|---|---|---|
| 0 | 0 | -45.646 | -2.6914725 | 45.646 |
| 0.1 | 1.695949 | -47.341949 | -2.7914725 | 47.341949 |
| 0.5 | 8.4797449 | -54.125745 | -3.1914725 | 54.125745 |
| 1 | 16.95949 | -62.60549 | -3.6914725 | 62.60549 |
| 2 | 33.91898 | -79.56498 | -4.6914725 | 79.56498 |
| 5 | 84.797449 | -130.44345 | -7.6914725 | 130.44345 |

Each fill pays abs(delta)*fill_price*cost_bps/10000 on top of quote crossing.
These are generic costs, **not the verified historical Nasdaq/Island 2003 fee
schedule**. Fees do not change the frozen order path. The analytic fee-only root
is -2.691472bps: **there is no nonnegative additional-cost
break-even allowance**. The negative root would require a subsidy after already
paying spread, not positive fee capacity or a proposed rebate strategy.

## Model comparison

| Model | Frozen OOS rho | Trades | Gross mid $ | Spread $ | 10ms quote $ | PnL/notional bps |
|---|---|---|---|---|---|---|
| LightGBM | 0.09863659 | 2989 | 2.4475 | 48.0935 | -45.646 | -2.6914725 |
| Ridge | 0.050746889 | 2529 | 1.348 | 42.732 | -41.384 | -2.9213707 |
| RF | 0.097539968 | 2991 | 2.2485 | 45.7735 | -43.525 | -2.5427287 |

The full model_economic_comparison.csv also gives every generic fee stress and
break-even root for every model; all five latencies use the same engine. LightGBM
has greater gross mid PnL and a less negative per-notional result than Ridge,
but **worse total quote PnL** because it turns over more. RF has the least negative
per-notional outcome of the three. Ranking improvement does not establish economic
improvement; H4 is not supported for aggregate executable dollars. None is positive.

## Threshold robustness

| Symmetric tail fraction | Trades | Time active | Gross mid $ | 10ms quote $ | Spread $ | PnL/notional bps | Signed fee root bps |
|---|---|---|---|---|---|---|---|
| 0.05 | 1955 | 0.12111111 | 1.7675 | -28.656 | 30.4235 | -2.6351754 | -2.6351754 |
| 0.1 | 2989 | 0.23717949 | 2.4475 | -45.646 | 48.0935 | -2.6914725 | -2.6914725 |
| 0.2 | 4056 | 0.42546726 | 2.8325 | -65.82 | 68.6525 | -2.7023054 | -2.7023054 |
| 0.3 | 4467 | 0.60922701 | 3.4325 | -79.374 | 82.8065 | -2.7125506 | -2.7125506 |

All four thresholds are validation-only and were predeclared. All lose. The primary
10% result was saved and hash-frozen before these robustness runs. No threshold
is promoted, optimized or selected for profitability. The 30% depth failures are
included rather than replaced with nonexistent liquidity.

## Turnover

Primary **2,989 filled transactions**, 193 two-share
flips, 1,591 closed episodes. Gross one-way traded notional is
**$169,594.898**, or $14.4953
per OOS second. A flip counts once as a transaction and twice in traded quantity.
Position changes equal fills; unchanged targets have no trade. No capital is
allocated separately to each overlapping prediction.

Duration-weighted long/flat/short fractions are
12.0342% / 76.2821% /
11.6838%; time in market is **23.7179%**.
Mean episode duration is 1.744186s. Repeated same-direction
targets can hold beyond 1s; the prediction horizon remains 1s. Mean/median episode
PnL are $-0.02869013 / $-0.02800000; strictly positive
episode win rate is 1.0685%. Zero-PnL episodes are not wins.

## Economic uncertainty

| Metric | Observed | 95% lower | 95% upper |
|---|---|---|---|
| normalized_pnl | -0.85160448 | -0.92262966 | -0.78671292 |
| pnl_per_notional_bps | -2.6914725 | -2.8187112 | -2.5512909 |
| mean_episode_pnl | -0.028690132 | -0.030073853 | -0.027223636 |

1,000 circular moving 60-second block draws, seed 606060, preserve the full
11,700-second clock. Resample additive marked PnL and traded notional jointly;
completed episode PnL/count are attributed to exit seconds for the mean-episode
ratio. This is a conditional path bootstrap, not execution on a synthetic stitched
market, independent episodes, or cross-day confidence. Episode boundaries can
cross blocks; model fitting, regime changes and previous research exposure are
not resolved by the interval. All three primary intervals are entirely negative.

500 within-fold large circular score rotations, seed 606042, preserve local score
order except the seam, with offsets from 600 to n-600 seconds and unchanged
validation thresholds. The same execution/depth/termination engine yields a
null mean of $-45.213958; null central 95% range
[$-46.309975, $-44.063750]. Observed quote PnL is at
the **22.20th percentile**; one-sided quote-PnL
p=0.778443, per-notional p=0.774451.
The observed economic path does not exceed this null. A predictive rho near 0.10
can coexist with this: score alignment may improve mid moves while selecting
different spread/turnover states. The null is within-day conditional, not a
cross-day exchangeability or causal test.

The independent null replay records 144 depth-limited instructions across 90 of
500 draws (maximum three per draw). These follow the same no-fill/hold policy;
no terminal inventory remains. They are retained in each null outcome and listed
in null_execution_quality.csv, rather than silently granting liquidity.

## Concentration

| Model | Fold | Mid gross $ | Spread $ | Quote PnL $ |
|---|---|---|---|---|
| LightGBM | 1 | 0.2675 | 8.6145 | -8.347 |
| LightGBM | 2 | 0.5635 | 10.0235 | -9.46 |
| LightGBM | 3 | 0.5305 | 10.2015 | -9.671 |
| LightGBM | 4 | 0.61 | 8.588 | -7.978 |
| LightGBM | 5 | 0.476 | 10.666 | -10.19 |
| Ridge | 1 | 0.1715 | 8.5735 | -8.402 |
| Ridge | 2 | 0.2035 | 5.8885 | -5.685 |
| Ridge | 3 | 0.328 | 10.474 | -10.146 |
| Ridge | 4 | 0.417 | 7.981 | -7.564 |
| Ridge | 5 | 0.228 | 9.815 | -9.587 |
| RF | 1 | 0.1945 | 8.1795 | -7.985 |
| RF | 2 | 0.6275 | 8.4695 | -7.842 |
| RF | 3 | 0.5035 | 9.2735 | -8.77 |
| RF | 4 | 0.565 | 9.231 | -8.666 |
| RF | 5 | 0.358 | 10.62 | -10.262 |

All three models have **0/5 positive quote-PnL folds**. No losing fold is removed.

| Diagnostic | Top k | Omitted fold | Contribution $ | Signed fraction of total | Remaining PnL $ |
|---|---|---|---|---|---|
| minute | 1 |  | -0.017 | 0.00037243132 | -45.629 |
| minute | 5 |  | -0.115 | 0.0025193883 | -45.531 |
| minute | 10 |  | -0.3255 | 0.0071309644 | -45.3205 |
| leave_one_fold_out |  | 1 |  |  | -37.299 |
| leave_one_fold_out |  | 2 |  |  | -36.186 |
| leave_one_fold_out |  | 3 |  |  | -35.975 |
| leave_one_fold_out |  | 4 |  |  | -37.668 |
| leave_one_fold_out |  | 5 |  |  | -35.456 |

Every one-minute primary marked-PnL bin is negative; even the best minutes lose.
Minute attribution uses clock equity increments, not exit-only episode totals.
All leave-one-fold-out results remain negative. For a negative denominator,
signed fractions are accounting ratios, not shares of positive profits.

| Top episodes | Contribution $ | Signed fraction of total | After exclusion $ |
|---|---|---|---|
| 1 | 0.012 | -0.0002628927 | -45.658 |
| 5 | 0.049 | -0.0010734785 | -45.695 |
| 10 | 0.061 | -0.0013363712 | -45.707 |

Episode exclusion is **NON-IMPLEMENTABLE EX-POST FRAGILITY TEST**. Removing the
best 1/5/10 episodes only deepens the aggregate loss; exclusions never become rules.

## Controls

| Control | Trades | Time active | Gross mid $ | Quote PnL $ | PnL/notional bps |
|---|---|---|---|---|---|
| RandomSignal | 3952 | 0.19760684 | -0.149 | -59.885 | -2.6903797 |
| Inversion | 2989 | 0.23717949 | -2.4475 | -50.541 | -2.9801017 |
| Delayed600 | 2122 | 0.16803419 | -0.16 | -32.825 | -2.7248241 |
| DelayMatchedOriginal | 2242 | 0.17905983 | 1.855 | -34.481 | -2.7171752 |

One fixed random seed, 606043; validation uniform scores determine thresholds,
not test ranks. It approximates the nominal 20% active frequency, but does not
match the real signal's persistence: 3,952 transactions versus 2,989. This is a
coarse activity control, not an exact turnover-matched comparison. It loses,
with an entirely negative normalized-PnL interval; no redraw.

Inversion negates both validation and test scores. It exactly reverses the primary
mid gross to -$2.4475, while both directions pay the same spread drag; therefore
quote PnL is not the negative of the original quote PnL. This is expected accounting,
not a failed control or an invitation to trade the inversion.

The 600s delayed signal uses only earlier same-fold saved scores; first 600s stay
flat, no circular wrap. Its mid gross is -$0.1600 versus +$1.8550 for the original
signal with the same flat warmup/eligible clock. The delayed total quote loss is
smaller because turnover is smaller; per-notional PnL is slightly worse. Thus local
midprice alignment weakens, but absolute-dollar comparisons need exposure context.
Controls are sane; none supplies a reliable positive economic result.

## Adverse selection

| Post-fill ms | Eligible fills | Unavailable | Mean signed move $ | Quantity-weighted $ | Positive fraction |
|---|---|---|---|---|---|
| 10 | 2989 | 0 | 2.0408163e-05 | 1.9170333e-05 | 0.0090331214 |
| 100 | 2989 | 0 | 3.7638006e-05 | 3.3155248e-05 | 0.06189361 |
| 1000 | 2989 | 0 | 0.0002295082 | 0.00022281584 | 0.26564068 |

Direction*(future_mid-execution_mid), positive for favorable buys or sells.
All marks are causal as-of states, within the fold, with invalid intervals rejected.
These small average favorable movements (roughly $0.000230 at 1s per transaction)
are far below the $0.015099 mean half-spread. Closing trades are included.
This describes post-fill price behavior, not a causal adverse-selection estimate.

| Entry validation percentile | Episodes | Mean mid gross $ | Mean quote $ | Win rate |
|---|---|---|---|---|
| short_0_1 | 89 | 0.0019325843 | -0.026 | 0 |
| short_1_5 | 354 | 0.0016850282 | -0.029418079 | 0 |
| short_5_10 | 332 | 0.001189759 | -0.029665663 | 0.012048193 |
| long_90_95 | 381 | 0.0011286089 | -0.02928084 | 0.015748031 |
| long_95_99 | 297 | 0.0018585859 | -0.028225589 | 0.013468013 |
| long_99_100 | 138 | 0.0021884058 | -0.02557971 | 0.02173913 |

The most extreme bins have greater mean mid gross in both directions, supporting
H2 descriptively. Every bin remains negative after quote crossing. This analysis
uses entry scores, not test-fitted quantiles, and never changes thresholds.

## Audit and reproducibility

Full pytest: **139 passed, zero failures/errors**; Phase 1–5 tests unchanged.
Synthetic cases cover every state transition, flips, quote sides, source ties,
as-of latency, missing/latest-invalid books, inadequate depth, terminal failure,
fees/cash/inventory/spread/turnover/episode arithmetic, validation thresholds,
future isolation, row permutation, event-level marking and block aggregation.

Real two-run primary paths match exactly. Mutating 64,191 later source states
does not change earlier execution, and shuffled input rows restore the identical
authoritative chronological path. Independent integer-price replay imports no
production execution/metric functions: **13,993 checks, zero critical
failures**, maximum cash discrepancy **$3.27e-13**,
maximum |inventory|=1. It rebuilds validation predictions/quantiles, all scenario
targets, side prices, cash, inventory, episodes, terminations and cost arithmetic;
checks primary-family event drawdowns and 1s paths; independently reproduces all
1,000 primary bootstrap draws and all 500 null execution paths/offsets. The separate
cost-stress ledger is checked transaction by transaction against independent cash.

16 independently selected real paths have explicit hand-formula
cash/episode comparisons in manual_pnl_verification.csv; 87 traces
cover entries/exits/flips/unchanged, positive/negative episodes, source ties,
0/10/100ms and quotes changing during latency. No claim of an outside human audit
is made; these are independently coded and inspected arithmetic checks.

Hypotheses: H1 and H5 supported on this day (spread consumes the predictive edge);
H2 supported descriptively for gross edge; H3 not monotonically supported; H4 not
supported for total executable dollars. No adverse result prompted retuning.

## Limitations

One MSFT day in 2003, repeatedly examined through earlier phases; no untouched day,
other-stock, regime or modern-market confirmation. OOS model parameters do not make
the research process pristine. Visible legacy Island/INET-lineage venue book only,
not consolidated NBBO. Historical fee schedule, transport completeness, receive
latency, borrow/financing, trading-status certification and competitor access to
the quote remain unverified. Features retain their frozen source-state staleness.

No passive queue model, hidden liquidity, maker fills/rebates, impact or
institutional-capacity estimate. One-share execution is an optimistic economic
diagnostic. Quote PnL is simulated, not realized brokerage profit. Fixed capital
normalization does not certify funding feasibility. A negative result here is
evidence against these primary assumptions, not proof that every execution style,
market or future day is uneconomic. No new strategy variant is authorized.

## Artifacts and reproduction

PHASE6_PROTOCOL.md, protocol_freeze.json and signal_freeze.json precede real PnL.
primary_result_freeze.json precedes robustness. execution_manifest.json records
executed production code hashes. PHASE6_MANIFEST.json freezes the final evidence.
Eleven PNG/SVG figure pairs identify the full historical/execution scope and were
visually inspected. Figures: [mid vs quote](reports/phase6/figures/01_mid_vs_quote.png),
[models](reports/phase6/figures/02_models.png),
[decomposition](reports/phase6/figures/03_decomposition.png),
[latency](reports/phase6/figures/04_latency.png),
[costs](reports/phase6/figures/05_costs.png),
[thresholds](reports/phase6/figures/06_thresholds.png),
[folds](reports/phase6/figures/07_folds.png),
[strength](reports/phase6/figures/08_strength.png),
[post-fill moves](reports/phase6/figures/09_adverse_selection.png),
[shift null](reports/phase6/figures/10_shift_null.png), and
[equity/drawdown](reports/phase6/figures/11_equity_drawdown.png).

prepare_phase6.py verifies upstream and reconstructs validation scores without
fitting; run_phase6.py refuses to overwrite frozen real PnL; verify_phase6.py
replays independently; build_phase6_report.py and plot_phase6.py read saved results.
The inherited requirements.phase5.txt environment suffices; no dependency added.

execution_ledger.parquet stores all zero-extra-cost fills and missed instructions;
scheduled_execution.parquet includes unchanged targets and terminal instructions.
execution_cost_ledger.parquet materializes all fee scenarios with per-fill costs
and cumulative cash. episode_ledger.csv stores zero-extra-fee episodes for every
base scenario (additional_cost=0); stressed episode fee is exactly
(entry_quote+exit_quote)*bps/10000, summarized in scenario_metrics.csv. Every
timestamp is UTC integer nanoseconds; source precision remains 1ms. Row-level
market/score/ledger artifacts and real traces stay local-only under inherited
redistribution restrictions. Aggregate tables, report, code and figures are reviewable.

**STOP after Phase 6. No predictive retuning, threshold search, passive execution
or Phase 7 red-team work has begun.**
