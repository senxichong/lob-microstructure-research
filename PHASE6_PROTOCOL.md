# Phase 6 preregistration — 6.0.0

Execution-aware economic validation, MSFT 2003-01-03, one previously examined
historical trading day. Freeze this document and the hypotheses before any real
strategy PnL. Prediction research is closed: no fitting, tuning, new features,
ensembles, horizon changes, passive fills or capacity claims.

## Provenance and signal

Discover Phase 2/3 matrices and Phase 4/5 evidence from their manifests. Verify
all frozen evidence and 55 saved Phase 5 models; RESEARCH_LOG is prefix-verified
where an earlier phase predates later append-only entries. Phase 2 must hash to
`a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96`.
Verify the expected Phase 5 metrics/verdicts and identical 11,695 OOS timestamps.
Any unexplained mismatch blocks execution, without rewriting upstream artifacts.

Primary: saved LightGBM regression scores, available at their 1-second grid g,
using frozen source-state features at or before g and the unchanged 1s horizon.
Comparators: saved Ridge and RF regression, exactly the same engine and rules.
Reconstruct VALIDATION predictions only using saved fitted tree objects and
training medians, or Ridge coefficients/intercept/training preprocessing. Verify
reconstructed TEST predictions against saved rows, but never use them to estimate
thresholds. No model.fit call is permitted. Use the exact purged Phase 4 partitions.

Per model/fold use numpy linear empirical validation quantiles q10/q90. Target
is +1 for score >= q90, -1 for score <= q10, otherwise 0. Nonfinite scores give
0; if both tails tie (q10>=q90), stop as degenerate. Freeze validation scores,
thresholds, source model hashes and validation timestamps before any real PnL.
Secondary threshold robustness: LightGBM symmetric 5%, 20%, 30% tails; report
all, primary remains 10%. Strength bins use validation q01/q05/q10/q90/q95/q99;
inclusive upper-tail boundaries, lower-tail boundaries assigned symmetrically.
Omit optional classification execution and larger-size capacity experiments.

## Stateful execution and time

One share-equivalent, |inventory|<=1, no impact, price-taking. Shares are verified
source units; this is not institutional execution. Target changes trade delta
target-current; flips trade two shares at the same prevailing side. No independent
overlapping round trips. Start each of five 39-minute test folds flat with zero
incremental cash/PnL. Aggregate fold increments without multiplying capital.

Primary latency 10ms; scenarios exactly 0/10/50/100/250ms (0/1/5/10/25% of 1s).
Zero is an IDEALIZED UPPER BOUND assumption, not a guaranteed better realized
sample outcome or a realistic latency estimate. Execution request = g+L.
Canonical ordering is (timestamp, source_order); last source row <= request
prevails, including all source ties at that timestamp. Check that row's book_valid,
finite positive bid/ask, ask>bid, and same half-open regular session. Never skip
an invalid latest row to revive an older valid quote. Never use future/nearest
events or interpolate. A valid book persists until its next state; no arbitrary
age cutoff. Record quote age and source order. Source precision is 1ms; local
receive latency and feed completeness remain unknown.

Buy at ask, sell at bid. Required displayed best-side size must be >=abs(delta),
including 2 for flips. Invalid/missing quotes or insufficient depth: record
missed execution, hold prior inventory, retry only at the next scheduled target;
no future fill substitution. Unchanged target has no transaction. On invalid
marks carry last valid mark for indicative equity and count stale marks; never
use that carried mark as a fill. If no mark exists, equity is undefined while
holding; if terminal liquidation cannot execute, FAIL/inconclusive, do not hide it.

No inventory across folds. Mandatory liquidation is requested at test_end-1ms,
the final source-precision instant strictly inside the half-open fold/session.
This fixed risk-close request has no extra signal latency. Last purged grid cell
has no new signal: hold until liquidation. This can make a final holding slightly
longer than one second; targets are rebalanced, not individually horizon-closed.
No next-day or at/after-session-end prices. Same boundary convention for all models.

## Cash, capital, episodes and costs

Cash change = -delta*side_price - abs(delta)*side_price*cost_bps/10000.
Inventory +=delta; equity=cash+inventory*prevailing_mid. Initial trading PnL=0.
Fixed capital denominator is one share at the FIRST valid regular-session Phase 2
midprice (session-open grid has no state); report its timestamp and value. This
small explicit session-start proxy is fixed for all folds, scenarios and controls.
No compounding, variable leverage, annualized return/Sharpe/turnover. Equity
drawdown includes initial zero and fold-stitched cash increments, sampled at every
execution request and every intervening source book event, not just 1s marks.

Costs are 0/.1/.5/1/2/5 bps per ONE-WAY executed notional, all three models at all
five latencies; threshold robustness at 10ms uses all six costs. Spread is already
in side prices, never charged twice. Exact historical fees remain unverified.
Fees do not affect targets/fills; derive cost curves linearly from the same trades.
Signed algebraic break-even = 10000*quote_PnL/gross_notional. If <=0, report no
nonnegative affordable fee (negative root is a required subsidy, not fee capacity).

An episode starts at flat->nonzero and ends at flat or reversal. A flip closes
the old episode and opens the new at the same side quote, allocating one share's
notional/fee to each. Store entry/exit, direction, duration, mid gross, executable
PnL and fees. Win rate is strictly positive closed-episode PnL; zero is not a win.
Trade count counts filled transactions (flip counts once, quantity two), position
changes equal fills; include terminal close. Turnover=sum(abs(delta)*fill_price).
Exposure fractions are duration-weighted over all five full 2340s folds; before
first execution and after terminal close is flat. Holding duration is episode time.

## Reconciled decomposition

MIDPRICE GROSS PNL is the non-executable cash result replacing every fill by its
same-request mid on that scenario's actual inventory path. Report gross at 0ms
and at the scenario latency. Spread_L = sum(abs(delta)*(ask-bid)/2) at actual fills.
Quote_L = Gross_L - Spread_L. Primary attribution uses the matched 0ms path:
latency_effect = Quote_0 - Quote_L (signed, may be negative), so
Gross_0 - Spread_0 - latency_effect - extra_cost = net_L exactly.
Also show latency_mid_effect=Gross_0-Gross_L and
latency_spread_effect=Spread_L-Spread_0, whose sum is latency_effect.
Do not subtract latency again from Gross_L. If fill failures change paths, this
attribution includes those changes and is not a causal price-only effect.

## Frozen diagnostics and uncertainty

Report all folds, leave-one-fold-out sums, positive-fold counts, top 1/5/10 minute
and episode contributions and aggregate with top episodes removed. Minute PnL is
the change in marked equity attributed to UTC clock minutes (net transaction fees
at execution time); negative total denominators give signed ratios, not shares of
positive profit. Removing top episodes is NON-IMPLEMENTABLE EX-POST FRAGILITY TEST.
Latency attribution includes PnL/trade, episode win rate, matched signed execution
price change versus 0ms, BBO price changed at request, and count of BBO price-change
events in (g,g+L]. Depth-only changes are not price changes. Separately count ties.

Adverse-selection diagnostic: direction*(mid_at_fill+h - mid_at_fill), h=10ms,
100ms,1s, using causal as-of valid states strictly within the same fold/session;
reject any intervening invalid state. Weight both equally per transaction and by
quantity; describe, do not infer causal adverse selection. Include closing trades.
Episode economic outcomes are binned by ENTRY validation score percentile range;
this is descriptive and does not alter any trading rule.

Bootstrap: 1000 circular moving 60-second blocks, seed 606060, on the complete
11,700s OOS clock. Preserve terminal intervals/purged cells. Form per-second marked
equity increments and transaction notional. Attribute completed episode PnL/count
to its exit second. Apply identical resampling weights to these four time series;
report percentile 95% intervals for total normalized PnL, PnL/notional bps and
mean episode PnL. These are conditional additive-path uncertainties, not a rerun
of a tradable stitched strategy or independent episode bootstrap; long episodes
can cross sampled blocks. No cross-day inference or model-fitting uncertainty.

Circular signal null: 500 draws, seed 606042. In each test fold independently
rotate saved LightGBM scores by an integer in [600,n-600] retained grid positions,
preserving within-fold order/autocorrelation except the seam; use its unchanged
validation thresholds. Same 10ms engine, 0 extra fee, same termination. Retained
test rows are contiguous 1s; do not rotate across fold/model boundaries. Save all
offsets and PnL/notional. One-sided p=(1+null>=observed)/(501), descriptive of this
conditional within-day null, not proof of exchangeable cross-day evidence.

Random signal: exactly one fixed-seed (606043) uniform(0,1) stream for VALIDATION
then TEST within each fold, validation-derived quantiles for approximate 20% active
frequency. Do not match realized test profitability or redraw. Inversion: negate
LightGBM validation AND test scores, use quantiles of negated validation scores.
Delayed placebo: at g use saved same-fold score at g-600s; missing early 600s are
flat, no wrap or earlier-model substitution. Retain original validation thresholds.
Also report original signal on the identical delayed-eligible time support, with
flat warmup, to separate a shorter observation window from alignment degradation.
All controls use 10ms, zero extra cost. Do not expect sign inversion to negate
net PnL: it approximately negates mid gross but BOTH sides pay spread.
Investigate random/control positive PnL with bootstrap lower>0; no redraw.

## Audit and verdict

Freeze code hashes before real simulation; replay deterministically. At least 50
deterministic trace rows cover entries/exits/flips/unchanged, 0/10/100ms, same-time
activity and quote changes, and profitable/losing episodes when present. Explicitly
report any absent natural category; synthetic tests still cover all transitions.
Independently select and arithmetically check >=10 real paths, including flips if
present. Independent verifier must reconstruct quotes, thresholds, targets, cash,
inventory, episodes and liquidations from upstream data, not import the engine.
Required full tests, upstream hashes, row-order invariance, future-data isolation,
side/depth checks, no missing terminal inventory, reconciliation tolerance $1e-8.

Methodological failure => INCONCLUSIVE and Phase 6 FAIL. Otherwise primary quote
PnL<=0 => PREDICTIVE SIGNAL DOES NOT SURVIVE PRIMARY EXECUTION ASSUMPTIONS.
If positive but normalized bootstrap lower<=0 or shift p>.05 => NO RELIABLE
ECONOMIC EDGE. Positive reliable evidence gets EXECUTION-ROBUST WITHIN-DAY EDGE
only if >=4/5 positive folds, all leave-one-fold-out sums positive, removal of top
10 episodes remains positive, >=1bp additional break-even and all tested latencies
positive. Otherwise MODEST / COST-SENSITIVE WITHIN-DAY EDGE. All verdicts conditional
on this single historical stock-day, never a deployable/profitable-strategy claim.
Methodology may PASS with losses. After report and audit, STOP for Phase 7.
