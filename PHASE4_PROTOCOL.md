# Phase 4 preregistration — version 4.0.0

Scope: MSFT, 2003-01-03, one historical session. Preserve Phase 1–3 code,
data, definitions and reports. Only append to the research log and gitignore.
No nonlinear models, new feature transforms, execution, PnL or Phase 5.

## Estimand and folds

Use the frozen Phase 3 1-second grid and GRID_FWD_MID_RET_1S; classes are
DOWN, FLAT, UP from exact mid movement. Keep FLAT and natural class frequencies.
No staleness filtering. Source features remain carried from their post-event
Phase 2 states. Canonically sort the grid by its unique timestamp before splitting.
Ten equal 39-minute wall-clock blocks span [09:30,16:00) Eastern.
Fold k=1..5: train blocks 1..k+3, validate block k+4, test block k+5.
All intervals are half-open. Freeze exact UTC boundaries in fold_manifest.csv
before fitting. Earlier tests can legitimately enter subsequent training/validation.

Use available targets only. Purge using the requested target timestamp g+1s,
not the possibly stale target-state timestamp. Each partition requires target
timestamp strictly before its end; thus even TEST labels cannot cross its end.
Assert disjointness within each fold and max training target < validation start,
max validation target < test start. Primary embargo is zero. The sole robustness
uses an additional 10-second gap at the END of training and validation:
target < next boundary minus 10 seconds. Test timestamps remain identical.
This is a pre-boundary information embargo; do not tune it.

## Preprocessing and fitting

Seven features are frozen in PHASE4_FEATURE_SET.md. Training-only finite medians
impute missing features, then training-only means and population standard
deviations scale them. Constant scale becomes 1; an entirely missing training
column is an error. No missing indicators, clipping, transforms or feature search.
Save medians, means, scales and all standardized coefficients/intercepts per fold.
Validation and test use these unchanged. Do not refit on train+validation.

OLS: unregularized least squares with intercept. Ridge minimizes sum squared
errors + alpha times squared coefficients (unpenalized intercept), using SVD;
alpha candidates [0, .01, .1, 1, 10, 100]. Select greatest validation Spearman;
ties within 1e-12 choose the largest alpha. Alpha zero uses OLS.
Multinomial logistic: lbfgs, L2, C [.01, .1, 1, 10], max_iter=2000,
tol=1e-8, random_state=404, class_weight=None. Select validation Macro F1;
ties within 1e-12 choose smallest C. All three classes required in training.
Convergence warnings are errors. Argmax in fixed DOWN/FLAT/UP order determines
classes. One BLAS thread for reproducibility. No probability threshold tuning.
OLS/Ridge and logistic settings follow the official
[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
and [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
definitions; package versions are separately pinned.

Baselines: zero return and training mean return; Always-FLAT and training
empirical class probabilities with argmax. Classification log loss clips
probabilities at float64 epsilon, then renormalizes; this makes one-hot baseline
loss finite, not an estimate of a calibrated probabilistic forecaster.

## Metrics, inference and comparisons

Primary regression metric Spearman; secondary Pearson, MAE, MSE and R2.
Primary classification metric Macro F1 over all three classes; secondary balanced
accuracy, accuracy, log loss, confusion, per-class precision/recall/F1.
Undefined constant-prediction correlations are NaN, never replaced by zero.
All final metrics are recomputed from persisted OOS predictions; report per fold
and concatenated test periods. No training/validation metrics described as OOS.

For OLS, full Ridge, static/flow Ridge, three single features, and both random
controls: 1,000 paired circular 60-second moving-block bootstrap draws, seed
404060, preserving the complete 11,700-second OOS clock with purged holes.
Recalculate ranks after resampling. Percentile 95% CIs condition on fitted
models; they do not include refitting/selection uncertainty. Same indices for
all paired differences. Circular target shifts: 1,000 offsets uniformly from
600 through N-600 seconds inclusive, seed 404600; preserve missing clock cells.
Two-sided empirical p=(1+count(abs(null)>=abs(observed)))/1001. These approximate
stationarity-based diagnostics have artificial circular seams, intraday regime
limitations and no independent-day interpretation. No naive iid primary p-values.

Compare raw OBI_L1, MICROPRICE_MINUS_MID and OFI_1S against Ridge on exactly
identical test timestamps. Report each paired difference and Ridge minus the
maximum of the three single-feature correlations, recalculating that maximum
in every bootstrap draw. The empirical best single is explicitly descriptive,
not a separately selected primary feature. OBI_L1 remains the named benchmark.
Also report full-minus-static and random-control paired differences.
Bootstrap logistic-minus-Always-FLAT Macro F1 with the same clock weights.

Four fixed Ridge ablations remove imbalance, microprice, OFI (both windows), or
context (spread/activity/timing), with the same validation-only alpha grid.
Static=[OBI_L1,MICROPRICE_MINUS_MID,RELATIVE_SPREAD].
Flow=[OFI_100MS,OFI_1S,EVENT_RATE_1S,TIME_SINCE_PREVIOUS_TRADE].
Full is their union. No ablation becomes the primary model after results.
The five test blocks are the only time-of-day regimes; opening performance
cannot be evaluated from this OOS design.

Probability calibration: fixed bins [0,.1,.2,...,1], left-closed/right-open
except final inclusive, empty bins retained, for UP and DOWN. Top-decile
P(UP)/P(DOWN), empirical linear 90th percentile with inclusive ties, is an
EX-POST DESCRIPTIVE diagnostic only. No calibration fit or trade entries.
Omit class-balanced and additional-horizon model variants to keep scope bounded.

## Controls and audits

Before fitting: independently audit OFI price/size signs and rolling windows;
run existing hand tests. Verified upstream defect means STOP, not repair.
Preserve negative Phase 3 100ms and positive 1s OFI findings without sign flips.
Random-label Ridge rotates TRAINING labels by exactly 600 positions (600 seconds
on the contiguous retained training clock). Reuse primary validation-selected
alpha to isolate alignment; never rotate validation/test labels into training.
No alternative offsets. Investigate if random-label rho is positive and at
least 80% of full Ridge rho, or its CI excludes zero positively. Record all
findings; do not redraw. Suspicious unexplained predictive controls mean FAIL.
Reuse exactly the frozen RANDOM_CONTROL (seed 161803); append to full Ridge,
select alpha by the same validation rule. A material improvement means paired
95% CI above zero AND delta rho >= .01; inspect standardized weight dominance.

Re-run complete deterministic fits twice (absolute tolerance 1e-12). Mutate all
raw Phase 2 rows after fold 1 test end in memory, rebuild only the earlier
causal grid, and compare fits, preprocessing, tuning, predictions and metrics.
Mutate fold 1 TEST labels, keeping availability fixed: fit/tuning/predictions
must stay fixed; metrics must change. Shuffle row storage, restore authoritative
timestamp/source ordering, and repeat. Source artifacts are never overwritten.
Unit tests cover boundaries, purging, preprocessing, selection, tagging,
isolation, controls, mutations and determinism. Independent verifier reloads
OOS tables, recomputes metrics/confusion/membership and report values.

## Verdict rules

Regression CLEAR INCREMENTAL LINEAR PREDICTIVE VALUE if Ridge-minus-best-single
paired CI lower >0, delta >=.02 and Ridge positive in at least 4/5 folds.
MODEST INCREMENTAL LINEAR PREDICTIVE VALUE if lower >0, delta >0 and positive
in at least 4/5 folds. LINEAR MODEL UNDERPERFORMS SIMPLE FEATURES if paired CI
upper <0. Otherwise NO RELIABLE INCREMENTAL VALUE OVER SINGLE FEATURES.
Classification CLEAR DIRECTIONAL IMPROVEMENT OVER BASELINE if Macro F1 paired
CI lower >0, delta >=.05 and positive difference in at least 4/5 folds;
MODEST DIRECTIONAL IMPROVEMENT for lower >0, delta >0 and at least 4/5 folds;
otherwise NO RELIABLE IMPROVEMENT. These are research labels, not trading claims.
Methodology PASS additionally requires correct leakage audits, reproducibility,
sane controls, immutable upstream, independently reproducible metrics and zero
test failures. Negative predictive results alone do not fail methodology.
Stop after Phase 4.
