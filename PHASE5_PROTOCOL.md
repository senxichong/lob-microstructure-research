# Phase 5 preregistration — 5.0.0

Question: does nonlinear extraction from exactly the Phase 4 information set
add reliable within-day OOS prediction beyond frozen Ridge / multinomial Logistic?
MSFT, 2003-01-03; one previously examined historical day. This is not an untouched
confirmatory dataset. No execution, PnL, new features, horizon search or Phase 6.

## Immutable framework

Read the authoritative Phase 4 manifest, reports, saved OOS tables and fold CSV.
Primary 1-second grid, GRID_FWD_MID_RET_1S and exact DOWN/FLAT/UP labels unchanged.
Reuse Phase 4 fold_manifest.csv and partitions(..., embargo=0) without modification:
five expanding chronological trains, 39-minute validation/test blocks; available
requested target timestamp < own partition end, including test-end purges.
All models must match the frozen 11,695 test timestamps exactly. Do not refit
Ridge/Logistic. No extra embargo or horizon/expanded-feature/class-balance experiment.
Training-only finite median imputation, no scaling for trees, no missing indicators.
No refit on train+validation. Fixed feature whitelist; unknown/future fields rejected.

## Candidate sets, frozen before any RF/LightGBM fit

Four candidates for each family, reused for regression and classification.
RF: n_estimators=150, bootstrap=True, n_jobs=1, random_state=505.
Candidate order (max_depth, min_samples_leaf, max_features):
0=(3,100,"sqrt"), 1=(6,100,1.0), 2=(6,20,"sqrt"), 3=(None,100,1.0).
Other sklearn defaults, squared-error regression / gini classification,
class_weight=None; natural class distribution, no balancing or resampling.

LightGBM 4.6.0: gbdt, CPU, n_jobs=1, deterministic=True, force_col_wise=True,
random_state=505, data_random_seed=505, feature_fraction_seed=505, bagging_seed=505,
subsample=1, subsample_freq=0, colsample_bytree=1, max_bin=63, verbosity=-1,
reg_alpha=0, reg_lambda=0. Regression L2; classification objective multiclass.
Candidate order (num_leaves,max_depth,learning_rate,n_estimators,min_child_samples):
0=(7,3,.03,100,200), 1=(7,3,.08,200,100),
2=(15,4,.03,200,100), 3=(15,4,.08,100,200).
No early stopping; selected rounds are fixed candidate rounds, never a test eval_set.
References: [LightGBM parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html),
[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

Fit candidates on TRAIN only. Select regression on validation Spearman, classifier
on external validation Macro F1. Tie within 1e-12: first candidate in frozen order.
Constant-score validation correlation is undefined and ranked -infinity; if all
candidates are undefined, stop. Fixed probability order DOWN/FLAT/UP and argmax.
Per fold/task, additionally select the better validation-scoring FAMILY (RF wins
ties within 1e-12). This ValidationSelected pipeline is the primary "best nonlinear"
comparison; never select its family using test results. Report both families too.
No grid expansion or tuning after validation/test inspection. Training/validation
scores and test gaps for every selected model are retained and clearly distinguished.

## Metrics and dependence

Regression primary Spearman; Pearson, MAE, MSE, R2 secondary, aggregate and per fold.
Classification primary Macro F1, then balanced accuracy, accuracy, epsilon-clipped
log loss and per-class precision/recall/F1/confusion. Retain natural FLAT prevalence.
Frozen Always-FLAT and Logistic comparisons; OBI_L1 and Phase 4 observed best-single
(OBI_L1) explicitly included, alongside Phase 4's conservative best-single inference.
Scores are calculated only after reloading saved OOS predictions.

1,000 paired circular 60-second moving-block bootstrap draws, seed 505060,
on the full 11,700-second OOS clock, preserving all five purged holes. Recompute
Spearman ranks per draw. Same clock weights for regression and classification.
Report observed and bootstrap-mean deltas, percentile 95% intervals for RF,
LightGBM and ValidationSelected versus their frozen linear benchmarks.
Also report model rho intervals, static/flow/full differences and control intervals.
The validation-selected comparison is primary; family-specific intervals are
secondary, without multiplicity-adjusted discovery claims. Intervals condition on
fitted models and previous research choices; no refitting bootstrap or iid p-values.
Circular seams and intraday nonstationarity limit interpretation. Effective
independent sample size is much smaller than raw rows; do not invent its value.

## Fixed controls

LightGBM regression controls reuse each fold's selected FULL hyperparameters:
1. Random labels: permute contiguous 60-observation TRAINING label blocks,
including the shorter last block, once, seed 505028+fold_id. No test/validation
labels enter the permutation. Preserve the multiset and within-block order.
2. Temporal placebo: rotate TRAINING targets by exactly 600 positions (seconds
on the contiguous retained clock), once. This is separate from block permutation.
3. Random feature: append exactly the frozen RANDOM_CONTROL, seed 161803,
with the same selected full configuration; no redraw or extra tuning. Record
validation/test changes and native/permutation importance. Flag material gain
if test paired CI lower>0 AND delta rho>=.01, or validation gain>=.01 in 4/5 folds.
Flag dominance if random feature has largest native gain in >=4/5 folds.

Investigate either randomized-target control if rho CI lower>0 or positive rho
>=max(.01,.8*real-LightGBM-rho); do not redraw. Unexplained strong controls FAIL.
Whitelist test intentionally adds FUTURE_RETURN_1S in isolation; it must raise
before fitting. Actual primary artifacts must never contain it. Repeat actual
fold 1 fits after mutating all later upstream payload rows in memory, preserving
authoritative ordering keys; earlier tuning, predictions and metrics unchanged.
Mutate test labels only: fitted models/tuning/predictions unchanged, metrics change.
Shuffle storage and restore chronology. Repeat the complete primary RF/LightGBM
experiment twice, tolerance 1e-12, fixed single-thread configuration.

## Predeclared interpretation diagnostics; never feed back to fitting

Native RF impurity and LightGBM gain + split importances, per fold. Impurity/gain
favor available split opportunities and correlated substitutes; not causality.
Held-out permutation: circularly rotate ONE test feature by fixed offsets
300,600,900 seconds within its test fold; all other columns and labels fixed.
Report Spearman / Macro F1 degradation and fold variation. This preserves a
column's local order with an artificial seam, breaks joint feature dependence,
and is interpretation only, never feature selection. Missing end cell is outside
the retained contiguous 2339-sample fold interval.

Response shape: LightGBM full regression, OBI_L1, MICROPRICE_MINUS_MID, OFI_1S,
OFI_100MS. 1D partial dependence at unique TRAIN empirical quantiles .05,.15,...,.95;
average over 200 evenly spaced TEST feature rows (no target use). No threshold rule.
2D partial dependence for (OBI_L1,OFI_1S), (OBI_L1,RELATIVE_SPREAD),
(OFI_1S,EVENT_RATE_1S), (MICROPRICE_MINUS_MID,OBI_L1), at unique TRAIN quantiles
.1,.3,.5,.7,.9; same 200-row background. Save grid and RMS after removing row/column
additive means. Off-support combinations and correlated features limit causal
interpretation; non-additivity is a model property, not verified economic interaction.

Static/flow/full LightGBM regression uses Phase 4 memberships; static/flow independently
select among the same four candidates on validation only. No primary replacement.
Prediction Spearman/Pearson: Ridge/RF/LightGBM. Disagreement: per-fold z-scores of
Ridge and LightGBM predictions using each model's TRAIN prediction mean/SD;
pooled top decile absolute difference (linear empirical quantile, inclusive ties).
EX-POST DESCRIPTIVE ONLY. Save feature means, realized outcome summaries, fold shares.
Calibration: fixed [0,.1,...,1] bins, left-closed/right-open except final inclusive,
UP/DOWN for Logistic/RF/LightGBM, empty bins retained. No probability recalibration.

## Verdicts, fixed rules

Primary validation-selected pipeline: RELIABLE NONLINEAR IMPROVEMENT (regression)
or RELIABLE NONLINEAR DIRECTIONAL IMPROVEMENT (classification) requires positive
paired CI lower, >=4/5 fold wins, sane controls and no severe fit gap. Severe gap
means median(train-primary-metric minus test-primary-metric)>.15 for regression
or >.20 for classification. These conservative thresholds are reporting rules.
If delta>0 but any reliability condition fails: MODEST BUT UNCERTAIN NONLINEAR
IMPROVEMENT / MODEST BUT UNCERTAIN NONLINEAR DIRECTIONAL IMPROVEMENT.
If BOTH individual families' paired upper bounds<0: NONLINEAR MODELS UNDERPERFORM
RIDGE / NONLINEAR CLASSIFIERS UNDERPERFORM LOGISTIC. Otherwise NO RELIABLE
NONLINEAR IMPROVEMENT. Report any highest TEST family only descriptively; do not
reselect or refit. Methodology can PASS with negative performance when all audits,
reproducibility, controls, full tests and independent verification pass.
After Phase 5 STOP. No Phase 6 work.
