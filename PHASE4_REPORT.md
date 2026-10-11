# Phase 4 — Linear baselines and intraday walk-forward validation

Regression verdict: **NO RELIABLE INCREMENTAL VALUE OVER SINGLE FEATURES**.
Classification verdict: **MODEST DIRECTIONAL IMPROVEMENT**.
Methodology decision: **PHASE 4: PASS**. Independent verification receipt:
reports/phase4/independent_verification.json (checks the final report hash).

**MSFT — 2003-01-03 — single historical trading day.** This is conditional
within-day research, not independent-day validation, modern-market evidence or
an execution result. Models are only OLS, Ridge and multinomial Logistic.

## Source integrity

Phase 2 SHA-256: `a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96`.
Phase 3 grid SHA-256: `1781e33df0ff416cdf9d377fd3a66eed3fed1718f21efec9f465b292276f9e5b`.
All 66 Phase 3 final evidence hashes matched before work. The saved upstream
freeze additionally protects tracked source, tests, definitions, reports and
config; only append-only research-log and gitignore additions are authorized.
The grid still has 23,400 points and 23,398 available 1s targets. No upstream
rebuild occurred. Phase 3's console-serialization-only correction remains
documented in its original audit; no numerical artifact was changed here.

## Fold design

Protocol/feature set/manifest/OFI audit were hash-frozen before any fit at
2026-10-11T06:55:00.023761+00:00.
Session blocks are 39 minutes, [09:30,16:00) Eastern. Boundaries below are UTC;
subtract five hours for this date's Eastern time. Every interval is half-open.

| Fold | Train start | Train end / validation start | Validation end / test start | Test end |
|---|---|---|---|---|
| 1 | 2003-01-03T14:30:00+00:00 | 2003-01-03T17:06:00+00:00 | 2003-01-03T17:45:00+00:00 | 2003-01-03T18:24:00+00:00 |
| 2 | 2003-01-03T14:30:00+00:00 | 2003-01-03T17:45:00+00:00 | 2003-01-03T18:24:00+00:00 | 2003-01-03T19:03:00+00:00 |
| 3 | 2003-01-03T14:30:00+00:00 | 2003-01-03T18:24:00+00:00 | 2003-01-03T19:03:00+00:00 | 2003-01-03T19:42:00+00:00 |
| 4 | 2003-01-03T14:30:00+00:00 | 2003-01-03T19:03:00+00:00 | 2003-01-03T19:42:00+00:00 | 2003-01-03T20:21:00+00:00 |
| 5 | 2003-01-03T14:30:00+00:00 | 2003-01-03T19:42:00+00:00 | 2003-01-03T20:21:00+00:00 | 2003-01-03T21:00:00+00:00 |

Train blocks 1–4 through 1–8; validation blocks 5–9; test blocks 6–10. Each
fold's partitions are disjoint. Prior tests may enter later folds chronologically.
Requested target time g+1s must be strictly before its own partition end.
Stale target-state timestamps never replace that information boundary.
Primary training retained counts: 9358, 11698, 14038, 16378, 18718; validation
2339 each; test **2339 each, 11,695 unique OOS samples**. The opening unavailable
training row and the last target-crossing row are excluded. The last session
target is unavailable; other block-end targets are purged. Counts distinguish
unavailability from purge in purge_summary.csv.

Primary embargo is zero; the fixed robustness additionally excludes targets
within ten seconds before train/validation ends. It removes ten more rows from
each earlier partition, leaving identical tests. No test feature or label fits
the imputer, scaler or model. No train+validation refit follows selection.
OOS coverage is **12:45–16:00 Eastern**; no opening-period performance claim is possible.

## Features and preprocessing

Primary features: OBI_L1, MICROPRICE_MINUS_MID, OFI_100MS, OFI_1S, RELATIVE_SPREAD, EVENT_RATE_1S, TIME_SINCE_PREVIOUS_TRADE.
Depth variants OBI_L5, OBI_L10, WOBI_L10 are excluded to keep the predeclared
set compact. OBI_L1/microprice rank correlation was about .945 in Phase 3;
conditional coefficients therefore cannot identify independent economic causes.
Context features were retained for fixed economic roles despite weak marginal
associations. All missing features use each fold's training median, followed
by training mean/population-SD standardization. Save every parameter in
preprocessing_parameters.csv and fitted_models.json; coefficients use those units.
No nonlinear transforms, class balancing, FLAT removal or feature reselection.

OLS is unregularized. Ridge alpha=[0,.01,.1,1,10,100], selected on validation
Spearman with larger-alpha tie preference. Logistic C=[.01,.1,1,10], selected
on validation Macro F1 with smaller-C tie preference. Fixed lbfgs, 2000 iterations,
1e-8 tolerance, seed 404; convergence warnings abort. Validation scores for
every candidate are saved; selected values below are not test-selected.

| Fold | Ridge alpha | Logistic C |
|---|---|---|
| 1 | 1.0 | 0.01 |
| 2 | 0.01 | 0.1 |
| 3 | 10.0 | 0.01 |
| 4 | 0.1 | 0.01 |
| 5 | 10.0 | 0.01 |

## Regression

All final metrics are recomputed from persisted test predictions. Returns are
fractions; MAE/MSE retain their corresponding units. Spearman is primary, R2 secondary.

| Model | Spearman | Pearson | MAE | MSE | R2 |
|---|---|---|---|---|---|
| OLS | 0.0507581467 | 0.078828029 | 4.06821127e-05 | 6.06164602e-09 | 0.00301915228 |
| Ridge | 0.0507468888 | 0.0788200571 | 4.06811844e-05 | 6.06163647e-09 | 0.00302072456 |
| Static | 0.0314524367 | 0.0602793702 | 4.04344879e-05 | 6.07402408e-09 | 0.000983289598 |
| Flow | 0.0337290484 | 0.0549362978 | 3.81210594e-05 | 6.07261044e-09 | 0.00121579517 |
| ZeroReturn | nan | nan | 3.60698483e-05 | 6.08144794e-09 | -0.000237740458 |
| TrainingMean | 0.00494090214 | 0.00276039451 | 3.6489346e-05 | 6.08336235e-09 | -0.000552610251 |

ZeroReturn correlation is undefined (nan), not zero. TrainingMean varies across
folds, giving a small aggregate correlation despite constant predictions inside
each fold. Ridge's positive R2 is small; its MAE is worse than zero return in this
zero-heavy target distribution. Positive rank association alone does not establish
a useful magnitude forecast.

The independent audit caught a floating-point centering residual in the inherited
Pearson helper when a fold's TrainingMean predictions were constant. A Phase 4
wrapper now explicitly reports NaN for constant inputs. Only within-fold baseline
Pearson cells and their duplicate time-of-day table changed; all predictions,
fits, aggregate metrics, inference and upstream files are byte-identical. The
original executed module and exact correction are preserved in
reports/phase4/metric_correction_audit.json. This is a numerical reporting fix,
not a revised hypothesis, model choice or upstream artifact repair.

| Model | 60s block CI lower | 60s block CI upper | Circular-shift p |
|---|---|---|---|
| OLS | 0.0301474912 | 0.071686318 | 0.000999000999 |
| Ridge | 0.030154073 | 0.0716762572 | 0.000999000999 |

1,000 paired circular moving-block draws, 60 seconds, seed 404060; 1,000 target
shifts with offsets 600..11100 seconds, seed 404600. The full 11,700-second clock
keeps all five missing test-end cells. Exact ranks are recalculated per draw.
Intervals condition on fitted models and do not include refitting uncertainty.
Circular seams and intraday nonstationarity limit coverage/null calibration;
no iid p-values or fabricated independent daily samples are used. These diagnostics
are not multiple-testing-adjusted discovery claims across variants.

## Single-feature comparison

Same 11,695 test timestamps for all three raw single-feature rank scores and Ridge.

| Score | Spearman | Pearson |
|---|---|---|
| OBI_L1 | 0.0360057941 | 0.064460969 |
| MICROPRICE_MINUS_MID | 0.0240654599 | 0.0578479263 |
| OFI_1S | 0.0244524995 | 0.0521494148 |
| Ridge | 0.0507468888 | 0.0788200571 |

Raw feature values are rank scores, not return predictions. Their mechanical
error columns in the general regression CSV have incomparable units and are
not used as forecast-quality evidence; only their correlations are compared here.

| Comparison | Delta rho | Paired CI lower | Paired CI upper |
|---|---|---|---|
| OBI_L1 | 0.0147410947 | 0.00156423706 | 0.0279374217 |
| MICROPRICE_MINUS_MID | 0.0266814289 | 0.0143308732 | 0.0389123728 |
| OFI_1S | 0.0262943893 | 0.000901376229 | 0.0510288692 |
| BEST_SINGLE | 0.0147410947 | -0.00174819762 | 0.0231059065 |

Ridge exceeds OBI_L1 by 0.0147410947, with positive paired
interval against that fixed benchmark. OBI_L1 is also the observed best of the
three singles. However, the preregistered conservative comparison recalculates
the maximum single-feature correlation in every paired bootstrap draw; that
interval crosses zero. Therefore the verdict remains **NO RELIABLE INCREMENTAL VALUE OVER SINGLE FEATURES**. It does
not erase the narrower positive OBI comparison or prove exact equivalence.

## Static vs Flow and ablation

Static uses imbalance/microprice/spread; flow uses both OFI windows/activity/timing.
Static rho=0.0314524367; flow=0.0337290484; full=0.0507468888.
Full-minus-static paired delta=0.0192944521, CI
[0.00694603546, 0.0316678124]. This supports
incremental flow information within this session, although flow alone reverses
sign in the first test block. It does not establish cross-day stability.

| Predeclared Ridge variant | Spearman |
|---|---|
| Ridge | 0.0507468888 |
| WithoutContext | 0.0497575716 |
| WithoutImbalance | 0.0470908792 |
| WithoutMicroprice | 0.0527002958 |
| WithoutOFI | 0.0301552122 |

Removing OFI weakens the rank score; removing microprice slightly improves its
point estimate. These fixed diagnostics do not replace the full primary model,
and their small differences are not newly selected discoveries.

## Classification

| Model | Macro F1 | Balanced accuracy | Accuracy | Log loss |
|---|---|---|---|---|
| Logistic | 0.302183044 | 0.352303848 | 0.624711415 | 0.893062539 |
| AlwaysFLAT | 0.255601835 | 0.333333333 | 0.62180419 | 13.6315587 |
| TrainingPrior | 0.255601835 | 0.333333333 | 0.62180419 | 0.938054339 |

Logistic-minus-Always-FLAT Macro F1=0.0465812091, paired 60s-block 95% CI
[0.0373029407, 0.0551189929]. All five folds improve Macro F1.
The increase is below the predeclared .05 CLEAR threshold, hence **MODEST DIRECTIONAL IMPROVEMENT**.
Accuracy increases only about 0.29 percentage points. UP and DOWN recall remain
extremely low; the classifier mostly predicts FLAT. No class weights or alternate
classification thresholds were tried. Always-FLAT's one-hot log loss uses float64
epsilon clipping/renormalization; TrainingPrior is the meaningful probabilistic
baseline (same argmax classes, substantially different loss).

| Class | Logistic precision | Recall | F1 |
|---|---|---|---|
| DOWN | 0.372 | 0.0432357043 | 0.0774677218 |
| FLAT | 0.633572886 | 0.981985699 | 0.770209783 |
| UP | 0.413793103 | 0.0316901408 | 0.0588716271 |

Confusion matrices for every fold/model and concatenated tests are saved in
confusion_matrix.csv. Calibration uses fixed width .1 probability bins for UP
and DOWN, retaining empty bins. Central .1–.3 bins are reasonably close on this
day; the lowest bin understates observed direction frequency, and high-probability
tails are sparse/noisy. Probabilities are not uniformly well calibrated, and no
calibrator was fitted.

| Direction probability | Top-decile cutoff | N | Mean forecast | P(DOWN) | P(FLAT) | P(UP) |
|---|---|---|---|---|---|---|
| DOWN | 0.281759594 | 1170 | 0.335166799 | 0.318803419 | 0.390598291 | 0.290598291 |
| UP | 0.274333334 | 1170 | 0.322008273 | 0.297435897 | 0.356410256 | 0.346153846 |

Top-decile thresholds are EX-POST DESCRIPTIVE empirical 90th percentiles, inclusive
of ties. Both tails include many opposite moves and FLATs; they are not trade entries.

## Coefficients

| Feature | Ridge positive folds | Negative folds | Min standardized weight | Max standardized weight |
|---|---|---|---|---|
| OBI_L1 | 5 | 0 | 3.95722037e-06 | 4.41404159e-06 |
| MICROPRICE_MINUS_MID | 5 | 0 | 2.13442523e-06 | 4.79524856e-06 |
| OFI_100MS | 0 | 5 | -3.57871088e-06 | -2.78971963e-06 |
| OFI_1S | 5 | 0 | 6.3105926e-06 | 8.22932368e-06 |
| RELATIVE_SPREAD | 0 | 5 | -3.81196939e-06 | -2.05821668e-06 |
| EVENT_RATE_1S | 1 | 4 | -3.19487133e-07 | 1.38208047e-07 |
| TIME_SINCE_PREVIOUS_TRADE | 5 | 0 | 9.95477886e-07 | 1.34395159e-06 |

OLS has the same directional signs as Ridge: OBI/microprice/OFI_1S positive and
OFI_100MS negative in all five folds. OFI_1S is the largest standardized return
coefficient in every fold. Microprice's magnitude roughly halves as training
expands; spread and OFI_1S magnitudes also shrink. Activity flips sign once and
has small return weights. Coefficient sign consistency does not prevent the
negative first-fold OOS correlation.

Logistic UP-vs-DOWN log-odds coefficient contrasts are positive for OBI_L1 and
OFI_1S, negative for OFI_100MS and microprice in all folds. Thus microprice does
not retain a positive conditional directional coefficient in this redundant
classification specification, despite positive marginal Phase 3 association
and positive regression weights. Context strongly separates FLAT from movement;
do not compare logistic weights directly with return coefficients. Full per-class
weights, intercepts, fold scales and sign summaries are saved.

## Intraday stability and embargo

| Fold | OBI rho | OLS rho | Ridge rho | Static rho | Flow rho | Logistic Macro F1 | Always-FLAT Macro F1 |
|---|---|---|---|---|---|---|---|
| 1 | 0.00440393783 | -0.0105837187 | -0.0105930686 | 0.00297489124 | -0.059173249 | 0.296845299 | 0.285784074 |
| 2 | 0.0320177477 | 0.0442920683 | 0.0442920683 | 0.0240171422 | 0.0238418025 | 0.29368037 | 0.252501107 |
| 3 | 0.0231063525 | 0.0443109126 | 0.044270427 | 0.016597828 | 0.0386298295 | 0.296433583 | 0.256855015 |
| 4 | 0.0761411663 | 0.102703812 | 0.102703965 | 0.0802636742 | 0.0758677215 | 0.302257363 | 0.247152722 |
| 5 | 0.0406546214 | 0.0701053542 | 0.0700729778 | 0.0317064403 | 0.0719422955 | 0.297518913 | 0.231827477 |

Ridge is positive in 4/5 test folds and negative in fold 1. Its strongest
test block is fold 4, not evidence that it works only at the close. OBI_L1 retains
positive point estimates in all five. Flow alone is especially unstable early.
These overlapping expanding training histories do not create independent models
or days. The 10-second embargo yields Ridge rho
0.0508026229
and Logistic Macro F1
0.303264477;
the substantive conclusions do not change.

## Controls and independent checks

Random-label Ridge: training labels circularly rotated by 600 seconds only;
rho=-0.0120348897, CI [-0.0367976259, 0.0135126864],
shift p=0.345654346. It is near noise and materially below the aligned
model. No investigation trigger fired, no offsets were retried.
Random-feature Ridge reuses seed 161803; delta rho versus full=
-0.000245838224, paired CI
[-0.00152046027, 0.00107013924]. The random
weight is positive in all folds but only approximately 0.00000029–0.00000094,
well below dominant signal coefficients, and does not improve OOS association.
It is not persistently dominant. No random feature redraw occurred.

Future mutation changed **64,191 later Phase 2 rows across
217 payload columns** in memory, preserving only
authoritative timestamp/source-order/event-index keys. Earlier grid, model
parameters, validation selection, predictions and metrics were unchanged. TEST
return and class mutations leave fits and predictions fixed while both task
metrics change. Raw and grid storage-order shuffles reproduce results after
canonical chronological restoration. **96 real-data isolation
checks PASS**. Two complete deterministic experiment runs agree, with
**300 comparisons PASS** at absolute tolerance 1e-12; model artifacts
match exactly. Thread limit one, fixed seeds, pinned Phase 4 requirements.

Full `pytest -q`: **114 passed, 0 failures,
0 errors, 0 skipped**; original 101 cases unchanged.
The independent verifier reloads saved predictions, uses SciPy/sklearn arithmetic,
reconstructs fitted predictions from coefficients/scales, audits fold membership,
preprocessing and validation scores, checks literal block samples, shift draws,
all metrics/confusion and these visible report tables. Its hash-recorded result
is reports/phase4/independent_verification.json, not an external replication.

## Hypothesis outcomes

H1: PARTIAL. Positive paired improvement over fixed OBI_L1, but conservative
best-single envelope includes zero. H2: SUPPORTED WITHIN DAY, full-minus-static
paired interval positive; flow alone varies materially. H3: SUPPORTED WITHIN DAY,
Macro F1 improves modestly with very low directional recall. H4: SUPPORTED
DESCRIPTIVELY by negative first-fold versus stronger later-fold correlations;
no independent-day significance is claimed.

## OFI and horizon interpretation

Phase 3 found **negative OFI_100MS and positive OFI_1S** association; the semantic
audit and nine hand sign/window tests found no coding defect. Regression and
logistic directional contrasts preserve those conditional signs here. Different
aggregation horizons, very short-lived quote/order reactions, sign convention
interacting with dynamics, stale source windows and one-day noise are plausible
possibilities, not established causal explanations. No sign flip was imposed.

Phase 3 association did **not** simply decay with horizon. Larger 5s/10s static
associations remain unresolved research observations. Primary modeling stays
at the predeclared 1s target; no horizon was switched and no secondary-horizon
models were added.

## Limitations

One stock, one 2003 day; no cross-day or cross-stock replication. OOS is only
the latter half of that same session; training/validation histories overlap.
More importantly, feature membership was informed by Phase 3 full-day research,
including the eventual Phase 4 test period: this is genuine chronological OOS
for fitted parameters/tuning, **not an untouched confirmatory dataset for the
overall research process**. Bootstrap intervals do not remove that prior exposure
or model-selection uncertainty. Fresh days are needed for confirmation.

Visible venue book only; hidden P executions are prints, not observable queues.
No consolidated market, transport-gap proof, independent book snapshot, verified
historical tick rule, certified halt/auction states, receive latency, fees or
execution model. Features carried from source states can be stale; some short
windows do not reach the current grid clock. No execution simulation or PnL
was computed. No evidence here establishes tradable alpha, capacity, modern-market
validity or profitability. Local row-level predictions are gitignored under
the inherited data-redistribution policy.

## Figures and reproduction

Seven PNG/SVG figures in reports/phase4/figures: prediction/realized binned density,
fold correlations, standardized coefficients, logistic confusion, UP/DOWN
calibration, static/flow/full, and intraday regression/classification metrics.
Every figure identifies MSFT, 2003-01-03, one historical trading day.

Install requirements.phase4.txt in the existing environment. `scripts/run_phase4.py`
refuses to overwrite completed predictions; it checks frozen hashes, fits twice,
performs real mutation audits, persists OOS rows, then evaluates. To inspect an
existing run without fitting, execute `scripts/verify_phase4.py`. Report and
figures can be rendered from saved outputs with build_phase4_report.py and
plot_phase4.py. No Phase 5 work has begun. **STOP after Phase 4.**
