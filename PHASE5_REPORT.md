# Phase 5 — Nonlinear machine learning versus frozen linear baselines

Regression verdict: **RELIABLE NONLINEAR IMPROVEMENT**.
Classification verdict: **RELIABLE NONLINEAR DIRECTIONAL IMPROVEMENT**.
Methodology decision: **PHASE 5: PASS**; the separate independent-verification
receipt verifies the final report hash. **MSFT, 2003-01-03, single historical trading day.**

## Upstream integrity

Phase 2 SHA-256 `a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96`
matches. Authoritative paths were discovered from PHASE2_MANIFEST.json,
PHASE3_MANIFEST.json and PHASE4_MANIFEST.json. All 67 frozen Phase 4 evidence
files initially matched; source code, protocols, matrices, reports, fold definitions
and benchmark predictions remain unchanged. RESEARCH_LOG.md is append-only.
Phase 4's documented constant-baseline Pearson correction is preserved as-is.

Expected Phase 4 facts were confirmed: 5 folds, 11,695 OOS timestamps; OBI_L1
.036006, OLS .050758, Ridge .050747 with 60s CI [.030154,.071676]; static .031452,
flow .033729; Logistic Macro F1 .302183, balanced accuracy .352304, accuracy .624711;
Always-FLAT accuracy .621804. Prior verdicts remain NO RELIABLE INCREMENTAL VALUE
OVER SINGLE FEATURES and MODEST DIRECTIONAL IMPROVEMENT.

## Research question

Can trees extract incremental prediction from the same information set?
No leaderboard search, feature mining or execution experiment. The primary
best-nonlinear pipeline is **ValidationSelected**: each fold/task selects its
family using validation only, then evaluates that frozen choice on test.
Highest test-family point estimates are descriptive and never select/refit a model.

## Data and folds

Exact Phase 4 fold_manifest.csv reused. Train blocks 1–4 through 1–8, validate
5–9, test 6–10 in ten 39-minute blocks covering [09:30,16:00) Eastern.
Test starts: 12:45, 13:24, 14:03, 14:42, 15:21; each fold has 2,339 scored rows.
Requested target timestamp g+1s must precede each partition end. Preserve
the unavailable opening state and purged test-end cells. Primary embargo zero.
**11,695 identical OOS timestamps per model**, covering 12:45–16:00 Eastern.
Earlier tests may enter later training/validation chronologically; partitions
are disjoint within each fold. No full-day fitting or train+validation refit.

## Feature set

OBI_L1; MICROPRICE_MINUS_MID; OFI_100MS; OFI_1S; RELATIVE_SPREAD;
EVENT_RATE_1S; TIME_SINCE_PREVIOUS_TRADE. Exactly the frozen Phase 4 set.
Training-only medians, no tree scaling. Whitelist rejects unregistered/future
fields; RANDOM_CONTROL only in its named diagnostic. No new inputs or interactions
were manually engineered. Source-event features carry forward to the grid with
their original staleness, while targets are clock-anchored. FLAT retained.

## Hyperparameter protocol

Preregistered at 2026-10-11T11:52:51.415588+00:00.
RF: 150 trees, four (depth, leaf minimum, features) candidates:
(3,100,sqrt), (6,100,1.0), (6,20,sqrt), (None,100,1.0).
LightGBM: four (leaves,depth,rate,rounds,leaf minimum) candidates:
(7,3,.03,100,200), (7,3,.08,200,100), (15,4,.03,200,100), (15,4,.08,100,200).
CPU/single thread, deterministic LightGBM, fixed seed 505, full column/row
fractions, max_bin=63. RF bootstrap is standard tree bagging, not class balancing.
Natural class proportions, no class weighting/SMOTE. Validation Spearman or
external Macro F1 selects the first best candidate (ties 1e-12); RF wins family
ties. No early stopping or test eval_set. No expanded features, alternative
horizons or balanced-class variants. Selected configurations and every validation
candidate score are in model_selection.csv; fitted models/metadata are local
under artifacts/phase5, with hashes and software versions recorded.

| Task | Fold | RF candidate | LightGBM candidate | Validation-selected family |
|---|---|---|---|---|
| regression | 1 | 3 | 3 | LightGBM |
| regression | 2 | 3 | 3 | LightGBM |
| regression | 3 | 3 | 3 | RF |
| regression | 4 | 3 | 2 | LightGBM |
| regression | 5 | 3 | 3 | LightGBM |
| classification | 1 | 3 | 3 | LightGBM |
| classification | 2 | 1 | 2 | LightGBM |
| classification | 3 | 1 | 1 | LightGBM |
| classification | 4 | 1 | 3 | LightGBM |
| classification | 5 | 3 | 3 | LightGBM |

## Regression results

All final scores reload persisted OOS predictions. Frozen Phase 4 benchmarks
are loaded, never retrained. Spearman primary; MAE/MSE use fractional-return units.

| Model | Spearman | Pearson | MAE | MSE | R2 |
|---|---|---|---|---|---|
| Ridge | 0.0507468888 | 0.0788200571 | 4.06811844e-05 | 6.06163647e-09 | 0.00302072456 |
| RF | 0.0975399682 | 0.115047396 | 4.14518725e-05 | 6.02814785e-09 | 0.00852871797 |
| LightGBM | 0.0986365897 | 0.111805083 | 4.17915696e-05 | 6.0514634e-09 | 0.00469392496 |
| ValidationSelected | 0.0981127702 | 0.111580785 | 4.16694795e-05 | 6.04657123e-09 | 0.00549855858 |

Single-feature reference: OBI_L1 and Phase 4's observed best single are the same
column, rho **0.0360057941** on identical timestamps.
Microprice rho=0.0240654599, OFI_1S rho=0.0244524995.
These raw features are rank scores, not return forecasts; their mechanical error
columns in the general CSV are not meaningful across incompatible units.
Phase 4's conservative bootstrap-best-single comparison included zero; this
finding is not rewritten merely because a more complex model is now available.

## Paired nonlinear-vs-Ridge inference

| Model | Observed delta rho | 95% CI lower | 95% CI upper | Fold wins / 5 |
|---|---|---|---|---|
| RF | 0.0467930795 | 0.029392063 | 0.065698019 | 5 |
| LightGBM | 0.0478897009 | 0.0303925125 | 0.0683177172 | 5 |
| ValidationSelected | 0.0473658814 | 0.0303078666 | 0.0671645413 | 5 |

1,000 paired circular 60-second moving-block draws, seed 505060, exact reranking,
full 11,700-second clock with five missing test-end cells. Observed and bootstrap
mean deltas and model intervals are saved. The primary inference is the
validation-selected pipeline; separate family comparisons are secondary, without
multiplicity-adjusted discovery claims. These are conditional on fixed fitted
models, not refitting/selection uncertainty. Circular seams/nonstationarity and
one-day prior examination limit nominal coverage. No iid p-values.
Verdict: **RELIABLE NONLINEAR IMPROVEMENT**. Validation-selected severe-gap flag:
**False**, under the preregistered median train-test rho gap >.15 rule.

## Classification results

| Model | Macro F1 | Balanced accuracy | Accuracy | Log loss |
|---|---|---|---|---|
| Logistic | 0.302183044 | 0.352303848 | 0.624711415 | 0.893062539 |
| RF | 0.34504328 | 0.372620756 | 0.626763574 | 0.875686919 |
| LightGBM | 0.384455023 | 0.394480351 | 0.627533134 | 0.873719534 |
| ValidationSelected | 0.384455023 | 0.394480351 | 0.627533134 | 0.873719534 |
| AlwaysFLAT | 0.255601835 | 0.333333333 | 0.62180419 | 13.6315587 |

Accuracy must be compared with retained FLAT prevalence .621804. Always-FLAT
one-hot log loss is epsilon-clipped; it is not a calibrated probabilistic model.
No probability threshold tuning or class resampling was used.

LightGBM improves Macro F1 by 0.08227 over Logistic but accuracy by only about
0.28 percentage points. Its DOWN/UP recalls rise to 11.20%/13.86%, still low;
FLAT recall remains 93.28%. The gain is meaningful under the frozen within-day
metric rubric without making directional classification broadly accurate.

| Model | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic | DOWN | 0.372 | 0.0432357043 | 0.0774677218 |
| Logistic | FLAT | 0.633572886 | 0.981985699 | 0.770209783 |
| Logistic | UP | 0.413793103 | 0.0316901408 | 0.0588716271 |
| RF | DOWN | 0.367245658 | 0.0688052069 | 0.115896633 |
| RF | FLAT | 0.646263543 | 0.959708471 | 0.772397764 |
| RF | UP | 0.411764706 | 0.0893485915 | 0.146835443 |
| LightGBM | DOWN | 0.381933439 | 0.112040911 | 0.17325665 |
| LightGBM | FLAT | 0.657076431 | 0.932755776 | 0.771014493 |
| LightGBM | UP | 0.425101215 | 0.138644366 | 0.209093926 |

Every per-fold/aggregate confusion matrix is saved. Coarse UP/DOWN calibration
uses fixed .1 bins; empty bins remain. The table below reports descriptive binned
mean absolute calibration error (count-weighted) and count with probability>=.5.
Sparse bins and regime changes prevent general calibration claims; no calibrator
was fitted and no thresholds are trade entries.

| Model | Class | Binned calibration error | P>=.5 count |
|---|---|---|---|
| Logistic | DOWN | 0.00990805802 | 19 |
| Logistic | UP | 0.00671322742 | 8 |
| RF | DOWN | 0.0160315175 | 0 |
| RF | UP | 0.00614006109 | 0 |
| LightGBM | DOWN | 0.021661036 | 84 |
| LightGBM | UP | 0.0099637165 | 150 |

## Paired nonlinear-vs-Logistic inference

LightGBM's coarse calibration is worse than Logistic's here: DOWN binned error
0.02166 versus 0.00991, and UP 0.00996 versus 0.00671. Its lower log loss and
higher Macro F1 therefore do not imply uniformly better probabilities. RF also
has worse DOWN calibration (0.01603), while its UP error is slightly smaller.
These probability shortcomings remain visible alongside the discrimination gain.

| Model | Observed delta Macro F1 | 95% CI lower | 95% CI upper | Fold wins / 5 |
|---|---|---|---|---|
| RF | 0.0428602362 | 0.0327315424 | 0.0531234833 | 5 |
| LightGBM | 0.0822719791 | 0.0697657215 | 0.0942498667 | 5 |
| ValidationSelected | 0.0822719791 | 0.0697657215 | 0.0942498667 | 5 |

Same paired temporal weights, all three classes retained. Verdict:
**RELIABLE NONLINEAR DIRECTIONAL IMPROVEMENT**. Severe-gap flag **False**
under the preregistered median train-test Macro-F1 gap >.20 rule. A positive point
estimate alone is insufficient: reliability also requires positive CI lower,
at least four fold wins and sane controls.

## Model stability

| Fold | Ridge rho | RF rho | LightGBM rho | Selected rho |
|---|---|---|---|---|
| 1 | -0.0105930686 | 0.0653097416 | 0.0742209812 | 0.0742209812 |
| 2 | 0.0442920683 | 0.108965548 | 0.110470838 | 0.110470838 |
| 3 | 0.044270427 | 0.0834116065 | 0.0852552507 | 0.0834116065 |
| 4 | 0.102703965 | 0.127683654 | 0.122922253 | 0.122922253 |
| 5 | 0.0700729778 | 0.0917387585 | 0.0994442455 | 0.0994442455 |

| Fold | Logistic F1 | RF F1 | LightGBM F1 | Selected F1 |
|---|---|---|---|---|
| 1 | 0.296845299 | 0.391806001 | 0.451822653 | 0.451822653 |
| 2 | 0.29368037 | 0.329023342 | 0.385690824 | 0.385690824 |
| 3 | 0.296433583 | 0.336202735 | 0.368721236 | 0.368721236 |
| 4 | 0.302257363 | 0.316588246 | 0.356354222 | 0.356354222 |
| 5 | 0.297518913 | 0.34459231 | 0.363156606 | 0.363156606 |

Best/worst blocks are reported, never discarded:

| Task | Model | Best fold | Best metric | Worst fold | Worst metric |
|---|---|---|---|---|---|
| regression | RF | 4 | 0.127683654 | 1 | 0.0653097416 |
| regression | LightGBM | 4 | 0.122922253 | 1 | 0.0742209812 |
| regression | ValidationSelected | 4 | 0.122922253 | 1 | 0.0742209812 |
| classification | RF | 1 | 0.391806001 | 4 | 0.316588246 |
| classification | LightGBM | 1 | 0.451822653 | 4 | 0.356354222 |
| classification | ValidationSelected | 1 | 0.451822653 | 4 | 0.356354222 |

Fit-gap summaries below are means over the five selected models, not independent
replications. Every train/validation/test value is saved per fold. Training scores
are in-sample and are never described as predictive evidence.

| Task | Model | Mean train | Mean validation | Mean test | Median train-test gap |
|---|---|---|---|---|---|
| regression | RF | 0.232502748 | 0.0953927373 | 0.0954218616 | 0.143680576 |
| regression | LightGBM | 0.194057191 | 0.098077124 | 0.0984627136 | 0.0893841807 |
| regression | ValidationSelected | 0.200948 | 0.0998760928 | 0.0980939848 | 0.0893841807 |
| classification | RF | 0.385392848 | 0.352321304 | 0.343642527 | 0.0485999812 |
| classification | LightGBM | 0.430390635 | 0.392600039 | 0.385149108 | 0.0572070988 |
| classification | ValidationSelected | 0.430390635 | 0.392600039 | 0.385149108 | 0.0572070988 |

Complexity_summary.csv records actual trees/rounds, RF depth distribution/leaf counts,
LightGBM leaf/depth limits and training sample sizes (9,358 to 18,718). These raw
counts overstate independent information; 150 bagged trees or up to 200 boosting
rounds can overfit weak, serially dependent signals. Any severe-gap flags above
explicitly block a reliable-improvement verdict even if a paired interval is positive.

The train-test gap is substantial even without a severe-gap flag: RF mean training
rho 0.2325 versus mean test 0.0954, and LightGBM 0.1941 versus 0.0985. The primary
selector's median gap 0.08938 is below the predeclared .15 cutoff, not proof of
absence of overfitting. RF regression selected unbounded-depth, minimum-100-leaf
models in every fold, with observed maximum depths 19–24. LightGBM selected
depth-4, 15-leaf regression configurations, usually 100 rounds (200 in fold 4).

## Interpretation

Native importance is split-based association, not causality. RF impurity importance
and LightGBM gain favor split opportunities and may redistribute credit among
correlated substitutes. Held-out permutation circularly shifts one column by
300/600/900 seconds, breaks its joint dependence, and creates a seam. It never
feeds back into model selection. Negative degradation means the perturbed model
scored better in that diagnostic; it is not a new feature-selection instruction.

| Task | Family | Feature | Mean native share | Top native folds | Mean OOS permutation loss | Positive-loss folds |
|---|---|---|---|---|---|---|
| regression | LightGBM | OBI_L1 | 0.153834046 | 0 | 0.029179111 | 5 |
| regression | LightGBM | MICROPRICE_MINUS_MID | 0.0606762216 | 0 | -0.00137001914 | 0 |
| regression | LightGBM | OFI_100MS | 0.0808245483 | 0 | 0.0203421862 | 5 |
| regression | LightGBM | OFI_1S | 0.226393927 | 0 | 0.0437855006 | 5 |
| regression | LightGBM | RELATIVE_SPREAD | 0.143377247 | 0 | 0.0157376434 | 5 |
| regression | LightGBM | EVENT_RATE_1S | 0.081980187 | 0 | 0.00404880955 | 4 |
| regression | LightGBM | TIME_SINCE_PREVIOUS_TRADE | 0.252913824 | 5 | 0.0312951057 | 5 |
| classification | LightGBM | OBI_L1 | 0.0716676056 | 0 | 0.00129170714 | 3 |
| classification | LightGBM | MICROPRICE_MINUS_MID | 0.0442449029 | 0 | 0.00350591904 | 4 |
| classification | LightGBM | OFI_100MS | 0.0848141481 | 0 | 0.0295787595 | 5 |
| classification | LightGBM | OFI_1S | 0.130710803 | 0 | 0.0273142876 | 5 |
| classification | LightGBM | RELATIVE_SPREAD | 0.287564785 | 5 | 0.0369355177 | 5 |
| classification | LightGBM | EVENT_RATE_1S | 0.237080353 | 0 | 0.0321728272 | 5 |
| classification | LightGBM | TIME_SINCE_PREVIOUS_TRADE | 0.143917403 | 0 | 0.0153111722 | 5 |

| Prediction pair | Spearman | Pearson |
|---|---|---|
| RF / Ridge | 0.661909298 | 0.603005557 |
| LightGBM / Ridge | 0.676951299 | 0.574555291 |
| LightGBM / RF | 0.866751545 | 0.897603634 |

For both regressors, time since previous trade has the largest native importance
in 5/5 folds (mean RF impurity share .2698, LightGBM gain share .2529). OFI_1S has
the largest mean held-out regression permutation loss (.04690 RF, .04379 LightGBM),
positive in all five folds; OBI and OFI_100MS also have positive loss in 5/5.
Microprice's mean regression permutation loss is negative, despite nonzero native
importance and correlation with OBI. It remains in the frozen primary set.
RF classification splits native leadership between spread (3/5) and activity (2/5);
LightGBM spread leads in 5/5. Spread, activity and both flow windows matter more
consistently for classification than marginal imbalance does. Full RF native
shares and every fold's results are in feature_importance.csv and importance_stability.csv.

Prediction differences do not themselves establish new signal: paired target
scores above determine whether changed predictions help. The top-decile absolute
prediction disagreement diagnostic uses each model's TRAIN prediction mean/SD,
then one fixed pooled 90th percentile with inclusive ties. Feature-state means,
realized returns and test-fold counts are in disagreement_analysis.csv. This is
EX-POST DESCRIPTIVE, not a deployable filter or a strategy.

The disagreement cutoff is 1.35174 standardized units, selecting 1,170 observations.
Their mean absolute realized return is 0.5485 bp versus 0.3398 bp elsewhere;
they have more negative average imbalance, larger spread/activity and shorter
time since trade. Conditional rho is -0.00413 for Ridge versus .12671 for LightGBM,
but the first test fold's selected subset remains negative for both. This
post-hoc subset cannot support an independently tested filtering rule.

The four response curves use unique training quantiles and a fixed 200-row
test-feature background. They may collapse for tied/zero-heavy OFI. Partial
dependence may create unsupported combinations, particularly redundant OBI and
microprice; shape is a property of the fitted model, not a causal effect.
Interaction RMS removes additive row/column means from the predeclared 2D surfaces:

| Interaction | Mean nonadditive RMS (return) | Min fold RMS | Max fold RMS |
|---|---|---|---|
| OBI_L1 × OFI_1S | 1.26235996e-06 | 8.00425714e-07 | 1.8963599e-06 |
| OBI_L1 × RELATIVE_SPREAD | 1.82225986e-06 | 1.14561378e-06 | 2.80605408e-06 |
| OFI_1S × EVENT_RATE_1S | 1.80202993e-06 | 9.26433861e-07 | 2.92473647e-06 |
| MICROPRICE_MINUS_MID × OBI_L1 | 6.46692692e-07 | 4.28050104e-07 | 9.14128943e-07 |

These non-additive components show how a tree responds to jointly changed inputs,
not that the interactions independently predict unseen days. No interaction
feature was added or threshold optimized.

The non-additive components are present for all four prespecified pairs, with
OBI×spread and OFI×activity larger on average than microprice×OBI. Static/flow
complementarity is therefore plausible in the fitted response surfaces, while
the OOS family comparison below supplies the separate predictive evidence.

### Static / Flow / Full nonlinear comparison

LightGBM uses the same static/flow memberships as Phase 4; each subset selects only
within the frozen four-candidate grid on validation. Full remains primary.

| LightGBM input | OOS Spearman | Full-minus-input | Paired CI lower | Paired CI upper |
|---|---|---|---|---|
| Static | 0.0402218356 | 0.0584147541 | 0.0384629252 | 0.0802170318 |
| Flow | 0.077954333 | 0.0206822567 | 0.00248550463 | 0.0364842731 |
| LightGBM | 0.0986365897 | 0 | 0 | 0 |

Compare these with Phase 4 static .031452, flow .033729, full .050747. A full-minus-static
gain can show within-architecture complementarity; only full-minus-Ridge inference
can establish that nonlinear extraction strengthens prediction beyond the linear
combination. Neither comparison licenses replacing the primary architecture after test.

Here full LightGBM exceeds both static and flow subsets with positive paired
intervals. Flow alone is .07795 versus static .04022; full reaches .09864 and
beats full Ridge. Thus nonlinear extraction strengthens the combined within-day
forecast under this design. This does not identify which interaction caused the gain.

## Controls

Random-label LightGBM (single 60-second TRAIN block permutation): rho
0.00922140899, CI [-0.0106750569,0.0301252909]; investigation flag False.
Separate 600-second training-target rotation placebo: rho -0.00909464454,
CI [-0.0273877878,0.00708090899]; flag False.
Both reuse each fold's selected full configuration and never import test outcomes.
No seeds, block permutations or offsets were retried.

Random feature reuses Phase 3 seed 161803 exactly once. Delta rho
-0.00706935423, paired CI [-0.0153087272,0.000877583476].
Top native gain in 0/5 folds; validation gain>=.01 in 0/5.
Material-test-gain flag False; overall controls sane **True**.
If any investigation flag is true, the methodological decision remains FAIL
pending a documented explanation; it must not be fixed by rerunning randomness.

Future mutation altered 64,191 later Phase 2 rows across
all loaded feature/book/label payload columns, preserving chronological keys.
Earlier fitted signatures, validation choices, predictions and metrics unchanged.
Test-label mutation leaves fits/choices/predictions unchanged and changes evaluation.
Authoritative source/grid sorting restores shuffled storage. Future-derived whitelist
trap rejected before fitting. Isolation checks **33 PASS**.
Two complete deterministic runs match signatures, metadata and predictions:
**55 comparisons PASS**, tolerance 1e-12. Versions recorded in model metadata.

A pre-result review caught a duplicate metadata-key merge in the save routine.
The initial fitting process was stopped before persistence; the serializer was
corrected, a prediction round-trip test added, and the same frozen experiment
restarted. No test scores were inspected and no model settings changed.
The aborted execution manifest and restart audit are retained.

Full repository tests: **126 passed, 0 failures,
0 errors, 0 skipped**. All Phase 1–4 tests remain.
scripts/verify_phase5.py independently reloads saved rows and models, reconstructs
predictions, train/validation scores, fold membership, class ordering, metrics,
correlations, paired deltas and literal temporal resamples; it checks visible
report tables and immutable hashes. Its receipt is independent_verification.json.

## OFI_100MS and horizon interpretation

Phase 3 and Phase 4 found negative OFI_100MS and positive OFI_1S relationships.
These upstream facts and source-side definitions remain unchanged. Tree partial
dependence and fixed interactions inspect sign/shape without imposing a flip.
Any apparent curvature, saturation or regime-specific reversal is descriptive;
zero-heavy features, correlation, source-state age and one-day noise are competing
explanations. No causal explanation is established. Phase 3 association did not
simply decay with horizon; larger 5s/10s values do not change the primary 1s target.

In the actual partial-dependence diagnostic, OFI_100MS decreases from the negative
to positive flow endpoint in all five folds. Four curves decrease at every saved
point; fold 1 has a small local upward bump near zero. The quantile grid contains
only three or four distinct points because this feature is heavily tied, so it
cannot establish detailed curvature or a general state-dependent sign reversal.
OFI_1S has a broadly positive end-to-end response with pronounced changes near
zero and nonmonotonic negative-flow segments. OBI is broadly increasing with
central plateaus and stronger positive-tail response; saturation is not established
over the examined range. Microprice response is weaker and varies by fold.

## Hypothesis outcomes

H1: SUPPORTED WITHIN THIS EXAMINED DAY. Both tree families and the validation-selected
pipeline beat Ridge in all five folds with positive paired intervals; threshold-like
responses are descriptive and do not prove the cause of the gain.
H2: SUPPORTED DESCRIPTIVELY. Fixed 2D model responses are non-additive and full
LightGBM beats both static and flow subsets; no causal interaction claim.
H3: PARTIAL. Metric magnitudes and training gaps vary substantially, but relative
tree-versus-linear wins are consistent in all five observed test blocks. The
one-day design cannot establish cross-day stability or dismiss the small-effective-N concern.
H4: SUPPORTED WITHIN DAY. Macro F1 improves more materially than raw accuracy;
directional recall remains limited and FLAT dominates classification.

## Limitations

One stock, one day in 2003; previously examined in Phases 3/4, no untouched day.
Chronological test predictions are OOS for fitted parameters, not pristine
confirmation of the overall research process. **11,695 timestamps are not
11,695 independent observations; effective independent sample size is much smaller.**
Overlapping training histories, regimes and serial dependence limit inference;
bootstrap intervals condition on fitted models and cannot remove prior exposure.
Weak signal/noise, largely monotonic effects, small effective sample size and
tree overfitting are possible reasons for limited gains, not established causes.

Visible venue book only, stale carried features, no consolidated market/hidden
queue/verified historical tick/receive latency/external book ground truth.
No cross-day/stock/modern-market generalization, execution simulation or PnL.
Predictive association is not tradable alpha. Row predictions and fitted tree
files stay local under the inherited data policy. No Phase 6 work.

## Artifacts and reproduction

Ten PNG/SVG figure pairs in reports/phase5/figures identify the historical scope.
requirements.phase5.txt pins LightGBM and inherits the Phase 4 environment.
run_phase5.py refuses to overwrite saved predictions; evaluate_phase5.py reloads
those rows and produces fixed diagnostics. verify_phase5.py audits independently.
Report and plots can be regenerated from saved outputs. **STOP after Phase 5.**
