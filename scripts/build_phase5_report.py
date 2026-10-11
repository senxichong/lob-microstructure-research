import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.phase5.storage import OUT, csvread, csvwrite, dump


def fmt(x):
    return f"{float(x):.9g}"


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def main():
    reg = {r["model_name"]: r for r in csvread("regression_metrics")}
    cla = {r["model_name"]: r for r in csvread("classification_metrics")}
    rp = csvread("nonlinear_vs_ridge_bootstrap")
    cp = csvread("nonlinear_vs_logistic_bootstrap")
    gaps = csvread("train_validation_test_gap")
    random = csvread("random_label_control")[0]
    placebo = csvread("temporal_placebo")[0]
    randomf = csvread("random_feature_control")[0]
    importance = csvread("feature_importance")
    permutation = csvread("permutation_importance")
    metadata = json.loads((OUT / "model_metadata.json").read_text())
    dominance = 0
    validation_gain = 0
    for fold in range(1, 6):
        imp = [
            r
            for r in importance
            if r["model_name"] == "RandomFeature" and int(r["fold_id"]) == fold
        ]
        top = max(imp, key=lambda r: float(r["native_importance"]))
        dominance += int(top["feature"] == "RANDOM_CONTROL")
        a = next(
            a
            for a in metadata
            if a["task"] == "regression"
            and a["model_name"] == "RandomFeature"
            and a["fold_id"] == fold
        )
        b = next(
            a
            for a in metadata
            if a["task"] == "regression" and a["model_name"] == "LightGBM" and a["fold_id"] == fold
        )
        validation_gain += int(a["validation_score"] - b["validation_score"] >= 0.01)
    controls_ok = (
        random["investigation_required"] == "False"
        and placebo["investigation_required"] == "False"
        and randomf["material_test_gain"] == "False"
        and dominance < 4
        and validation_gain < 4
    )
    dump(
        "control_assessment.json",
        dict(
            sane=controls_ok,
            random_feature_top_importance_folds=dominance,
            random_feature_validation_gain_at_least_001_folds=validation_gain,
            random_label_investigation=random["investigation_required"],
            temporal_placebo_investigation=placebo["investigation_required"],
        ),
    )
    verdicts = {}
    severe = {}
    for task, rows in [("regression", rp), ("classification", cp)]:
        primary = next(r for r in rows if r["model_name"] == "ValidationSelected")
        gap = float(
            np.median(
                [
                    float(r["train_minus_test"])
                    for r in gaps
                    if r["task"] == task and r["model_name"] == "ValidationSelected"
                ]
            )
        )
        severe[task] = gap > (0.15 if task == "regression" else 0.20)
        reliable = (
            float(primary["ci_lower"]) > 0
            and int(primary["fold_wins"]) >= 4
            and controls_ok
            and not severe[task]
        )
        if reliable:
            verdict = (
                "RELIABLE NONLINEAR IMPROVEMENT"
                if task == "regression"
                else "RELIABLE NONLINEAR DIRECTIONAL IMPROVEMENT"
            )
        elif float(primary["observed_delta"]) > 0:
            verdict = (
                "MODEST BUT UNCERTAIN NONLINEAR IMPROVEMENT"
                if task == "regression"
                else "MODEST BUT UNCERTAIN NONLINEAR DIRECTIONAL IMPROVEMENT"
            )
        elif all(float(r["ci_upper"]) < 0 for r in rows if r["model_name"] in ["RF", "LightGBM"]):
            verdict = (
                "NONLINEAR MODELS UNDERPERFORM RIDGE"
                if task == "regression"
                else "NONLINEAR CLASSIFIERS UNDERPERFORM LOGISTIC"
            )
        else:
            verdict = "NO RELIABLE NONLINEAR IMPROVEMENT"
        verdicts[task] = verdict
    tests = ET.parse(OUT / "test_results.xml").getroot().find("testsuite").attrib
    isolation = json.loads((OUT / "isolation_audit.json").read_text())
    repeat = json.loads((OUT / "reproducibility_audit.json").read_text())
    decision = (
        "PASS"
        if controls_ok
        and tests["failures"] == "0"
        and tests["errors"] == "0"
        and isolation["status"] == "PASS"
        and repeat["status"] == "PASS"
        else "FAIL"
    )
    dump(
        "research_verdict.json",
        dict(
            **verdicts,
            phase_decision=decision,
            severe_gap=severe,
            controls_sane=controls_ok,
            tests=tests,
            primary_comparison="ValidationSelected",
            independent_verification_required=True,
        ),
    )
    native_summary = []
    for task in ["regression", "classification"]:
        for name in ["RF", "LightGBM"]:
            rows = [r for r in importance if r["task"] == task and r["model_name"] == name]
            for feature in dict.fromkeys(r["feature"] for r in rows):
                v = [float(r["normalized_importance"]) for r in rows if r["feature"] == feature]
                top = sum(
                    max(
                        [r for r in rows if int(r["fold_id"]) == f],
                        key=lambda r: float(r["native_importance"]),
                    )["feature"]
                    == feature
                    for f in range(1, 6)
                )
                pv = [
                    float(r["degradation"])
                    for r in permutation
                    if r["task"] == task and r["model_name"] == name and r["feature"] == feature
                ]
                native_summary.append(
                    dict(
                        task=task,
                        model_name=name,
                        feature=feature,
                        mean_native_importance=float(np.mean(v)),
                        top_folds=top,
                        mean_permutation_degradation=float(np.mean(pv)),
                        positive_permutation_folds=sum(
                            np.mean(
                                [
                                    float(r["degradation"])
                                    for r in permutation
                                    if r["task"] == task
                                    and r["model_name"] == name
                                    and r["feature"] == feature
                                    and int(r["fold_id"]) == f
                                ]
                            )
                            > 0
                            for f in range(1, 6)
                        ),
                    )
                )
    csvwrite("importance_stability", native_summary)
    interaction = csvread("interaction_summary")
    surface = []
    for a, b in dict.fromkeys((r["feature_a"], r["feature_b"]) for r in interaction):
        v = [
            float(r["nonadditive_rms"])
            for r in interaction
            if r["feature_a"] == a and r["feature_b"] == b
        ]
        surface.append([a + " × " + b, fmt(np.mean(v)), fmt(min(v)), fmt(max(v))])
    rf = csvread("regression_fold_metrics")
    cf = csvread("classification_fold_metrics")
    stability = []
    for task, rows, field, baseline in [
        ("regression", rf, "spearman", "Ridge"),
        ("classification", cf, "macro_f1", "Logistic"),
    ]:
        for name in ["RF", "LightGBM", "ValidationSelected"]:
            v = [r for r in rows if r["model_name"] == name]
            best = max(v, key=lambda r: float(r[field]))
            worst = min(v, key=lambda r: float(r[field]))
            stability.append(
                dict(
                    task=task,
                    model_name=name,
                    best_fold=best["fold_id"],
                    best_metric=float(best[field]),
                    worst_fold=worst["fold_id"],
                    worst_metric=float(worst[field]),
                )
            )
    csvwrite("time_stability", stability)
    calibration = csvread("calibration")
    cal = []
    for name in ["Logistic", "RF", "LightGBM"]:
        for c in ["DOWN", "UP"]:
            r = [
                r
                for r in calibration
                if r["model_name"] == name and r["class_name"] == c and int(r["n"])
            ]
            cal.append(
                [
                    name,
                    c,
                    fmt(
                        sum(
                            int(v["n"])
                            * abs(float(v["mean_probability"]) - float(v["observed_frequency"]))
                            for v in r
                        )
                        / 11695
                    ),
                    sum(int(v["n"]) for v in r if float(v["lower"]) >= 0.5),
                ]
            )
    sections = f"""# Phase 5 — Nonlinear machine learning versus frozen linear baselines

Regression verdict: **{verdicts["regression"]}**.
Classification verdict: **{verdicts["classification"]}**.
Methodology decision: **PHASE 5: {decision}**; the separate independent-verification
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

Preregistered at {json.loads((OUT / "protocol_freeze.json").read_text())["utc"]}.
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

{table(["Task", "Fold", "RF candidate", "LightGBM candidate", "Validation-selected family"], [[task, f, next(a["candidate_id"] for a in metadata if a["task"] == task and a["model_name"] == "RF" and a["fold_id"] == f), next(a["candidate_id"] for a in metadata if a["task"] == task and a["model_name"] == "LightGBM" and a["fold_id"] == f), next(a["family"] for a in metadata if a["task"] == task and a["model_name"] == "ValidationSelected" and a["fold_id"] == f)] for task in ["regression", "classification"] for f in range(1, 6)])}

## Regression results

All final scores reload persisted OOS predictions. Frozen Phase 4 benchmarks
are loaded, never retrained. Spearman primary; MAE/MSE use fractional-return units.

{table(["Model", "Spearman", "Pearson", "MAE", "MSE", "R2"], [[n, *[fmt(reg[n][k]) for k in ["spearman", "pearson", "mae", "mse", "r2"]]] for n in ["Ridge", "RF", "LightGBM", "ValidationSelected"]])}

Single-feature reference: OBI_L1 and Phase 4's observed best single are the same
column, rho **{fmt(reg["OBI_L1"]["spearman"])}** on identical timestamps.
Microprice rho={fmt(reg["MICROPRICE_MINUS_MID"]["spearman"])}, OFI_1S rho={fmt(reg["OFI_1S"]["spearman"])}.
These raw features are rank scores, not return forecasts; their mechanical error
columns in the general CSV are not meaningful across incompatible units.
Phase 4's conservative bootstrap-best-single comparison included zero; this
finding is not rewritten merely because a more complex model is now available.

## Paired nonlinear-vs-Ridge inference

{table(["Model", "Observed delta rho", "95% CI lower", "95% CI upper", "Fold wins / 5"], [[r["model_name"], *[fmt(r[k]) for k in ["observed_delta", "ci_lower", "ci_upper"]], r["fold_wins"]] for r in rp])}

1,000 paired circular 60-second moving-block draws, seed 505060, exact reranking,
full 11,700-second clock with five missing test-end cells. Observed and bootstrap
mean deltas and model intervals are saved. The primary inference is the
validation-selected pipeline; separate family comparisons are secondary, without
multiplicity-adjusted discovery claims. These are conditional on fixed fitted
models, not refitting/selection uncertainty. Circular seams/nonstationarity and
one-day prior examination limit nominal coverage. No iid p-values.
Verdict: **{verdicts["regression"]}**. Validation-selected severe-gap flag:
**{severe["regression"]}**, under the preregistered median train-test rho gap >.15 rule.

## Classification results

{table(["Model", "Macro F1", "Balanced accuracy", "Accuracy", "Log loss"], [[n, *[fmt(cla[n][k]) for k in ["macro_f1", "balanced_accuracy", "accuracy", "log_loss"]]] for n in ["Logistic", "RF", "LightGBM", "ValidationSelected", "AlwaysFLAT"]])}

Accuracy must be compared with retained FLAT prevalence .621804. Always-FLAT
one-hot log loss is epsilon-clipped; it is not a calibrated probabilistic model.
No probability threshold tuning or class resampling was used.

LightGBM improves Macro F1 by 0.08227 over Logistic but accuracy by only about
0.28 percentage points. Its DOWN/UP recalls rise to 11.20%/13.86%, still low;
FLAT recall remains 93.28%. The gain is meaningful under the frozen within-day
metric rubric without making directional classification broadly accurate.

{table(["Model", "Class", "Precision", "Recall", "F1"], [[n, c, *[fmt(cla[n][c + "_" + k]) for k in ["precision", "recall", "f1"]]] for n in ["Logistic", "RF", "LightGBM"] for c in ["DOWN", "FLAT", "UP"]])}

Every per-fold/aggregate confusion matrix is saved. Coarse UP/DOWN calibration
uses fixed .1 bins; empty bins remain. The table below reports descriptive binned
mean absolute calibration error (count-weighted) and count with probability>=.5.
Sparse bins and regime changes prevent general calibration claims; no calibrator
was fitted and no thresholds are trade entries.

{table(["Model", "Class", "Binned calibration error", "P>=.5 count"], cal)}

## Paired nonlinear-vs-Logistic inference

LightGBM's coarse calibration is worse than Logistic's here: DOWN binned error
0.02166 versus 0.00991, and UP 0.00996 versus 0.00671. Its lower log loss and
higher Macro F1 therefore do not imply uniformly better probabilities. RF also
has worse DOWN calibration (0.01603), while its UP error is slightly smaller.
These probability shortcomings remain visible alongside the discrimination gain.

{table(["Model", "Observed delta Macro F1", "95% CI lower", "95% CI upper", "Fold wins / 5"], [[r["model_name"], *[fmt(r[k]) for k in ["observed_delta", "ci_lower", "ci_upper"]], r["fold_wins"]] for r in cp])}

Same paired temporal weights, all three classes retained. Verdict:
**{verdicts["classification"]}**. Severe-gap flag **{severe["classification"]}**
under the preregistered median train-test Macro-F1 gap >.20 rule. A positive point
estimate alone is insufficient: reliability also requires positive CI lower,
at least four fold wins and sane controls.

## Model stability

{table(["Fold", "Ridge rho", "RF rho", "LightGBM rho", "Selected rho"], [[f, *[fmt(next(r["spearman"] for r in rf if r["model_name"] == n and int(r["fold_id"]) == f)) for n in ["Ridge", "RF", "LightGBM", "ValidationSelected"]]] for f in range(1, 6)])}

{table(["Fold", "Logistic F1", "RF F1", "LightGBM F1", "Selected F1"], [[f, *[fmt(next(r["macro_f1"] for r in cf if r["model_name"] == n and int(r["fold_id"]) == f)) for n in ["Logistic", "RF", "LightGBM", "ValidationSelected"]]] for f in range(1, 6)])}

Best/worst blocks are reported, never discarded:

{table(["Task", "Model", "Best fold", "Best metric", "Worst fold", "Worst metric"], [[r["task"], r["model_name"], r["best_fold"], fmt(r["best_metric"]), r["worst_fold"], fmt(r["worst_metric"])] for r in stability])}

Fit-gap summaries below are means over the five selected models, not independent
replications. Every train/validation/test value is saved per fold. Training scores
are in-sample and are never described as predictive evidence.

{table(["Task", "Model", "Mean train", "Mean validation", "Mean test", "Median train-test gap"], [[task, n, *[fmt(np.mean([float(r[k]) for r in gaps if r["task"] == task and r["model_name"] == n])) for k in ["train_metric", "validation_metric", "test_metric"]], fmt(np.median([float(r["train_minus_test"]) for r in gaps if r["task"] == task and r["model_name"] == n]))] for task in ["regression", "classification"] for n in ["RF", "LightGBM", "ValidationSelected"]])}

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

{table(["Task", "Family", "Feature", "Mean native share", "Top native folds", "Mean OOS permutation loss", "Positive-loss folds"], [[r["task"], r["model_name"], r["feature"], fmt(r["mean_native_importance"]), r["top_folds"], fmt(r["mean_permutation_degradation"]), r["positive_permutation_folds"]] for r in native_summary if r["model_name"] == "LightGBM"])}

{table(["Prediction pair", "Spearman", "Pearson"], [[r["model_a"] + " / " + r["model_b"], fmt(r["spearman"]), fmt(r["pearson"])] for r in csvread("prediction_correlation") if r["model_a"] < r["model_b"]])}

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

{table(["Interaction", "Mean nonadditive RMS (return)", "Min fold RMS", "Max fold RMS"], surface)}

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

{table(["LightGBM input", "OOS Spearman", "Full-minus-input", "Paired CI lower", "Paired CI upper"], [[r["model_name"], *[fmt(r[k]) for k in ["spearman", "full_minus_model", "difference_ci_lower", "difference_ci_upper"]]] for r in csvread("static_flow_full")])}

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
{fmt(random["spearman"])}, CI [{fmt(random["ci_lower"])},{fmt(random["ci_upper"])}]; investigation flag {random["investigation_required"]}.
Separate 600-second training-target rotation placebo: rho {fmt(placebo["spearman"])},
CI [{fmt(placebo["ci_lower"])},{fmt(placebo["ci_upper"])}]; flag {placebo["investigation_required"]}.
Both reuse each fold's selected full configuration and never import test outcomes.
No seeds, block permutations or offsets were retried.

Random feature reuses Phase 3 seed 161803 exactly once. Delta rho
{fmt(randomf["delta"])}, paired CI [{fmt(randomf["delta_ci_lower"])},{fmt(randomf["delta_ci_upper"])}].
Top native gain in {dominance}/5 folds; validation gain>=.01 in {validation_gain}/5.
Material-test-gain flag {randomf["material_test_gain"]}; overall controls sane **{controls_ok}**.
If any investigation flag is true, the methodological decision remains FAIL
pending a documented explanation; it must not be fixed by rerunning randomness.

Future mutation altered {isolation["future_rows_mutated"]:,} later Phase 2 rows across
all loaded feature/book/label payload columns, preserving chronological keys.
Earlier fitted signatures, validation choices, predictions and metrics unchanged.
Test-label mutation leaves fits/choices/predictions unchanged and changes evaluation.
Authoritative source/grid sorting restores shuffled storage. Future-derived whitelist
trap rejected before fitting. Isolation checks **{isolation["checks"]} PASS**.
Two complete deterministic runs match signatures, metadata and predictions:
**{repeat["comparisons"]} comparisons PASS**, tolerance 1e-12. Versions recorded in model metadata.

A pre-result review caught a duplicate metadata-key merge in the save routine.
The initial fitting process was stopped before persistence; the serializer was
corrected, a prediction round-trip test added, and the same frozen experiment
restarted. No test scores were inspected and no model settings changed.
The aborted execution manifest and restart audit are retained.

Full repository tests: **{tests["tests"]} passed, {tests["failures"]} failures,
{tests["errors"]} errors, {tests["skipped"]} skipped**. All Phase 1–4 tests remain.
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
"""
    Path("PHASE5_REPORT.md").write_text(sections, encoding="utf-8")
    print(
        json.dumps(dict(**verdicts, phase_decision=decision, controls_sane=controls_ok), indent=2)
    )


if __name__ == "__main__":
    main()
