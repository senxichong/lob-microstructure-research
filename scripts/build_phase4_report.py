"""Render frozen results; do not fit, select or redefine any experiment."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.phase4.core import FEATURES, OUT
from lob_research.phase4.storage import dump, read_csv


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    )


def f(v):
    return f"{float(v):.9g}"


def main():
    reg = {
        r["model_name"]: r for r in read_csv("regression_metrics") if r["embargo_seconds"] == "0"
    }
    cla = {
        r["model_name"]: r
        for r in read_csv("classification_metrics")
        if r["embargo_seconds"] == "0"
    }
    rf = [r for r in read_csv("regression_fold_metrics") if r["embargo_seconds"] == "0"]
    cf = [r for r in read_csv("classification_fold_metrics") if r["embargo_seconds"] == "0"]
    infer = {r["model_name"]: r for r in read_csv("regression_inference")}
    paired = {
        r["comparison"].removeprefix("Ridge minus "): r
        for r in read_csv("single_feature_comparison")
    }
    ci = json.loads((OUT / "classification_inference.json").read_text())
    positive = sum(float(r["spearman"]) > 0 for r in rf if r["model_name"] == "Ridge")
    best = paired["BEST_SINGLE"]
    verdict = "NO RELIABLE INCREMENTAL VALUE OVER SINGLE FEATURES"
    if float(best["ci_upper"]) < 0:
        verdict = "LINEAR MODEL UNDERPERFORMS SIMPLE FEATURES"
    elif float(best["ci_lower"]) > 0 and float(best["difference"]) > 0 and positive >= 4:
        verdict = (
            "CLEAR INCREMENTAL LINEAR PREDICTIVE VALUE"
            if float(best["difference"]) >= 0.02
            else "MODEST INCREMENTAL LINEAR PREDICTIVE VALUE"
        )
    gains = []
    for k in range(1, 6):
        a = next(r for r in cf if r["model_name"] == "Logistic" and int(r["fold_id"]) == k)
        b = next(r for r in cf if r["model_name"] == "AlwaysFLAT" and int(r["fold_id"]) == k)
        gains.append(float(a["macro_f1"]) - float(b["macro_f1"]))
    cv = "NO RELIABLE IMPROVEMENT"
    if ci["ci_lower"] > 0 and ci["difference"] > 0 and sum(v > 0 for v in gains) >= 4:
        cv = (
            "CLEAR DIRECTIONAL IMPROVEMENT OVER BASELINE"
            if ci["difference"] >= 0.05
            else "MODEST DIRECTIONAL IMPROVEMENT"
        )
    random = read_csv("random_label_control")[0]
    randomf = read_csv("random_feature_control")[0]
    tests = ET.parse(OUT / "test_results.xml").getroot().find("testsuite").attrib
    audit = json.loads((OUT / "isolation_audit.json").read_text())
    repeat = json.loads((OUT / "reproducibility_audit.json").read_text())
    controls_ok = (
        random["investigation_required"] == "False" and randomf["material_improvement"] == "False"
    )
    decision = (
        "PASS"
        if controls_ok
        and tests["failures"] == "0"
        and tests["errors"] == "0"
        and audit["status"] == "PASS"
        and repeat["status"] == "PASS"
        else "FAIL"
    )
    dump(
        "research_verdict.json",
        dict(
            regression=verdict,
            classification=cv,
            methodology=decision,
            tests=tests,
            controls_sane=controls_ok,
            independent_verification_required=True,
        ),
    )
    coef = [r for r in read_csv("model_coefficients") if r["embargo_seconds"] == "0"]
    summaries = []
    for name in ("OLS", "Ridge", "Logistic", "RandomFeature"):
        cats = ["DOWN", "FLAT", "UP"] if name == "Logistic" else ["return"]
        for feature in FEATURES + (["RANDOM_CONTROL"] if name == "RandomFeature" else []):
            for category in cats:
                values = np.array(
                    [
                        float(r["standardized_coefficient"])
                        for r in coef
                        if r["model_name"] == name
                        and r["feature"] == feature
                        and r["class_name"] == category
                    ]
                )
                summaries.append(
                    dict(
                        model_name=name,
                        feature=feature,
                        class_name=category,
                        positive_folds=int((values > 0).sum()),
                        negative_folds=int((values < 0).sum()),
                        minimum=float(values.min()),
                        maximum=float(values.max()),
                        mean=float(values.mean()),
                    )
                )
    from lob_research.phase4.storage import write_csv

    write_csv("coefficient_stability", summaries)
    text = f"""# Phase 4 — Linear baselines and intraday walk-forward validation

Regression verdict: **{verdict}**.
Classification verdict: **{cv}**.
Methodology decision: **PHASE 4: {decision}**. Independent verification receipt:
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
{json.loads((OUT / "protocol_freeze.json").read_text())["utc"]}.
Session blocks are 39 minutes, [09:30,16:00) Eastern. Boundaries below are UTC;
subtract five hours for this date's Eastern time. Every interval is half-open.

{table(["Fold", "Train start", "Train end / validation start", "Validation end / test start", "Test end"], [[r["fold_id"], r["train_start"], r["train_end"], r["validation_end"], r["test_end"]] for r in read_csv("fold_manifest")])}

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

Primary features: {", ".join(FEATURES)}.
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

{table(["Fold", "Ridge alpha", "Logistic C"], [[k, next(a["parameter"] for a in json.loads((OUT / "fitted_models.json").read_text()) if a["model_name"] == "Ridge" and a["fold_id"] == k and a["embargo_seconds"] == 0), next(a["parameter"] for a in json.loads((OUT / "fitted_models.json").read_text()) if a["model_name"] == "Logistic" and a["fold_id"] == k and a["embargo_seconds"] == 0)] for k in range(1, 6)])}

## Regression

All final metrics are recomputed from persisted test predictions. Returns are
fractions; MAE/MSE retain their corresponding units. Spearman is primary, R2 secondary.

{table(["Model", "Spearman", "Pearson", "MAE", "MSE", "R2"], [[name, *[f(reg[name][k]) for k in ["spearman", "pearson", "mae", "mse", "r2"]]] for name in ["OLS", "Ridge", "Static", "Flow", "ZeroReturn", "TrainingMean"]])}

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

{table(["Model", "60s block CI lower", "60s block CI upper", "Circular-shift p"], [[name, *[f(infer[name][k]) for k in ["ci_lower", "ci_upper", "shift_p"]]] for name in ["OLS", "Ridge"]])}

1,000 paired circular moving-block draws, 60 seconds, seed 404060; 1,000 target
shifts with offsets 600..11100 seconds, seed 404600. The full 11,700-second clock
keeps all five missing test-end cells. Exact ranks are recalculated per draw.
Intervals condition on fitted models and do not include refitting uncertainty.
Circular seams and intraday nonstationarity limit coverage/null calibration;
no iid p-values or fabricated independent daily samples are used. These diagnostics
are not multiple-testing-adjusted discovery claims across variants.

## Single-feature comparison

Same 11,695 test timestamps for all three raw single-feature rank scores and Ridge.

{table(["Score", "Spearman", "Pearson"], [[name, f(reg[name]["spearman"]), f(reg[name]["pearson"])] for name in ["OBI_L1", "MICROPRICE_MINUS_MID", "OFI_1S", "Ridge"]])}

Raw feature values are rank scores, not return predictions. Their mechanical
error columns in the general regression CSV have incomparable units and are
not used as forecast-quality evidence; only their correlations are compared here.

{table(["Comparison", "Delta rho", "Paired CI lower", "Paired CI upper"], [[name, *[f(paired[name][k]) for k in ["difference", "ci_lower", "ci_upper"]]] for name in ["OBI_L1", "MICROPRICE_MINUS_MID", "OFI_1S", "BEST_SINGLE"]])}

Ridge exceeds OBI_L1 by {f(paired["OBI_L1"]["difference"])}, with positive paired
interval against that fixed benchmark. OBI_L1 is also the observed best of the
three singles. However, the preregistered conservative comparison recalculates
the maximum single-feature correlation in every paired bootstrap draw; that
interval crosses zero. Therefore the verdict remains **{verdict}**. It does
not erase the narrower positive OBI comparison or prove exact equivalence.

## Static vs Flow and ablation

Static uses imbalance/microprice/spread; flow uses both OFI windows/activity/timing.
Static rho={f(reg["Static"]["spearman"])}; flow={f(reg["Flow"]["spearman"])}; full={f(reg["Ridge"]["spearman"])}.
Full-minus-static paired delta={f(paired["Static"]["difference"])}, CI
[{f(paired["Static"]["ci_lower"])}, {f(paired["Static"]["ci_upper"])}]. This supports
incremental flow information within this session, although flow alone reverses
sign in the first test block. It does not establish cross-day stability.

{table(["Predeclared Ridge variant", "Spearman"], [[r["model_name"], f(r["spearman"])] for r in read_csv("feature_ablation")])}

Removing OFI weakens the rank score; removing microprice slightly improves its
point estimate. These fixed diagnostics do not replace the full primary model,
and their small differences are not newly selected discoveries.

## Classification

{table(["Model", "Macro F1", "Balanced accuracy", "Accuracy", "Log loss"], [[name, *[f(cla[name][k]) for k in ["macro_f1", "balanced_accuracy", "accuracy", "log_loss"]]] for name in ["Logistic", "AlwaysFLAT", "TrainingPrior"]])}

Logistic-minus-Always-FLAT Macro F1={f(ci["difference"])}, paired 60s-block 95% CI
[{f(ci["ci_lower"])}, {f(ci["ci_upper"])}]. All five folds improve Macro F1.
The increase is below the predeclared .05 CLEAR threshold, hence **{cv}**.
Accuracy increases only about 0.29 percentage points. UP and DOWN recall remain
extremely low; the classifier mostly predicts FLAT. No class weights or alternate
classification thresholds were tried. Always-FLAT's one-hot log loss uses float64
epsilon clipping/renormalization; TrainingPrior is the meaningful probabilistic
baseline (same argmax classes, substantially different loss).

{table(["Class", "Logistic precision", "Recall", "F1"], [[c, *[f(cla["Logistic"][c + "_" + m]) for m in ["precision", "recall", "f1"]]] for c in ["DOWN", "FLAT", "UP"]])}

Confusion matrices for every fold/model and concatenated tests are saved in
confusion_matrix.csv. Calibration uses fixed width .1 probability bins for UP
and DOWN, retaining empty bins. Central .1–.3 bins are reasonably close on this
day; the lowest bin understates observed direction frequency, and high-probability
tails are sparse/noisy. Probabilities are not uniformly well calibrated, and no
calibrator was fitted.

{table(["Direction probability", "Top-decile cutoff", "N", "Mean forecast", "P(DOWN)", "P(FLAT)", "P(UP)"], [[r["class_name"], f(r["cutoff"]), r["n"], f(r["mean_probability"]), f(r["DOWN_frequency"]), f(r["FLAT_frequency"]), f(r["UP_frequency"])] for r in read_csv("extreme_probability")])}

Top-decile thresholds are EX-POST DESCRIPTIVE empirical 90th percentiles, inclusive
of ties. Both tails include many opposite moves and FLATs; they are not trade entries.

## Coefficients

{table(["Feature", "Ridge positive folds", "Negative folds", "Min standardized weight", "Max standardized weight"], [[r["feature"], r["positive_folds"], r["negative_folds"], f(r["minimum"]), f(r["maximum"])] for r in summaries if r["model_name"] == "Ridge"])}

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

{table(["Fold", "OBI rho", "OLS rho", "Ridge rho", "Static rho", "Flow rho", "Logistic Macro F1", "Always-FLAT Macro F1"], [[k, *[f(next(r["spearman"] for r in rf if r["model_name"] == name and int(r["fold_id"]) == k)) for name in ["OBI_L1", "OLS", "Ridge", "Static", "Flow"]], *[f(next(r["macro_f1"] for r in cf if r["model_name"] == name and int(r["fold_id"]) == k)) for name in ["Logistic", "AlwaysFLAT"]]] for k in range(1, 6)])}

Ridge is positive in {positive}/5 test folds and negative in fold 1. Its strongest
test block is fold 4, not evidence that it works only at the close. OBI_L1 retains
positive point estimates in all five. Flow alone is especially unstable early.
These overlapping expanding training histories do not create independent models
or days. The 10-second embargo yields Ridge rho
{f(next(r["spearman"] for r in read_csv("regression_metrics") if r["model_name"] == "Ridge" and r["embargo_seconds"] == "10"))}
and Logistic Macro F1
{f(next(r["macro_f1"] for r in read_csv("classification_metrics") if r["model_name"] == "Logistic" and r["embargo_seconds"] == "10"))};
the substantive conclusions do not change.

## Controls and independent checks

Random-label Ridge: training labels circularly rotated by 600 seconds only;
rho={f(random["spearman"])}, CI [{f(random["ci_lower"])}, {f(random["ci_upper"])}],
shift p={f(random["shift_p"])}. It is near noise and materially below the aligned
model. No investigation trigger fired, no offsets were retried.
Random-feature Ridge reuses seed 161803; delta rho versus full=
{f(randomf["control_minus_full"])}, paired CI
[{f(randomf["difference_ci_lower"])}, {f(randomf["difference_ci_upper"])}]. The random
weight is positive in all folds but only approximately 0.00000029–0.00000094,
well below dominant signal coefficients, and does not improve OOS association.
It is not persistently dominant. No random feature redraw occurred.

Future mutation changed **{audit["future_mutated_rows"]:,} later Phase 2 rows across
{audit["future_mutated_columns"]} payload columns** in memory, preserving only
authoritative timestamp/source-order/event-index keys. Earlier grid, model
parameters, validation selection, predictions and metrics were unchanged. TEST
return and class mutations leave fits and predictions fixed while both task
metrics change. Raw and grid storage-order shuffles reproduce results after
canonical chronological restoration. **{audit["numerical_checks"]} real-data isolation
checks PASS**. Two complete deterministic experiment runs agree, with
**{repeat["checks"]} comparisons PASS** at absolute tolerance 1e-12; model artifacts
match exactly. Thread limit one, fixed seeds, pinned Phase 4 requirements.

Full `pytest -q`: **{tests["tests"]} passed, {tests["failures"]} failures,
{tests["errors"]} errors, {tests["skipped"]} skipped**; original 101 cases unchanged.
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
"""
    Path("PHASE4_REPORT.md").write_text(text, encoding="utf-8")
    print(json.dumps(dict(regression=verdict, classification=cv, methodology=decision), indent=2))


if __name__ == "__main__":
    main()
