# Research log

## 2026-10-11 — Phase 3 preregistration, before predictive results

The frozen Phase-2 matrix hash was verified as `a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96`. No predictive results have been calculated at this entry. PHASE3_PROTOCOL.md and PHASE3_FEATURE_SET.md govern the new work; the following hypotheses will not be rewritten afterward.

1. Positive order-book imbalance should be associated with positive future mid-price movement.
2. Microprice deviation should provide directional information at very short horizons.
3. Order-flow imbalance should contain stronger or more immediate information than static imbalance.
4. Predictive association should decay as horizon increases.
5. Full event-level estimates may overstate statistical confidence relative to fixed-time sampling.

Primary horizon is 1s, primary sample a 1s clock grid, primary statistic Spearman, primary block length 60s. Exactly ten existing core features; fixed seeds; no models or PnL. Results will be appended below this immutable preregistration entry after analysis.

## Phase 3 execution note

The first and only numerical study completed and saved its matrix, CSVs, manifest and complete status. Its final console JSON print then failed on a NumPy boolean. Only that print was amended to serialize NumPy scalars; no analytic module, protocol, seed, statistic or saved study output changed, and the numerical study was not rerun. Original runner bytes and original/current code digests are preserved in reports/phase3/console_fix_audit.json and run_phase3_at_execution.py.

## Phase 3 frozen results

MODEST WITHIN-DAY MICROSTRUCTURE ASSOCIATION. No hypotheses, seeds, features or horizons were changed.

- Hypothesis 1: SUPPORTED. 4/4 imbalance columns satisfy the predeclared consistency rubric; all four have positive full-day rho and positive point estimates in all five time blocks.
- Hypothesis 2: SUPPORTED. Microprice has positive 1s association and monotonic mean returns; its 100ms rank association is approximately zero and slightly negative. Support is limited to the primary 1s result, not all horizons.
- Hypothesis 3: PARTIALLY SUPPORTED. At 1s, |OFI_1S rho| is smaller than |OBI_L1 rho|. At 100ms, |OFI_100MS rho| is larger but NEGATIVE, contrary to the directional hypothesis. Magnitude-only partial support is not positive-flow confirmation or a formal dominance test.
- Hypothesis 4: NOT SUPPORTED. 0/7 signed features show nonincreasing absolute rho across all four horizons. Static imbalance and microprice generally strengthen over longer horizons; OFI changes sign.
- Hypothesis 5: INCONCLUSIVE. Event/grid magnitudes differ in both directions. No naive event-level standard errors were computed, so excess statistical confidence is not directly measured. Dependence remains a design constraint rather than a demonstrated iid comparison.

Full numerical evidence: PHASE3_REPORT.md and reports/phase3/*.csv. Original preregistration entry above remains verbatim. No Phase 4 work.


## Phase 4 preregistered hypotheses (before any model fits/results)

H1: Combining microstructure features with Ridge should modestly improve on OBI_L1 alone.

H2: Dynamic flow features should add incremental information beyond static book imbalance.

H3: Logistic Regression should improve directional discrimination over Always-FLAT in Macro F1, even if raw accuracy improvement is modest.

H4: Linear model performance may vary substantially across intraday periods.

Frozen design: PHASE4_PROTOCOL.md and PHASE4_FEATURE_SET.md, five chronological folds in reports/phase4/fold_manifest.csv. No test-driven feature or horizon changes.


## Phase 4 frozen results

Primary horizon, feature membership, seeds, folds, grids and verdict rules remain as preregistered. Five walk-forward folds; 11,695 unique test samples.

H1: PARTIAL. Ridge rho 0.050746889 versus OBI_L1 0.036005794; paired difference 0.014741095, 95% CI [0.001564237, 0.027937422]. Against the bootstrap maximum of three predeclared single features the CI is [-0.001748198, 0.023105906], so no reliable incremental value under the frozen overall rubric.

H2: SUPPORTED WITHIN DAY. Static rho 0.031452437, flow 0.033729048, full 0.050746889; full-minus-static CI [0.006946035, 0.031667812]. Flow has a negative first test fold.

H3: SUPPORTED WITHIN DAY. Logistic Macro F1 0.302183044 versus Always-FLAT 0.255601835, paired improvement CI [0.037302941, 0.055118993]. Classification verdict MODEST DIRECTIONAL IMPROVEMENT; UP/DOWN recall remains very low.

H4: SUPPORTED DESCRIPTIVELY. Ridge fold rho ranges from -0.010593069 to 0.102703965, positive in four of five periods. These are not independent daily replications.

Random-label control rho -0.012034890 (CI contains zero); random feature does not improve OOS rho. Real future mutation, test-label isolation, shuffled storage and duplicate deterministic runs pass. Independent verification found and prompted a documented Phase 4 constant-baseline Pearson guard; prediction bytes, fits and all upstream results remain unchanged.

Full suite: 114 passed, zero failures/errors/skips. Independent verification: 6,127 checks, zero critical failures. PHASE 4: PASS. Regression verdict NO RELIABLE INCREMENTAL VALUE OVER SINGLE FEATURES. Full report: PHASE4_REPORT.md. Prior Phase 3 exposure of this day's data limits confirmatory interpretation. No Phase 5, nonlinear models, execution simulation or PnL. STOP.


## Phase 5 preregistered hypotheses (before tree fitting or test results)

H1: Nonlinear tree models may capture threshold/saturation effects not represented by Ridge and therefore modestly improve OOS Spearman.

H2: Static book and flow features may interact nonlinearly.

H3: Any nonlinear improvement may be unstable because the effective independent sample size is small.

H4: For classification, nonlinear models may improve Macro F1 more than raw accuracy because the dominant FLAT class makes accuracy a weak metric.

Frozen rules: PHASE5_PROTOCOL.md and PHASE5_FEATURE_SET.md. Four candidates per family, validation-only family selection, unchanged Phase 4 folds and benchmarks; no test-driven retuning.


## Phase 5 frozen results

No feature, fold, horizon, candidate set, seed or verdict rule changed after results. Primary best-nonlinear comparison remains the per-fold validation-selected family, not the highest test-family point estimate.

H1: SUPPORTED WITHIN THIS EXAMINED DAY. Ridge rho 0.050746889; RF 0.097539968; LightGBM 0.098636590; validation-selected 0.098112770. Selected-minus-Ridge 0.047365881, paired 60-second CI [0.030307867, 0.067164541]; RF and LightGBM each beat Ridge in 5/5 test folds.

H2: SUPPORTED DESCRIPTIVELY. LightGBM static/flow/full rho 0.040221836 / 0.077954333 / 0.098636590. Full-minus-static and full-minus-flow paired intervals are positive. Fixed model response surfaces are non-additive; no causal interaction claim is established.

H3: PARTIAL. Intraday magnitudes and training gaps vary, but relative nonlinear wins persist in 5/5 folds. Effective independent sample size remains far smaller than raw rows. Median primary regression train-test gap 0.089384181 does not trigger the frozen .15 severe-gap rule, while substantial in-sample optimism is explicitly reported.

H4: SUPPORTED WITHIN DAY. Logistic Macro F1 0.302183044; RF 0.345043280; LightGBM (validation selected in every classification fold) 0.384455023. Gain 0.082271979, paired CI [0.069765722, 0.094249867], with only about 0.28 percentage-point accuracy improvement. UP/DOWN recall remains limited.

Random training-block labels rho 0.009221409, CI [-0.010675057, 0.030125291]; separate 600-second target-rotation placebo -0.009094645, CI [-0.027387788, 0.007080909]. Fixed random feature does not materially improve validation/test or dominate importance. All controls sane. Future mutation of 64,191 later rows, test-label isolation, row-order invariance, invalid future-feature rejection and two deterministic runs pass.

OFI_100MS response remains broadly decreasing end-to-end in all five folds, with one near-zero local bump and only 3–4 quantile points per curve; OFI_1S is broadly positive but locally nonlinear. Definitions unchanged. No causality inferred.

Full suite: 126 passed, zero failures/errors/skips. Independent verification: 3,120 checks, zero critical failures. Regression verdict RELIABLE NONLINEAR IMPROVEMENT; classification verdict RELIABLE NONLINEAR DIRECTIONAL IMPROVEMENT, both limited to this already-examined stock/day. PHASE 5: PASS. No execution, PnL, alternate-horizon tuning or Phase 6. STOP.

## Phase 6 preregistration — before execution PnL

H1: The LightGBM mid-price predictive improvement may be substantially reduced after aggressive bid/ask execution.

H2: Higher prediction strength should be associated with better economic outcomes, but spread may dominate weaker signals.

H3: Latency should degrade the short-horizon signal.

H4: LightGBM should economically outperform Ridge if its predictive improvement is genuinely useful.

H5: The signal may remain statistically predictive while failing to be executable after spread and costs.

Protocol 6.0.0 fixes validation-only 10% tails, stateful one share, 10ms aggressive BBO execution and all robustness/control assumptions. No real PnL viewed.

## Phase 6 outcome — frozen execution assumptions retained

Primary LightGBM 10% validation tails, 10ms, one share: gross mid $2.4475, spread $48.0935, quote PnL $-45.6460, no extra fees. 2,989 fills, 193 flips, 0/5 positive folds. No primary depth/missing failures.

H1/H5 supported: spread consumes edge. H2 gross-strength association supported descriptively, every bin loses after crossing. H3 not supported monotonically: 10ms latency effect -$0.0710 versus 0ms. H4 not supported for aggregate dollars: Ridge -$41.3840, RF -$43.5250, LightGBM -$45.6460; turnover differs.

60s/1,000 bootstrap normalized PnL CI [-0.922629664, -0.786712920]. 500-shift one-sided PnL p=0.778443114. Random/inversion/600s controls sane; delayed mid gross loses alignment. All predefined thresholds/latencies lose; no result-driven changes.

PREDICTIVE SIGNAL DOES NOT SURVIVE PRIMARY EXECUTION ASSUMPTIONS. PHASE 6: PASS. Full pytest 139 passed, zero failures/errors. Independent replay PASS; discrepancy 3.27e-13 USD. Single examined 2003 stock/day only. STOP; wait for Phase 7 authorization.
