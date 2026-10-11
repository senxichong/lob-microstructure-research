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
