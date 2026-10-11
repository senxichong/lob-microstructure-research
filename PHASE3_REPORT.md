# Phase 3 — Univariate Microstructure Signal Research & Statistical Validation

**Overall research verdict: MODEST WITHIN-DAY MICROSTRUCTURE ASSOCIATION.** Methodology decision: **PHASE 3: PASS**.

**One stock, one trading day, 2003.** These results concern only this historical MSFT session. They establish no cross-day or cross-stock stability, modern-market relevance, live tradability, capacity or execution profitability. A feature predicting the midprice is not automatically tradable alpha.

## Sample and immutable upstream

Instrument/date: **MSFT — 2003-01-03**, Nasdaq-hosted historical ITCH 2.0, legacy Island/INET lineage (historically inferred). Session: **[09:30,16:00) America/New_York**, **[14:30,21:00) UTC**. This remains a time-of-day proxy without certified halt/auction states. Phase 2 has 162,879 rows, 220 columns and 149 frozen features. Phase 3 uses exactly ten predeclared features.

Verified Phase-2 SHA-256: `a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96`. The matrix, Phase-2 code, tests, labels, root config and all original reports are unchanged. Phase-3 grid SHA-256: `1781e33df0ff416cdf9d377fd3a66eed3fed1718f21efec9f465b292276f9e5b`; path `data/processed/phase3/msft_20030103_grid_v3.0.0.parquet`.

There are **23,400 grid points**, **23,398 valid 1s targets** and 23,399 points with valid current states. The opening grid point precedes the first retained event and is unavailable; the final point's 1s target equals the excluded close. OFI_1S and EVENT_RATE_1S have 23,397 valid primary pairs because one additional source event lacks full rolling history; other primary features have 23,398. FLAT observations are retained.

At grid time g, select the last Phase-2 state with timestamp<=g, resolving ties by final source order. Features are copied at that post-event source key; their rolling windows and time-since values are NOT recomputed at g. Targets ARE reanchored to g+h, using the final same-session state<=g+h, with exact integer-mid direction, invalid-segment rejection and tau<close. No-update intervals persist. Event-anchored forward labels are never substituted for grid targets.

Grid-state age: mean 340.617ms, median 148ms, p95 1381ms, p99 2759ms, maximum 7359ms. No staleness cutoff was imposed. Carried source-state features can be materially older than the grid clock; this is part of the frozen estimand, especially for 100ms OFI.

| Horizon | Available | Missing | UP | FLAT | DOWN |
|---|---|---|---|---|---|
| 100MS | 23399 | 1 | 1021 | 21345 | 1033 |
| 1S | 23398 | 2 | 4776 | 13912 | 4710 |
| 5S | 23394 | 6 | 8868 | 5790 | 8736 |
| 10S | 23389 | 11 | 10264 | 3232 | 9893 |

## Preregistration and statistical method

The protocol and feature set were hash-frozen at `2026-10-11T06:18:19.652115+00:00`, before study execution. Primary horizon remains **1 second**. No feature, direction, seed, block length or boundary was changed after results. `PHASE3_PROTOCOL.md`, `PHASE3_FEATURE_SET.md`, `config.phase3.yaml` and the immutable opening of `RESEARCH_LOG.md` record all decisions.

Primary statistic: Spearman correlation of average ranks on finite grid pairs. Pearson is secondary. No ordinary iid/asymptotic p-value is reported. Adjacent clock samples remain dependent, and event rows are strongly dependent through clustering, persistent books, overlapping history and labels. Neither 162,879 events nor 23,400 grid points are asserted to be independent observations.

Primary inference uses **1,000 paired circular moving-block bootstrap draws of 60 seconds**, with exact ranks recalculated in each draw. Seeds are 314219 (60s), 314189 (30s robustness) and 314459 (300s robustness). All share paired clock positions, including missing cells; sample weights implement literal duplication exactly. Percentile 95% intervals are reported for rho and Q5-Q1 mean return. Q5-Q1 intervals condition on fixed observed-day bin edges, not estimated-threshold uncertainty.

The null uses **1,000 target circular shifts**, seed 271828, offsets uniformly sampled from 600..22,800 seconds inclusive. Target ordering and missingness shift together. Two-sided p=(1+number of |null rho|>=|observed rho|)/1001. Minimum p is 0.000999, not zero. This approximate alignment null preserves circular temporal structure but does not guarantee exact exchangeability under intraday nonstationarity. The circular bootstrap also creates an artificial end/start seam and assumes approximate stationarity.

BH correction covers exactly the ten primary feature p-values. BY is also reported because positive dependence needed for BH is not guaranteed by correlated features. Both corrections concern these approximate marginal p-values and cannot fix their calibration. Secondary horizon/state/event diagnostics have no discovery-level inferential status. Method conventions: [SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), [SciPy BH/BY](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

## Primary 1-second results

Rows remain in preregistered order. Delta return is Q5 minus Q1 mean **mid return**, expressed in basis points (return x10,000); delta edge is the difference in P(UP)-P(DOWN), a probability difference. These are ex-post outcome contrasts, not strategy returns. Mono is the correlation of quintile index with quintile mean return; nan means five nonempty bins do not exist.

| Feature | N | Spearman rho | 60s-block 95% CI | Shift p | BH q | BY q | Delta return bp | Delta edge | Mono |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| OBI_L1 | 23,398 | 0.043818 | [0.030828, 0.056045] | 0.000999 | 0.001665 | 0.004877 | 0.196446 | 0.062351 | 1.000 |
| OBI_L5 | 23,398 | 0.030302 | [0.017675, 0.043236] | 0.000999 | 0.001665 | 0.004877 | 0.094640 | 0.060043 | 1.000 |
| OBI_L10 | 23,398 | 0.022133 | [0.009210, 0.036086] | 0.001998 | 0.002854 | 0.008360 | 0.065119 | 0.038048 | 0.900 |
| WOBI_L10 | 23,398 | 0.035528 | [0.023083, 0.048850] | 0.000999 | 0.001665 | 0.004877 | 0.127995 | 0.062400 | 1.000 |
| MICROPRICE_MINUS_MID | 23,398 | 0.032462 | [0.018904, 0.045642] | 0.000999 | 0.001665 | 0.004877 | 0.173537 | 0.032270 | 1.000 |
| OFI_100MS | 23,398 | -0.033494 | [-0.049572, -0.017727] | 0.000999 | 0.001665 | 0.004877 | 0.026742 | -0.048158 | nan |
| OFI_1S | 23,397 | 0.035205 | [0.017593, 0.053248] | 0.000999 | 0.001665 | 0.004877 | 0.185554 | 0.037704 | nan |
| RELATIVE_SPREAD | 23,398 | -0.008985 | [-0.024082, 0.006835] | 0.210789 | 0.263487 | 0.771744 | -0.029944 | -0.011395 | -0.500 |
| EVENT_RATE_1S | 23,397 | -0.002766 | [-0.018211, 0.013859] | 0.686314 | 0.686314 | 1.000000 | -0.004922 | -0.003115 | -0.300 |
| TIME_SINCE_PREVIOUS_TRADE | 23,398 | 0.004613 | [-0.009893, 0.018721] | 0.481518 | 0.535021 | 1.000000 | 0.030392 | 0.004274 | 0.600 |

Seven primary columns pass both BH and BY at .05, but they are not seven independent discoveries. All four imbalance columns and microprice have small positive rank associations. OFI_100MS has a significant **negative** rank association, contrary to the predeclared positive direction. OFI_1S is positive overall but less stable across time and regimes. The three context features have intervals containing zero and no adjusted primary evidence. No sign was flipped.

### Bootstrap sensitivity and outcome-difference intervals

| Feature | 30s rho CI | 60s rho CI (primary) | 300s rho CI | 60s Q5-Q1 mean-return CI, bp |
|---|---|---|---|---|
| OBI_L1 | [0.031188, 0.055943] | [0.030828, 0.056045] | [0.029648, 0.058006] | [0.155326, 0.236705] |
| OBI_L5 | [0.017352, 0.043253] | [0.017675, 0.043236] | [0.017699, 0.042877] | [0.061055, 0.133558] |
| OBI_L10 | [0.008668, 0.035336] | [0.009210, 0.036086] | [0.009483, 0.034851] | [0.029853, 0.105605] |
| WOBI_L10 | [0.022637, 0.048232] | [0.023083, 0.048850] | [0.022430, 0.048337] | [0.090413, 0.168155] |
| MICROPRICE_MINUS_MID | [0.018488, 0.044844] | [0.018904, 0.045642] | [0.016224, 0.048598] | [0.132551, 0.212556] |
| OFI_100MS | [-0.048620, -0.018079] | [-0.049572, -0.017727] | [-0.053225, -0.014550] | [-0.014912, 0.074752] |
| OFI_1S | [0.018295, 0.050380] | [0.017593, 0.053248] | [0.014060, 0.056933] | [0.143185, 0.233383] |
| RELATIVE_SPREAD | [-0.023870, 0.006263] | [-0.024082, 0.006835] | [-0.024058, 0.006098] | [-0.077488, 0.022480] |
| EVENT_RATE_1S | [-0.020544, 0.013054] | [-0.018211, 0.013859] | [-0.018276, 0.014643] | [-0.046347, 0.040151] |
| TIME_SINCE_PREVIOUS_TRADE | [-0.008943, 0.019049] | [-0.009893, 0.018721] | [-0.009939, 0.018958] | [-0.013092, 0.070834] |

All reported bootstrap intervals use 1,000 finite rho replicates per feature and length. The sign/exclusion conclusions for primary rho are unchanged at 30s and 300s. This sensitivity does not establish nominal coverage or eliminate slow intraday confounding.

## Quintiles, direction and price scale

Full-day empirical cutoffs are **EX-POST DESCRIPTIVE BINS, not executable online thresholds**. Equal values stay together in the lower bin. OBI_L1/L5/WOBI and microprice have mean-return monotonicity 1.0; OBI_L10 has 0.9 with a local reversal. Direction-edge monotonicity is 1.0 for OBI_L1/L5 and 0.9 for OBI_L10/WOBI/microprice; these high five-point summaries are diagnostics, not separate significance tests.

OFI_100MS has only Q1 and Q5 populated (20,260 and 3,138 rows); Q2-Q4 remain empty because all four quantile cutoffs coincide at zero. OFI_1S has four populated bins with Q3 empty. Their five-bin monotonicity is explicitly unavailable. We did not split ties, relabel bins or retune boundaries. OFI_100MS has positive Q5-Q1 mean return but negative direction-edge difference and negative overall rank association; mean effects and ranked directional behavior need not agree for tied/heavy-tailed distributions.

OBI_L1 Q1/Q5 mean returns are -0.089308/+0.107138 bp, with P(UP)-P(DOWN) -0.028284/+0.034067 (difference 6.235081 percentage points). The corresponding mean USD movement contrast is $0.001046760. Microprice Q5-Q1 mean-return contrast is 0.173537 bp ($0.000925049); OFI_1S is 0.185554 bp ($0.000989183). These are small conditional outcome differences. No unverified tick conversion is made.

Every primary feature has per-bin counts, mean/median returns and direction probabilities in `primary_feature_quintiles.csv`; conditional feature p05/p25/median/p75/p95/IQR by UP/FLAT/DOWN are in `primary_direction_statistics.csv`. All populated primary quintile median future returns are zero, reflecting retained FLAT observations. Fixed imbalance bins are separate secondary diagnostics in `extreme_imbalance.csv`. Median absolute 1s mid movement is $0.000 and median spread is $0.029; this is descriptive price-scale context only, not an execution-profit calculation.

## Predeclared hypothesis outcomes

| Hypothesis | Verdict | Evidence |
|---|---|---|
| 1 | SUPPORTED | 4/4 imbalance columns satisfy the predeclared consistency rubric; all four have positive full-day rho and positive point estimates in all five time blocks. |
| 2 | SUPPORTED | Microprice has positive 1s association and monotonic mean returns; its 100ms rank association is approximately zero and slightly negative. Support is limited to the primary 1s result, not all horizons. |
| 3 | PARTIALLY SUPPORTED | At 1s, |OFI_1S rho| is smaller than |OBI_L1 rho|. At 100ms, |OFI_100MS rho| is larger but NEGATIVE, contrary to the directional hypothesis. Magnitude-only partial support is not positive-flow confirmation or a formal dominance test. |
| 4 | NOT SUPPORTED | 0/7 signed features show nonincreasing absolute rho across all four horizons. Static imbalance and microprice generally strengthen over longer horizons; OFI changes sign. |
| 5 | INCONCLUSIVE | Event/grid magnitudes differ in both directions. No naive event-level standard errors were computed, so excess statistical confidence is not directly measured. Dependence remains a design constraint rather than a demonstrated iid comparison. |

## Horizon decay / persistence

| Feature | 100ms rho | 1s rho — PRIMARY | 5s rho | 10s rho |
|---|---|---|---|---|
| OBI_L1 | 0.008525 | 0.043818 | 0.077688 | 0.082070 |
| OBI_L5 | 0.007252 | 0.030302 | 0.048071 | 0.050064 |
| OBI_L10 | 0.007434 | 0.022133 | 0.037382 | 0.046604 |
| WOBI_L10 | 0.010431 | 0.035528 | 0.060966 | 0.064346 |
| MICROPRICE_MINUS_MID | -0.000223 | 0.032462 | 0.072667 | 0.081568 |
| OFI_100MS | -0.074146 | -0.033494 | 0.049016 | 0.046959 |
| OFI_1S | -0.039037 | 0.035205 | 0.115278 | 0.099557 |
| RELATIVE_SPREAD | -0.004066 | -0.008985 | -0.014621 | -0.000532 |
| EVENT_RATE_1S | -0.009346 | -0.002766 | -0.022695 | -0.014282 |
| TIME_SINCE_PREVIOUS_TRADE | 0.001370 | 0.004613 | 0.024134 | 0.018576 |

Association does **not** show the proposed general monotonic decay. Static imbalance strengthens toward 5-10s. Microprice is nearly zero/slightly negative at 100ms and positive at later horizons. OFI_100MS moves from negative at 100ms/1s to positive at 5s/10s; OFI_1S is negative at 100ms, positive at 1s, larger at 5s and somewhat lower at 10s. This does not authorize changing the 1s primary horizon. Horizon-specific quintile direction differences use the original 1s edges and are saved in `horizon_decay.csv` and `horizon_quintiles.csv`.

## Time-of-day and state stability

Blocks are fixed 78-minute windows: A 09:30-10:48, B 10:48-12:06, C 12:06-13:24, D 13:24-14:42, E 14:42-16:00 Eastern. Each contains 4,680 grid points; per-feature valid counts are explicitly recorded.

| Feature | A rho | B rho | C rho | D rho | E rho | Interior rho |
|---|---|---|---|---|---|---|
| OBI_L1 | 0.069116 | 0.047048 | 0.012721 | 0.026212 | 0.055435 | 0.037626 |
| OBI_L5 | 0.061602 | 0.024079 | 0.004833 | 0.022743 | 0.028732 | 0.028000 |
| OBI_L10 | 0.034596 | 0.018059 | 0.015881 | 0.008121 | 0.025047 | 0.022557 |
| WOBI_L10 | 0.069677 | 0.023391 | 0.015290 | 0.016936 | 0.042560 | 0.031180 |
| MICROPRICE_MINUS_MID | 0.060445 | 0.027327 | 0.003036 | 0.013893 | 0.045467 | 0.025061 |
| OFI_100MS | -0.001236 | -0.042714 | -0.092487 | -0.035565 | -0.019169 | -0.039378 |
| OFI_1S | 0.055260 | 0.041364 | -0.042376 | 0.019178 | 0.071148 | 0.030888 |
| RELATIVE_SPREAD | -0.006100 | -0.004581 | -0.009809 | 0.005705 | -0.008995 | -0.009269 |
| EVENT_RATE_1S | 0.009963 | -0.026063 | -0.006102 | -0.007517 | 0.014024 | -0.010438 |
| TIME_SINCE_PREVIOUS_TRADE | -0.007007 | 0.017274 | 0.016826 | 0.002641 | -0.007520 | 0.011411 |

Imbalance and microprice point estimates are positive in all five periods; their size is generally larger near the open and weaker in the middle. This is distributed sign evidence, not proof that each period is significant. OFI_1S is negative in block C and positive in four blocks. OFI_100MS is negative in every block, against the original direction hypothesis. Removing the first/last 30 minutes leaves 19,800 interior pairs and preserves those full-day directional signs; the full session remains primary.

### EX-POST DESCRIPTIVE REGIME ANALYSIS

| Feature | Activity low | Activity medium | Activity high | Spread low | Spread medium | Spread high |
|---|---|---|---|---|---|---|
| OBI_L1 | 0.043548 | 0.044261 | 0.046330 | 0.101841 | 0.018267 | 0.019122 |
| OBI_L5 | 0.044545 | 0.029941 | 0.018919 | 0.067951 | -0.002096 | 0.029226 |
| OBI_L10 | 0.054865 | 0.020037 | -0.004665 | 0.035283 | 0.010749 | 0.020949 |
| WOBI_L10 | 0.059576 | 0.035587 | 0.015116 | 0.075597 | 0.006747 | 0.029219 |
| MICROPRICE_MINUS_MID | 0.032545 | 0.031660 | 0.036174 | 0.095472 | 0.014913 | 0.021887 |
| OFI_100MS | -0.055759 | -0.043907 | -0.005204 | 0.052632 | -0.030068 | -0.084328 |
| OFI_1S | -0.043379 | -0.000423 | 0.112587 | 0.113408 | 0.036228 | -0.015977 |

Activity terciles contain 7,987 / 8,765 / 6,645 usable source-activity observations because equal integer rates are not split; spread terciles contain 7,806 / 7,802 / 7,790 target-valid observations (OFI_1S has one additional missing value). These ex-post regimes are not online filters. OBI_L1/microprice remain positive across activity states, while deep OBI weakens at high activity. OFI_1S changes from negative low-activity to positive high-activity association; low-spread effects are stronger, and its high-spread estimate is negative. OFI_100MS also changes sign across spread states. These results caution against a universal flow interpretation.

## Feature redundancy

| Pair | Grid Spearman |
|---|---|
| OBI_L1 / OBI_L5 | 0.333962 |
| OBI_L1 / OBI_L10 | 0.212844 |
| OBI_L5 / OBI_L10 | 0.622249 |
| OBI_L1 / WOBI_L10 | 0.567626 |
| OBI_L1 / MICROPRICE_MINUS_MID | 0.944948 |
| OFI_1S / OFI_100MS | 0.431577 |
| OFI_1S / EVENT_RATE_1S | -0.000187 |

OBI_L1 and microprice deviation have rho approximately 0.945 and are strongly redundant. The two named families satisfy the preregistered overall rubric but are NOT two independent economic discoveries. OFI window variants also overlap. A near-zero OFI/activity rank correlation does not remove the strong conditional activity dependence shown above. All 100 pairwise values are saved in `feature_correlation.csv`.

## Event-level versus grid

| Feature | Event N | Event rho — DESCRIPTIVE | Grid N | Grid rho — PRIMARY |
|---|---|---|---|---|
| OBI_L1 | 162877 | 0.055444 | 23398 | 0.043818 |
| OBI_L5 | 162877 | 0.024922 | 23398 | 0.030302 |
| OBI_L10 | 162877 | 0.013296 | 23398 | 0.022133 |
| WOBI_L10 | 162877 | 0.030682 | 23398 | 0.035528 |
| MICROPRICE_MINUS_MID | 162877 | 0.040181 | 23398 | 0.032462 |
| OFI_100MS | 162876 | -0.028515 | 23398 | -0.033494 |
| OFI_1S | 162875 | 0.058718 | 23397 | 0.035205 |
| RELATIVE_SPREAD | 162877 | -0.001632 | 23398 | -0.008985 |
| EVENT_RATE_1S | 162875 | -0.015735 | 23397 | -0.002766 |
| TIME_SINCE_PREVIOUS_TRADE | 162876 | 0.005982 | 23398 | 0.004613 |

Event-level estimates are HIGH-FREQUENCY DESCRIPTIVE ESTIMATES. They are not uniformly larger: OBI_L1, microprice and OFI_1S are larger in magnitude, but several depth imbalances are smaller. Event/grid differences combine activity weighting, repeated information, source-feature age and distinct event/grid target anchors; they cannot be attributed solely to pseudo-replication. The much larger event count is not independent evidence or permission to use iid uncertainty. Primary inference remains grid-based and block-aware.

## Controls and independent verification

Exactly one random feature, seed 161803: rho -0.001168989, 60s bootstrap CI [-0.013650213, 0.011711192], shift p=0.856143856. **PASS**, no redraws. It is excluded from the ten-feature FDR family. Its 30s/300s intervals also contain zero.

| Feature | Aligned rho on same pairs | 600s lag placebo rho | Absolute association weakens |
|---|---|---|---|
| OBI_L1 | 0.042868 | -0.003351 | True |
| OBI_L5 | 0.028333 | -0.002025 | True |
| OBI_L10 | 0.021944 | -0.014744 | True |
| WOBI_L10 | 0.034122 | -0.009145 | True |
| MICROPRICE_MINUS_MID | 0.030468 | -0.006697 | True |
| OFI_100MS | -0.034032 | 0.000640 | True |
| OFI_1S | 0.035312 | 0.002355 | True |
| RELATIVE_SPREAD | -0.005417 | -0.000094 | True |
| EVENT_RATE_1S | -0.002913 | 0.007102 | False |
| TIME_SINCE_PREVIOUS_TRADE | 0.004216 | 0.004709 | False |

The fixed +10-minute causal lag weakens all seven directional features on identical surviving pairs. It weakens spread but does not weaken the two context features with already negligible primary associations; those exceptions are reported rather than treated as positive evidence. No alternative placebo offset was tried. Sign inversion passes all seven signed features to 1e-12.

Independent audit: **1,257 checks, 0 critical failures**; 10 independent small-sample feature analyses and 8,080 scalar comparisons, 0 mismatches. Grid source/target selections are independently reconstructed with Python bisect; full primary rho is recomputed with SciPy; scalar average ranks on 101 deterministic samples per feature independently reproduce ranks/rho/quintiles. All saved interval bounds and p-values are recomputed from draw distributions, with 21 null draws per feature and three literal bootstrap replicates per feature/block length independently recalculated. This is independent arithmetic/code, not an external dataset replication.

**27 deterministic real grid traces** cover time, positive/negative/near-zero imbalance, signed flow/microprice, activity extremes and UP/FLAT/DOWN examples. Traces and row-level manual comparisons remain local-only. `scripts/verify_phase3.py` also checks immutable hashes, feature list, primary horizon, seeds, FDR, bin reconciliation, time blocks, controls, event/grid separation and absence of model artifacts; the final report table is reconciled to CSV values.

Complete repository suite: **101 passed, 0 failures/errors, 0 skipped, 0 warnings, 5.97s**. All 49 Phase-1 and 32 Phase-2 tests remain unchanged; 20 Phase-3 cases cover grid/as-of/ties/invalid intervals, bins, ranks, literal versus weighted block resampling, shifts/seeds, FDR, time blocks, random controls, lagging/sign inversion and event/grid target separation.

## Figures

1. [OBI_L1 quintile mean return](reports/phase3/figures/01_obi_quintile_return.png) ([SVG](reports/phase3/figures/01_obi_quintile_return.svg)).
2. [OFI_1S quintile mean return](reports/phase3/figures/02_ofi_quintile_return.png) ([SVG](reports/phase3/figures/02_ofi_quintile_return.svg)).
3. [Microprice quintile direction difference](reports/phase3/figures/03_microprice_direction_edge.png) ([SVG](reports/phase3/figures/03_microprice_direction_edge.svg)).
4. [Primary rho with block intervals](reports/phase3/figures/04_primary_spearman.png) ([SVG](reports/phase3/figures/04_primary_spearman.svg)).
5. [Horizon persistence/decay](reports/phase3/figures/05_horizon_decay.png) ([SVG](reports/phase3/figures/05_horizon_decay.svg)).
6. [UP/FLAT/DOWN by quintile](reports/phase3/figures/06_direction_probabilities.png) ([SVG](reports/phase3/figures/06_direction_probabilities.svg)).
7. [Time-of-day stability](reports/phase3/figures/07_time_of_day.png) ([SVG](reports/phase3/figures/07_time_of_day.svg)).
8. [Feature correlation heatmap](reports/phase3/figures/08_feature_correlation.png) ([SVG](reports/phase3/figures/08_feature_correlation.svg)).

All eight figures explicitly state MSFT, single historical trading day and 2003-01-03. PNG and editable SVG exports were visually inspected; empty bins are visible and not interpolated into fake observations.

## Family and overall research verdicts

| Family | Verdict |
|---|---|
| Imbalance | CONSISTENT WITH SHORT-HORIZON PREDICTIVE INFORMATION |
| Microprice | CONSISTENT WITH SHORT-HORIZON PREDICTIVE INFORMATION |
| Order flow | WEAK / FRAGILE ASSOCIATION |
| Context | NO RELIABLE ASSOCIATION |

| Feature | Evidence rubric | Same-sign blocks | Nonempty quintiles |
|---|---|---|---|
| OBI_L1 | CONSISTENT EVIDENCE | 5/5 | 5 |
| OBI_L5 | CONSISTENT EVIDENCE | 5/5 | 5 |
| OBI_L10 | CONSISTENT EVIDENCE | 5/5 | 5 |
| WOBI_L10 | CONSISTENT EVIDENCE | 5/5 | 5 |
| MICROPRICE_MINUS_MID | CONSISTENT EVIDENCE | 5/5 | 5 |
| OFI_100MS | WEAK EVIDENCE | 5/5 | 2 |
| OFI_1S | WEAK EVIDENCE | 4/5 | 4 |
| RELATIVE_SPREAD | NO EVIDENCE | 4/5 | 5 |
| EVENT_RATE_1S | NO EVIDENCE | 3/5 | 5 |
| TIME_SINCE_PREVIOUS_TRADE | NO EVIDENCE | 3/5 | 5 |

**MODEST WITHIN-DAY MICROSTRUCTURE ASSOCIATION**. The positive book-state associations are small and highly redundant; flow relationships depend on horizon/time/state and partly oppose the original hypothesis. This is within-session microstructure evidence only. The rubric intentionally marks flow fragile because its signed/five-quintile stability requirements are not met, even though its two-sided tests reject the approximate alignment null.

## Limitations

Single MSFT day; 2003 market structure; historically inferred Island/INET venue lineage; transport completeness unverified; visible book only; hidden liquidity and hidden queues absent despite observed P executions; unverified historical tick size; no independent exchange snapshot; regular-session proxy without halt/auction certification; millisecond timestamps without receive time. There is no queue-position model, no execution costs, no latency model, no cross-stock replication, no cross-day replication, no modern-market validation and no strategy PnL.

Within-day bootstrap and circular-shift evidence rely on approximate temporal regularity and cannot create independent days. Nonstationarity, zero-heavy returns/OFI, carried source-state features, finite permutation resolution and dependence between hypotheses limit interpretation. BY robustness does not cure null misspecification. No whole-day quantile boundary is an online decision rule. Raw redistribution rights remain unestablished; raw data, row-level grid/traces and manual records stay excluded from Git.

## Engineering provenance and stopping point

Measured single numerical study runtime: 84.585s; peak working set 259,321,856 bytes. `PHASE3_MANIFEST.json` records grid/CSV hashes, package versions, seeds and the original executed code digest. `requirements.phase3.txt` pins this environment without changing upstream requirements.

One presentation-only defect occurred after all numerical outputs, manifest and complete status were saved: the final console JSON print could not serialize a NumPy boolean. That print was amended; no analytic module or saved result changed, and the study was not rerun. Original runner bytes, original/current code digests and the exact console-only change are independently verified in `console_fix_audit.json`. This is not a protocol or statistical-method amendment.

Methodology gates: frozen upstream/protocol/features; primary 1s clock inference; 60s bootstrap plus 30/300s robustness; circular-shift null and BH/BY; all horizons/time/state diagnostics; random/placebo/sign controls; independent scalar checks and reproducible report; full tests; zero models and zero execution simulations. **PHASE 3: PASS** means the methodology gate is satisfied, not that tradable alpha is established. Phase 4 has not begun. Stop and await the next instruction.

PHASE 3: PASS
