# Phase 2 preregistration — version 2.0.0

Frozen on 2026-10-11 (Asia/Shanghai), before feature/label summaries. No predictive comparison, model, normalization fitted across the day, class balancing, threshold optimization, future-move sample selection or execution assumptions are permitted. Stop after Phase 2.

Primary feature timestamp: **post-event reconstructed book state**. Primary sample: every valid retained observation, including unchanged-book trade states and FLAT targets. Canonical output preserves even unavailable/invalid diagnostic rows with flags, rather than conditioning the population on future availability.

Primary horizon: **1,000 ms**. Secondary horizons: **100, 5,000, 10,000 ms**. Historical windows: **100, 1,000, 5,000, 10,000 ms**. OBI depths: **1, 3, 5, 10**. WOBI weights: **1/k**, fixed without optimization. Primary price: **midprice**. Primary labels: wall-clock last-state-as-of, never the next event after the requested horizon. No event-time targets.

The inherited session is **[09:30,16:00) America/New_York, 2003-01-03**, a regular-hours time-of-day proxy, not independently verified continuous/auction/halt status. No premarket rolling history or cross-session borrowing. Reconstruction's premarket carry-in remains part of each Phase 1 book, but is not a Phase 2 event-history observation.

Window history is `(t-w,t]` with the right endpoint capped at the current source row, even within a timestamp tie. Require cutoff `t-w >= first retained timestamp` and a valid uninterrupted book segment; otherwise all features in that rolling group are null, not zero. This deliberately excludes the first undefined OFI increment from any available window. Historical point lookups choose the last state at or before `t-w`. Partial depth is not invented; a K-level feature requires K valid levels per side.

For label target tau=t+h, choose the last source row with time <=tau in the same session (all events exactly at tau are included). No update is needed near tau; a valid last state persists. Reject a label if current state is invalid, an intervening invalid state is observed, or **tau >= session end**. Equality is unavailable because the inherited session is half-open. A recovered valid state can start a new valid segment; do not bridge an invalid interval. No arbitrary staleness threshold.

Tick size is **unknown**, as established in Phase 1. All requested tick-denominated features and labels are present but null, with `tick_size_available=false`; no observed GCD or encoding quantum is substituted. Direction compares exact integer bid+ask sums, without epsilon. USD midprice has an exact half-encoding-unit representation (sum / 20,000).

Execution activity uses E+P, with separate visible E and hidden P diagnostics. Bid/ask-side names refer to executed resting orders, never aggressors. B is counted as an event but does not retroactively erase an execution; the real input has no B. Time-since-previous trade/price-change is strictly prior to the current event, not including it.

Future-mutation checks freeze every feature through event key T. Labels remain fixed when their resolved as-of event key is <=T. For a cutoff ending a complete timestamp group this includes every requested target time <=T.time. If T is inside a tie group, a target exactly at T.time includes the group's later rows and is not yet causally settled; it must not be falsely declared invariant. This distinction follows the frozen wall-clock label definition.

Phase 1 raw, reconstructed artifacts, source code, tests and reports remain untouched. `config.phase1.yaml` preserves original config bytes; `config.yaml` only gains a separate phase2 mapping. `reports/phase2/upstream_freeze.json` records original hashes and the narrow authorized config/README/gitignore edits. No upstream defect has been identified or repaired.

Final output: immutable versioned Parquet, feature metadata, summaries, >=20 real traces, >=5 independent spot checks, full tests and independent verification. Protocol/config/code/input hashes are recorded. Re-running with different definitions must use a new version/output location, not overwrite the frozen matrix.
