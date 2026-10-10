# Preregistered research protocol

Phase 1 establishes trustworthy data only. No predictive results, model fitting, label construction, backtesting, execution simulation or cost calculations are authorized in this phase.

1. A feature at time t may use only information available by t, with source-row ordering resolving equal timestamps.
2. Preserve exchange/source sequencing semantics. IDs are not substitutes for event sequence numbers. Clock regressions and gaps must be investigated, not silently sorted away.
3. Future observations may construct labels in a later phase, never features. Store feature and label code separately when authorized.
4. Resampling must be backward-looking; define inclusive/exclusive boundaries and avoid selecting later events within timestamp ties.
5. Fit normalization, imputation, clipping thresholds and any learned transforms only on the training period. Do not reuse FI-2010-style full-sample normalized matrices without auditing their provenance.
6. Use chronological train/validation/test splits and a declared embargo/purge appropriate to overlapping horizons.
7. Keep the final test period out of model selection, feature selection, threshold tuning and repeated exploratory decisions.
8. Spread, fees, rebates, slippage, latency and adverse selection must be addressed before claiming tradability, in a separately authorized later phase.
9. Record failed experiments, hypotheses, configurations, versions, data hashes and exclusions; never retain only favorable runs.
10. Raw market data are immutable. Corrections and exclusion masks are derived, timestamped and reversible; broken-trade knowledge is only available when its B message arrives.
11. Respect source licensing. No public redistribution of source rows, reconstructed book rows or real event traces without established permission.
12. Predictive accuracy is not equivalent to executable alpha. Displayed queues do not reveal fills, hidden liquidity, our queue priority or our network latency.
13. Freeze dataset and diagnostics decisions before testing hypotheses. Do not choose instruments/days based on desired trading returns.
14. Separate measured evidence from assumptions. Single-day legacy data can validate methodology but cannot establish modern US-equity generalization.
15. Do not proceed automatically beyond Phase 1 even if it passes. A failed integrity gate must not be bypassed by tuning or deleting problematic rows.
