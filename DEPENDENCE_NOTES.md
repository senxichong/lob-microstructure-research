# Dependence and later inference

Retained observations are event-time clustered, while labels have fixed wall-clock horizons. Adjacent samples one millisecond apart with 1-second labels share almost all their forward interval. Timestamp ties can share exactly the same target but have different current states. Unchanged-book events remain observations. Consequently rows, labels and features are not iid.

Book/flow persistence, repeated executions of one order, bursts of activity and microstructure autocorrelation can materially reduce effective sample size. Event-weighted summaries overweight busy periods relative to time-weighted summaries; neither is independent evidence of prediction. Horizon class summaries describe different targets and must not be ranked as predictive performance.

Later research will need chronological/blocked validation, overlap-aware purging or embargo, uncertainty methods respecting serial dependence, and train-only normalization. Single-day data cannot supply convincing out-of-time/regime validation. No split optimization, model selection or predictive test is performed here.
