# Labels — version 2.0.0

For row i with key (t,k), define tau=t+h, h in {100,1000,5000,10000} milliseconds. Let j be the **last** row with timestamp <=tau within the same allowed session, resolving equal timestamps by the highest source_order. The right-continuous book is piecewise constant. No update between t and tau means j can equal i and the target can be available and FLAT. Do not choose the first row after tau.

Availability requires current row valid, tau strictly before the half-open session close, j>=i, and no observed invalid book row from i through j inclusive. An invalid interval blocks carry-forward even if a later state recovers. If the last state at tau is invalid, do not skip it to revive an older quote. No next-session state or post-close forward fill is allowed.

Let s_i=bid_price_units_1+ask_price_units_1 (twice midprice in USD/10,000 units). `FWD_MID_RET_h = s_j/s_i - 1`; `MID_MOVE_h = (s_j-s_i)/20000` USD. `DIRECTION_h` is UP/FLAT/DOWN by exact integer comparison of s_j and s_i. No epsilon or label balancing; FLAT rows remain. `MID_MOVE_TICKS_h=MID_MOVE_h/tick_size_usd` only if a verified tick exists; current tick-dependent columns are entirely null.

Each horizon records `target_time_requested_h`, `target_state_timestamp_h`, `target_state_source_order_h`, `target_age_ms_h=(tau-time_j)/1e6`, `target_midprice_h`, `target_available_h`, and `target_unavailable_reason_h`. Requested times remain populated even when targets are unavailable. Resolved state fields and labels are null for unavailable targets. Age must be nonnegative; it describes the age of the selected retained observation, not specifically time since a price change.

The primary horizon is 1S. Suffixes are 100MS, 1S, 5S, 10S for numerical labels, and 100ms, 1s, 5s, 10s for target metadata/flags. Feature existence and feature flags never depend on any target. End-of-session rows remain in the matrix with unavailable targets. Summary class proportions use available targets as denominator, and unavailable counts are reported separately. No feature–label statistic is calculated.
