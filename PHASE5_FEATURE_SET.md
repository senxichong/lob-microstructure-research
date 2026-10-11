# Phase 5 frozen input — 5.0.0

Exactly Phase 4 FULL features, in order:

1. OBI_L1 — (bid size minus ask size)/(sum), top visible level.
2. MICROPRICE_MINUS_MID — opposite-size-weighted BBO price minus mid, USD.
3. OFI_100MS — frozen consecutive BBO increments summed over (t-100ms,t].
4. OFI_1S — same source convention, (t-1s,t].
5. RELATIVE_SPREAD — spread/mid.
6. EVENT_RATE_1S — source-event count over 1s divided by one second.
7. TIME_SINCE_PREVIOUS_TRADE — milliseconds since strictly previous trade.

Authoritative definitions: FEATURE_DEFINITIONS.md, CAUSAL_ORDERING.md,
PHASE3_PROTOCOL.md, PHASE4_FEATURE_SET.md and frozen Phase 2 feature code.
Grid features carry the last causal source state, including stale rolling/timing
values, and are not recalculated at clock time. Labels reanchor at grid time.
No OFI sign changes: 100ms was negative, 1s positive in prior results.
Static=[OBI_L1,MICROPRICE_MINUS_MID,RELATIVE_SPREAD].
Flow=[OFI_100MS,OFI_1S,EVENT_RATE_1S,TIME_SINCE_PREVIOUS_TRADE].
Other depth variants and all other Phase 2 columns excluded. Expanded features,
interactions as new inputs, alternate horizons and balanced-class experiments omitted.
RANDOM_CONTROL may appear only in the explicitly named random-feature diagnostic.
FUTURE_RETURN_1S exists only in a rejection test and is never permitted in a fit.
