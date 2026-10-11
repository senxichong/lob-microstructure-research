# Features — version 2.0.0

**Feature(t,k) is computed AFTER applying event k.** It may use the current post-event book, current event and prior rows. It may not see later source orders at the same timestamp, later trades/books or labels. Every numeric feature's name, formula, required fields, lookback, timestamp convention, unit and missing rule are also written to `reports/phase2/feature_metadata.json` by the engine.

## Static book

Prices use USD, quantities individual shares. MIDPRICE=(bid+ask)/2, SPREAD=ask-bid, RELATIVE_SPREAD=spread/mid. MICROPRICE=(ask*bid_qty+bid*ask_qty)/(bid_qty+ask_qty); MICROPRICE_MINUS_MID is its USD deviation. MICROPRICE_DEVIATION_TICKS is null because verified tick size is unavailable. A known tick is supported for synthetic tests only, not substituted into real data.

For K in 1,3,5,10: BID_DEPTH_LK=sum bid sizes through K; ASK_DEPTH_LK analogous; TOTAL_DEPTH_LK is their sum; OBI_LK=(bid_depth-ask_depth)/total. DEPTH_RATIO_LK=bid_depth/ask_depth (unbounded ratio, denominator must be positive). The extra L3 depth fields make the corresponding OBI auditable. WOBI_L10 uses weights 1/k in both numerator and denominator. K valid levels per side are required; zero denominators yield null, not infinity. No clipping. Depth ratios and OBI are redundant transforms, not independent signals: OBI=(ratio-1)/(ratio+1).

`book_valid` requires a Phase 1 eligible state and valid ordered positive top-of-book. Each depth-specific feature also validates all requested levels. `feature_static_available` requires all configured depths, including ten; it does not require tick knowledge. `tick_size_available` is separate.

## OFI

OFI_EVENT=e_i uses consecutive retained post-event BBOs:

`1[b_i>=b_prev]*q_bi - 1[b_i<=b_prev]*q_bprev - 1[a_i<=a_prev]*q_ai + 1[a_i>=a_prev]*q_aprev`.

Thus equal bid price gives current-minus-previous quantity; equal ask price gives previous-minus-current quantity. Bid improvement contributes current bid size, bid deterioration minus prior bid size; ask improvement contributes minus current ask size, ask deterioration prior ask size. Unchanged-book P observations contribute zero. First row or a transition touching an invalid state yields null. Shares are the unit. OFI_100MS/1S/5S/10S are sums over `(t-w,t]` capped at current source order.

## Activity and rates

For each window: EVENT_COUNT; ADD_COUNT/SIZE (A); CANCEL_COUNT/SIZE (X, including full deletion); EXECUTE_COUNT/SIZE (E+P). For each A/X/E+P family also BID_SIDE_COUNT/SIZE and ASK_SIDE_COUNT/SIZE; these are resting sides. VISIBLE_EXECUTE_COUNT/SIZE (E) and HIDDEN_EXECUTE_COUNT/SIZE (P) preserve the distinction. EVENT_RATE, ADD_RATE, CANCEL_RATE and EXECUTION_RATE divide corresponding count by the complete window's seconds, not observed active time. Counts are events; sizes shares; rates events/second. B would count in EVENT_COUNT only, with no retrospective trade correction.

## Availability and history

All 100ms/1s/5s/10s rolling groups require a full observed history since the first retained session row and no invalid interval. Until then they are null and `feature_<window>_available=false`. True zero means a fully covered window with no event of that family. Current event is included; events exactly at the left boundary and later rows at the right-boundary timestamp are excluded. No premarket or prior-session rolling observations are used.

TIME_SINCE_PREVIOUS_EVENT, TIME_SINCE_PREVIOUS_TRADE (E/P), TIME_SINCE_PREVIOUS_BOOK_PRICE_CHANGE (either BBO price changes) are milliseconds to the strictly prior qualifying event. Current trade/price-change is not its own previous event. If none exists in the current uninterrupted valid segment, result is null. Same-timestamp prior events can yield zero. No time-to-next fields.

MID_RETURN_100MS/1S/5S/10S = current_mid / last_mid_at_or_before(t-w) - 1. SPREAD_CHANGE_1S and OBI_CHANGE_1S use current minus historical as-of spread and OBI_L1. All require available history and an uninterrupted valid segment including the historical state. No next-state lookup, interpolation, full-day normalization or fitted transform.
