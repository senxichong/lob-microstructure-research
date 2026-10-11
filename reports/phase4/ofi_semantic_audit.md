# OFI semantic audit — before model fitting

PASS. Read frozen phase2/order_flow.py, features.py, common.py and existing
definitions. OFI is a consecutive visible BBO increment, not a signed trade.
Source B/S identify resting bid/ask; E removes visible resting depth, while P
is a hidden print and does not itself change visible queues. No aggressor inference.

Formula: I(b>=b_prev) q_bid - I(b<=b_prev) q_bid_prev
- I(a<=a_prev) q_ask + I(a>=a_prev) q_ask_prev.
At unchanged prices, bid additions positive, bid removals negative; ask
additions negative, ask removals positive. A bid improvement contributes new
bid size; worsening subtracts old size. Ask improvement subtracts new ask size;
worsening adds old size. First/invalid transitions are missing.

100ms and 1s OFI sum increments in (t-w,t], capped at the current source order
for same-time ties. Full observed history and valid book required. Phase 3
carries these source-event features to grid time, without refreshing stale
windows. This age convention may matter especially at 100ms.

Executed existing tests: `pytest -q tests/test_phase2_features.py -k 'ofi or
window_left'`: **9 passed, 10 deselected**, including eight hand-calculated
price/size signs and exact exclusive-left/inclusive-source rolling endpoints.
No verified defect found. Upstream definitions/code remain unchanged.
Phase 3 negative OFI_100MS and positive OFI_1S association are retained as facts.
Aggregation horizon, quote reactions, sign/dynamics interaction and sampling
noise are possible explanations; none is established by this audit.
