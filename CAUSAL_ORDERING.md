# Causal ordering

State identity: `(session_date, symbol, timestamp UTC, source_order)`, where source_order is the unchanged Phase 1 physical source_row. `event_index` is zero-based position in the retained session. Time is nondecreasing and source_order strictly increases; gaps are expected because other instruments and off-session rows are omitted. Unordered input is rejected, not sorted by timestamp. One-symbol/one-session input is explicitly enforced by this adapter.

At `(t,k)` the feature is observable immediately after applying source row k. It may use the current book/event and preceding rows only. Later `(t,k+1)` is forbidden despite identical millisecond time. Millisecond **precision** does not replace true source **ordering**. The lack of transport sequence numbers prevents definitive proof of feed completeness.

Rolling windows use `time > t-w` and row index <= current index. A historical as-of cutoff has an inclusive timestamp upper bound, with last source row breaking ties; w is strictly positive so historical lookup cannot select a later current-time event. Future labels use `(t+h,+infinity)` as their lookup bound, so all observed rows exactly at the horizon are included. This label is not a feature.

Partition before all lookups. The current adapter rejects multi-session/multi-symbol inputs rather than letting searchsorted cross a boundary. UTC arithmetic is integer nanoseconds; input retains 1 ms source precision. No interpolation, nearest-neighbor matching or timestamp-only deduplication is used.

Mutation tests distinguish a complete timestamp cutoff from a source-row cutoff inside a tie. A horizon ending at an incompletely observed tie group is allowed to change when later rows of that group change; features through the source-row cutoff must not change.
