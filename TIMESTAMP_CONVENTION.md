# Time and ordering contract

The [ITCH 2.0 specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/tv-itch2.pdf) identifies milliseconds after midnight Eastern Time. The filename `S010303-v2` and ZIP-member date `(2003,1,3)` support **2003-01-03**. Date is external metadata, not embedded per message. Do not infer dates from current local time.

Use IANA `America/New_York`, not a fixed EST offset. The selected date is UTC-05:00; 09:30 local is 14:30 UTC. Summer tests verify 09:30 in June is 13:30 UTC. The Windows host is in China Standard Time; that setting has no effect on source conversion. User-facing run dates use Asia/Shanghai; manifests use explicit UTC.

Parse the eight-byte integer exactly. Compute the source day's aware local midnight, convert it to UTC, and add integer milliseconds. Parquet stores UTC nanoseconds to avoid rounding and ease future adapters; the extra digits **do not imply nanosecond information**. Every source timestamp is representable exactly. No float parsing, naive research timestamps, or date-dependent local-host conversion is used.

Physical source row order is authoritative. The file does not contain transport sequence numbers. Order IDs and match IDs are identifiers, not gap-free event sequences. Maintain `source_row` and `source_time_ms`; never sort merely by timestamp. Equal timestamps remain separate observations in source order. The ordering validator counts duplicate times and clock regressions and rejects missing/reordered physical rows. A clock regression is reported without reordering; its presence must block a later time-based research gate until resolved.

The report distinguishes raw-file duplicate timestamps from MSFT-window duplicates. Duplicate count means a row shares a timestamp with its immediately preceding row in that population; it is not a duplicate-message count. Timestamp ties cannot reveal sub-millisecond latency or independent within-millisecond arrival times. Source order is deterministic for replay but is not a claimed receipt timestamp.

## Session filter

Configuration defaults to the half-open interval **[09:30:00,16:00:00)** in source local time. This is a time-of-day regular-session proxy, not a verified exchange-state classification. Events at 16:00 are excluded. Full source replay begins at the explicit start-of-day around 07:00 and continues through the end and trailing cancellations around 20:00. Premarket orders must remain in the live book when the output window opens. Filtering input before reconstruction is forbidden.

Premarket, regular-window and after-hours counts are reported separately. Start/end-of-day controls are broader than the regular equity session. This legacy protocol has no separate auction or halt indicator. We do not label absent auction messages as proof that no exceptional trading state occurred. Near-open/close observations can be excluded by narrowing the configured window in a separately recorded run; no such discretionary cleaning is hidden here.

The date-specific configuration applies only to this sample. No exchange holiday/early-close calendar is bundled and no generic full-day claim is made. For another date, verify its schedule and auction conventions first. [Nasdaq's trading-hours overview](https://www.nasdaq.com/stock-market-trading-hours-for-nasdaq) provides general context, not historical-day proof.

Day transitions require a new dated input. Values outside [0,86,400,000) fail. Days with a daylight-saving offset change are explicitly rejected by this adapter rather than ambiguously interpreting repeated/nonexistent wall times; normal US equity trading dates do not include the Sunday DST transitions. Tests cover winter, summer, date rollover, precision, session boundaries and rejection of DST-transition days.

## Quote/trade ordering

For source event r, the pre-trade book is the selected instrument's state after its preceding event; the post-trade book includes r. An E execution both reports a trade and decrements the referenced displayed order. It is not two independent events. A P hidden trade leaves the displayed book unchanged. Two messages sharing a millisecond remain causally distinct by source row.

Trades store pre/post source-row references and BBO values, so a prequote immediately before the output window is still interpretable even if its row is not in the retained Parquet window. Never timestamp-join a trade to an arbitrary last row at its timestamp: that can select later same-millisecond events. No impact or future-return labels are computed.
