# Exact source schema and normalized representation

The primary artifact is **event data**, not snapshots. `S010303-v2.txt` contains fixed-width ASCII ITCH 2.0 records terminated by LF, plus one terminal empty line. That framing line is counted and accepted only at EOF. A blank in the middle, unknown message, incorrect record length or malformed number aborts ingestion and is logged.

Offsets below are zero-based, lengths exclude LF. Numerics are space-padded on the left; alpha fields are padded on the right. These layouts follow pages 2-5 of the [official ITCH 2.0 specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/tv-itch2.pdf). All observed lengths match those layouts.

## Source fields

| Field | Types | Offset:length | Internal type | Units / semantics / missing |
|---|---|---|---|---|
| Time Stamp | all | 0:8 | int64 | Milliseconds after Eastern local midnight; date external to records; no missing value permitted. |
| Message Type | all | 8:1 | string | S, A, E, X, P, B; no missing. |
| Event Code | S | 9:1 | string | S=start of day, E=end of day. |
| Order Reference Number | A/E/X/P | 9:9 | int64 | Order identifier; A identifiers increase but need not be consecutive. Never an event sequence. |
| Buy/Sell Indicator | A/P | 18:1 | string | B=buy resting order, S=sell resting order. Not an aggressor flag. |
| Shares | A/P | 19:6 | int64 | Shares added / shares executed, not lots. |
| Stock | A/P | 25:6 | string | Source ticker, right spaces removed; no ticker identity mapping inferred. |
| Price | A/P | 31:10 | int64 | USD multiplied by 10,000; six integer positions followed by four decimal positions, no decimal point. |
| Display | A | 41:1 | string | Y=displayed; historical S is acknowledged by the later spec but insufficiently defined there. Selected-symbol S raises an explicit error; every observed A is Y. |
| Executed Shares | E | 18:6 | int64 | Positive decrement to referenced live order; price/symbol/side inherited from its A. |
| Canceled Shares | X | 18:6 | int64 | Positive decrement, partial cancel or full delete when remainder is zero. |
| Match Number | E | 24:9 | int64 | Execution identifier. |
| Match Number | P | 41:9 | int64 | Hidden-trade identifier. |
| Match Number | B | 9:9 | int64 | Prior E/P being broken; no book restoration. |

Required fields have no null sentinel. Fields absent for a message type become null in normalized output; they are never filled with zero. Record lengths: S=10, A=42, E=33, X=24, P=50, B=18 bytes. There is no standalone DELETE, REPLACE, halt or auction message in this version. Do not rename X to a modern ITCH X-only partial cancellation: it can remove the entire order. Do not invent U or D semantics.

## Event effects

- A: establish a unique order and add its visible quantity to its side/price aggregate.
- X: reduce outstanding quantity; remove the order and empty aggregate level at zero.
- E: same quantity reduction, plus a trade at the original limit price.
- P: hidden-order execution; publish a trade but leave visible depth unchanged.
- B: publish a correction linked to its match; no visible-book change and no retrospective deletion of the original trade row.
- S: control state. Require start-of-day before orders. At end-of-day prohibit new adds but retain the ledger for trailing cancellations. The specification's page-2 purge language conflicts with its revision-history explanation; retaining until explicit X messages supports both trailing-cancel processing and auditability. No post-close quotes are retained for research.

Integer arithmetic controls every book mutation. Quantity below zero, unknown ID, duplicate add, nonincreasing A ID or unsupported event raises a logged exception before a new book state is emitted. Premarket orders seed the selected instrument from an empty day-start book. A parallel market-wide active-ID registry allows rigorous routing of E/X messages without guessing symbols.

## Scaling and tick diagnostics

The source layout fixes prices at USD/10,000 and quantity in individual shares. For example, a price field of `911900` represents $91.19 and a shares field of `100` represents 100 shares; the parser's hand-calculated tests check this conversion. Real event traces cross-check source fields against resulting depth changes. No splits or inflation adjustments are applied to historical prices.

**Encoding precision is not the legal tick size.** The file has no tick-rule field. We check positive integer units and measure the GCD of MSFT add prices and penny-grid deviations. We do not apply today's penny rule to 2003: [Island's contemporary SEC comment](https://www.sec.gov/rules/concept/s71401/smith1.htm) describes subpenny trading. The exact historical instrument minimum remains unverified. `price_grid_units: 1` means the representable $0.0001 grid, not a claimed venue rule.

## Canonical Parquet tables

All tables use UTC `timestamp[ns]` (source accuracy remains 1 ms), `source_time_ms:int64`, `source_row:int64`, and `event_type:string`. The physical row number is provenance, not an exchange sequence number. Schema metadata records raw SHA-256, selected ZIP member, config hash, symbol and date; `lineage.json` additionally records member and output hashes.

`events.parquet`: selected-window A/X/E/P/B events; order_id, match_id, size and price_units are nullable int64; price is float64 USD; side, display and symbol are nullable strings. For E/X, price/side/symbol/display are deterministically recovered from the referenced A, not independently transmitted fields. Size is this event's quantity change, not remaining order size. B has null order fields.

`books.parquet`: post-event state for every retained selected event, including unchanged P/B states. Ten nullable price/size slots per side: `bid_price_units_1..10`, `bid_price_1..10`, `bid_size_1..10`, and corresponding ask fields. Prices use int64 units plus float64 USD; sizes are int64 shares. Absent levels are null, never dummy prices or zero shares. Actual total occupied levels are recorded separately. The source does not promise exactly ten occupied levels.

Best bid/ask fields repeat level 1. Midprice=(bid+ask)/2; spread=ask-bid; relative_spread=spread/mid; microprice=(ask*bid_size+bid*ask_size)/(bid_size+ask_size). These float64 fields are diagnostic only. Missing sides yield null diagnostics. `issues:list[string]` preserves invalid states; `research_eligible:bool` requires no listed issue. Crossed/locked rows remain stored.

`trades.parquet`: E and P events, with original match and order IDs, shares and price, executed resting side, source-row links and pre/post BBO in integer units. `aggressor_side` remains null because no aggressor field exists; `aggressor_basis` explains this. No quote/tick classification is implemented.

`trade_breaks.parquet`: append-only B events linking `match_id` to `original_trade_source_row`. Corrections must be applied as of their arrival time in later work; future corrections must not alter what was known earlier. This sample has no observed B messages; behavior is synthetically tested.

Machine-readable authoritative output types are defined in `src/lob_research/storage.py`. Empty trade/correction tables retain their schema. Per-row source market data and traces remain local-only.
