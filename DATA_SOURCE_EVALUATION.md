# Data-source feasibility decision

Evaluated 2026-10-10 (Asia/Shanghai). Access claims below distinguish downloaded bytes from advertised availability. Repeat with `python scripts/probe_sources.py`; actual HTTP responses are saved in `reports/source_access_checks.json`. A successful web page is not proof of accessible data.

## Decision

Select **Nasdaq-hosted `S010303-v2.zip`, legacy ITCH 2.0, 2003-01-03**, and reconstruct MSFT. The complete archive was downloaded, hashed and CRC-checked by reading the ZIP member through EOF. It has an explicit start-of-day and order IDs, allowing an empty initial book rather than an invented snapshot seed. One stock is reconstructed; all symbols' active references are tracked to distinguish foreign events from missing orders.

US-equity historical order messages **are obtainable without a paid login for this sample**. This does not establish free access to an arbitrary modern date range. No asset-class substitution was made. Modern Nasdaq ITCH 5.0 is preferable for later contemporary research, but requires substantially larger ingestion and separate validation. The old sample is selected for documented semantics, first-party provenance, and complete-day reconstruction, not merely download convenience.

## Market and data matrix

| Candidate | Market / venue / asset class | Instruments / dates / historical coverage | Depth and observation type | Clock, trades, IDs and sequence |
|---|---|---|---|---|
| **Selected: Nasdaq-hosted ITCH 2.0 sample** | US equities and ETFs; historical Island/INET lineage, hosted by Nasdaq. Not a 2003 consolidated Nasdaq book. Exact originating venue inferred from protocol history, not a venue field. | Raw file has 6,116 symbols with adds, including test symbols; MSFT selected. One day: filename and ZIP timestamp identify 2003-01-03. | Full visible order messages for the represented feed; reconstructed variable depth, save up to 10 levels per side. A/X/E/P/B/S. | 1 ms Eastern; E visible and P hidden executions; order and match IDs. No transport sequence in the text file; physical row order is authoritative. |
| Nasdaq ITCH 5.0 public samples | US equities / Nasdaq single venue | Multiple dated files listed; concrete candidate `12302019.NASDAQ_ITCH50.gz`, 2019-12-30. Multi-stock, count not inspected. Arbitrary dates not promised free. | Full displayed order events; adds, executions, cancel/delete/replace, hidden/cross trades and controls in the specification. | Nanosecond encoding; order and match IDs. Tracking number is not a transport sequence. File framing and session state would need a dedicated parser. |
| LOBSTER official sample / subscription | US equities / Nasdaq | Legacy sample listing: AAPL, AMZN, GOOG, INTC, MSFT, 2012-06-21; additional SPY high-depth sample. Subscription date coverage must be verified with vendor. | Paired event-indexed L1/L5/L10 snapshots and filtered message annotations; not an independently complete top-N order ledger. | Legacy sample advertised with fine fractional-second times; exact precision not measured because no official ZIP obtained. Order IDs; no transport sequence in published CSV schema. Visible/hidden executions. |
| FI-2010 academic benchmark | Finnish equities / Nasdaq Nordic Helsinki | Five stocks, ten days, June 2010 (authors' benchmark); exact downloadable package contents not inspected here. | Ten levels, transformed benchmark matrices with normalization and precomputed targets; unsuitable as primary event-reconstruction ground truth. | Raw time, order IDs, sequence and separate trade semantics not established from fetched catalog. No claim of timestamp precision or complete messages. |
| Tardis crypto archive (fallback only) | Crypto / Binance spot, BTCUSDT probe | Concrete probe 2020-01-01; broad exchange/symbol coverage vendor-dependent. | Snapshot-5 probe; vendor also offers incremental L2, top-25 snapshots and trades as separate datasets. Not L3 on Binance. | Vendor CSV schema uses microsecond epoch encoding, actual exchange precision varies. Capture order; snapshot CSV has no order IDs. No order lifecycle inference from snapshots. |

## Access, cost and reproducibility

| Candidate | Tested method / result | Cost and limits | Licensing / reproducibility / caveats |
|---|---|---|---|
| ITCH 2.0 | Complete HTTPS ZIP, 58,907,174 bytes; immutable local cache and pinned SHA-256. Member 212,330,490 bytes. | No payment/login for this URL; no published quota established; single cached request with retry backoff. | No open redistribution grant established. Keep raw, normalized rows and real traces local. Strong byte-level reproducibility while URL remains live; very old market regime. |
| ITCH 5.0 | First bytes probed, not a full download or parser validation; listing gives 3,524,013,057 compressed bytes for 2019-12-30. | Public sample URL; broader historical service is commercial, price not verified. Rate quota not established. | Better contemporary semantics, greater engineering/disk cost. No redistribution permission assumed. |
| LOBSTER | Legacy official host failed TLS in this environment. Main-domain legacy ZIP path returned HTML, not ZIP. New official application has gated book-sample requests. | Former public samples advertised free; current book sample requires purchase proof and approval. Subscription pricing not verified; no rate quota verified. | No unofficial mirror substituted. Availability changed; search indexing is insufficient. Restrictive terms; no GitHub data redistribution assumed. |
| FI-2010 | Official Fairdata catalog HTTP 200; JavaScript page, not a data artifact. Full archive not obtained. | Advertised public benchmark; no payment checked. Package URL and rate limits unresolved. | Authors' paper verified. License of exact current downloadable package not verified; do not repeat third-party CC claims as fact. Precomputed labels/normalization outside Phase 1. |
| Tardis | Small streaming GET of explicit first-of-month snapshot URL; see machine-readable probe for actual status. Not selected or fully ingested. | Vendor documents first day of each month without API key; broader access paid. Limits/prices not established. | Conditional fallback; vendor redistribution terms not verified. Crypto microstructure does not directly transfer to US equities. |

## Suitability assessment

Scores are engineering judgments, 1 (weak) to 5 (strong); they are not statistical results. Execution suitability is relative and does not imply executable alpha.

| Source | Microstructure research | Reconstruction | Execution research | Reproducible access here | Cost suitability |
|---|---:|---:|---:|---:|---:|
| ITCH 2.0 sample | 4 for method validation, 1 for modern generalization | 5 | 2 | 5 for pinned sample | 5 |
| ITCH 5.0 samples | 5 | 5 after implementation | 3 | 3: prefix only tested | 4: bandwidth cost |
| LOBSTER | 4 | 2 from filtered top-N messages | 2 | 1: official ZIP failed | 2: current access unclear |
| FI-2010 | 2 for this phase | 1 | 1 | 2: catalog only | 4: package unverified |
| Tardis Binance | 3 for crypto methodology | 3 for L2, 1 for snapshot-only | 2 | 3: probe only | 4 for sample days |

## Primary evidence

- [Nasdaq sample directory](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/) and [selected ZIP](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/S010303-v2.zip).
- [Nasdaq ITCH 2.0 specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/tv-itch2.pdf), including its revision history; locally hashed in the manifest.
- [Nasdaq ITCH 5.0 specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf).
- [LOBSTER legacy sample listing](https://data.lobsterdata.com/info/DataSamples.php), [output schema](https://data.lobsterdata.com/info/DataStructure.php), [current book-sample route](https://lobsterdata.com/book-sample), [current trial/sample terms](https://lobsterdata.com/terms/trial-and-sample-data). The current application's public JavaScript was inspected to verify the gate; no request was submitted.
- [FI-2010 authors' paper](https://arxiv.org/abs/1705.03233), [Fairdata catalog](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649).
- [Tardis CSV availability](https://docs.tardis.dev/downloadable-csv-files/overview), [Tardis data semantics](https://docs.tardis.dev/faq/data).
