# Scope and unresolved data limitations

## Market and provenance

The selected artifact is the 2003-01-03 ITCH 2.0 sample hosted by Nasdaq. Its protocol revision history refers to Island before 2004, INET thereafter, and Nasdaq after 2006. Nasdaq's [2005 acquisition announcement](https://ir.nasdaq.com/news-releases/news-release-details/nasdaq-acquire-instinet) corroborates that the 2003 venue must not be presented as today's Nasdaq matching engine. We describe it as **legacy Island/INET lineage**. There is no MIC/venue/date field in each event; venue attribution remains a documented historical inference. The date is corroborated by filename and archive-member metadata.

The complete feed contains many symbols, including test symbols, but only MSFT is reconstructed into research tables. One stock/day is adequate for code and unit diagnostics, not for statistical generalization, seasonality, regime robustness or modern execution research. Unadjusted 2003 prices must not be compared directly with today's split-adjusted prices. No current-data representativeness is claimed.

## Coverage and reconstruction

Only this feed's displayed orders are reconstructible. This is neither a consolidated NBBO nor all US liquidity. Full visible order IDs exist; the engine tracks outstanding orders and aggregates by price, saving up to ten occupied levels. It does not infer missing liquidity, participant attribution, venue routing, dark-pool interest, off-exchange prints, iceberg reserve sizes or other ECN books. Hidden executions are visible as P prints, but hidden orders are not added to the displayed book.

Prices/quantities are verified against official field definitions and real replay, but there is no independent exchange snapshot in the ZIP. Order-level versus price-level aggregate audits and end-of-day exhaustion are strong consistency checks, not independent ground-truth certification. Deterministic replay does not prove that the publisher's input feed omitted no messages.

## Time, sequence and sessions

Source precision is 1 ms. Multiple events share that time; file row order is retained. No original transport sequence or local receive timestamp survives. Missing/reordered upstream messages cannot be exhaustively detected, even with no unknown references. Order-reference gaps are normal. There is no estimate of clock error, network latency, packet loss, retransmission timing or our receipt time.

The configured 09:30–16:00 interval is a regular-hours proxy. ITCH 2.0 lacks explicit halt and cross/auction states. The book is replayed through all available source hours to preserve state; auctions are not intentionally mixed or modeled, but absence of an auction flag prevents complete state certification. A modern adapter with trading-action/cross controls is required before any claim that every observation is continuous trading.

## Trading interpretation

E/P side is the executed order's resting side. No aggressor column is directly transmitted; normalized aggressor remains null. No Lee–Ready, tick-rule or quote-test inference is performed. Multiple E/P prints may represent one larger interaction; no grouping is invented. B corrections are supported and tested but none occur in this sample. Like the other normalized outputs, the correction table uses the configured event-time window; an application needing after-hours corrections to regular-session trades must extend that export policy before using another dataset.

The displayed order ledger supports quantity/priority diagnostics, not our actual queue position or a fill model. Time priority can be changed by venue rules and hidden/reserve mechanics not represented here. Fees, rebates, trade-through constraints, broker routing and latency are unmeasured. No execution simulation or cost estimate has been made.

## Tick, locked/crossed states and licensing

The $0.0001 encoding unit is known; the legally permitted minimum for this instrument/venue/day is not supplied. Subpenny observations in the pre-Regulation-NMS era are not automatically bad data. Report empirical grid/GCD statistics rather than imposing today's rules.

For a continuous single matching book, locked/crossed states warrant investigation. They are not automatically classified as corrupt: possible sequencing and unreported exceptional states must be considered. Preserve them, attach flags and make them ineligible for ordinary continuous-book diagnostics until resolved. There is no multi-venue-consolidation excuse for this single-feed reconstruction. Actual counts are in the generated report.

No affirmative redistribution license has been established. Raw and normalized rows and real traces are ignored by Git. Public accessibility alone does not grant publishing rights. Rights-holder clarification is required before an external data release.

## Engineering scope

The parser streams a buffered ZIP member, writes bounded Parquet groups, and uses integer book arithmetic. Memory grows with active orders, executed match IDs, and distinct diagnostic values, not with a loaded raw DataFrame. Match-ID history and exact histograms still grow across a day; a multi-day service must partition by session and may need disk-backed correction indexes or bounded approximate diagnostics. The measured one-day peak memory/throughput is evidence only for this adapter and host.

No ML, feature optimization, future-return labels, strategy backtest or portfolio metrics are included. The next phase, if authorized, must preserve these caveats rather than convert an engineering PASS into a trading claim.
