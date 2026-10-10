# Data licensing and repository policy

Source: [Nasdaq's publicly accessible ITCH sample directory](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/), accessed 2026-10-10. Public download availability is **not** an open-data redistribution license. The archive contains a text member and no license grant. We have not established permission to republish its records or normalized equivalents.

This repository conservatively excludes `data/raw/`, `data/interim/`, `data/processed/`, `reports/event_trace_examples.md`, and row-level failure/issue logs from Git. The required real traces are generated locally for inspection. Do not upload them or the Parquet files without rights-holder permission. Do not assume Nasdaq hosting transfers data ownership to this project. The downloaded specification is also local-only.

The public portfolio is designed to contain original code, hand-calculated synthetic tests, configuration, source URLs, SHA-256 manifests, methodological notes and aggregate quality summaries. This policy is not a legal determination that every use or published statistic is licensed. No raw or event-level derived data have been committed, and nothing has been published. Confirm intended public use with the rights holder before any external data release.

To reproduce locally, run the documented pipeline, which downloads directly from the source and verifies the pinned hash. If the URL disappears, obtain the identical artifact through an authorized route and check `DATA_MANIFEST.json`; do not silently use a mirror or a different day. A paid history subscription is a separate authorization, not provided by access to this sample.

[Nasdaq historical-data access documentation](https://www.nasdaqtrader.com/content/technicalSupport/specifications/dataproducts/ITCHFTP.pdf) describes the broader historical service. LOBSTER, Fairdata and Tardis alternatives were not adopted and their licenses do not apply to this artifact.
