# Limit Order Book Microstructure & Short-Horizon Alpha Research

限价订单簿微观结构与短周期 Alpha 研究

**Phase 1 only: data feasibility, market-data semantics and book integrity.** The project reconstructs real historical US-equity order data and documents what can and cannot be inferred from it. No prediction, model training, alpha optimization, strategy backtest, cost calculation or execution simulation is implemented.

Selected input: Nasdaq-hosted `S010303-v2.zip`, legacy Island/INET-lineage ITCH 2.0, **2003-01-03**, with **MSFT** as the initial research instrument. The entire 6,115,981-message feed is streamed to validate references and reconstruct MSFT from day-start; 162,879 regular-window observations and 13,419 trades are retained. Source time precision is 1 ms. Full visible order state is maintained and up to ten occupied levels per side are saved.

Read [PHASE1_REPORT.md](PHASE1_REPORT.md) for the gate decision, [DATA_SOURCE_EVALUATION.md](DATA_SOURCE_EVALUATION.md) for tested alternatives, and [reports/data_quality_report.md](reports/data_quality_report.md) for measured diagnostics. This old single-day sample demonstrates methodology, not modern tradability.

## Reproduce on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/inspect_environment.py
.\.venv\Scripts\python.exe scripts/probe_sources.py
.\.venv\Scripts\python.exe scripts/run_phase1.py
.\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/test_results.xml
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
.\.venv\Scripts\python.exe scripts/build_phase1_report.py
```

On Linux/macOS use `.venv/bin/python`. Python >=3.11 is required; the measured run used 3.12.4. `requirements.txt` pins the tested environment. `pyproject.toml` supports installation with `pip install -e .` and the `lob-phase1` command, but an editable install is not required for the scripts.

The first pipeline run downloads approximately 59 MB and streams a 212 MB ZIP member. It does not extract the full text, read it into pandas, or download the multi-GB alternative. Subsequent runs verify the SHA-256 cache without network access. A changed/unregistered raw file is refused; there is no force-overwrite switch. Retry failures go to `reports/ingestion.log`. An incomplete run has `reports/run_status.json` marked incomplete and must not be treated as passing, even if older outputs exist.

Artifact verification also obtains the small official specification PDF when absent, using its pinned manifest hash. The verification script targets this Phase-1 MSFT/top-ten configuration; a different dataset/depth requires an explicit review of its verification contract.

`reports/semantic_review.json` records the source-document and trace review for this exact raw/config hash. Numerical checks alone cannot certify semantics. A different artifact or configuration needs a fresh review; the report builder must not inherit this sample's decision automatically.

The pipeline also accepts `--config path.yaml`. Date, symbol, hours, depth and batch size are explicit. This adapter is for the selected ITCH 2.0 archive schema; changing a URL does not make it an ITCH 5.0 parser. More dates require verified source semantics and session calendars.

## Outputs and layout

| Path | Purpose |
|---|---|
| `src/lob_research/` | Streaming ingestion, exact schema, timestamps, order-level engine, invariants and Parquet storage |
| `scripts/` | Reproduction, environment/access probes, artifact verification and final report |
| `tests/` | Synthetic hand-calculated unit and end-to-end tests |
| `data/raw/` | Immutable ZIP and downloaded specification; ignored by Git |
| `data/processed/` | Events, post-event books, trades, trade corrections and lineage; ignored by Git |
| `reports/data_quality.json` | Machine-readable measured counts and distributions |
| `reports/data_quality_report.md` | Diagnostic report, including performance and limitations |
| `reports/event_trace_examples.md` | Twelve real before/event/after examples, local-only |
| `reports/book_issues.jsonl` | All observed selected-book issues, including outside the output window; local-only |
| `DATA_MANIFEST.json` | Exact source URLs, download times, sizes and SHA-256 hashes |

Review [MARKET_DATA_SCHEMA.md](MARKET_DATA_SCHEMA.md), [TIMESTAMP_CONVENTION.md](TIMESTAMP_CONVENTION.md), [DATA_LICENSE.md](DATA_LICENSE.md), and [DATA_LIMITATIONS.md](DATA_LIMITATIONS.md) before using the data. [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) and [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) govern later work but authorize no experiments now.

Public download does not establish redistribution rights. Raw records, normalized Parquet, row-level logs and real event traces are excluded from Git. No data or code have been published by this workflow. The repository intentionally contains no empty ML/backtest placeholder modules.

**Stop after Phase 1.** Any later feature or prediction work requires a new instruction.
