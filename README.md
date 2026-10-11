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

## Phase 2 — completed under a separate instruction

The Phase-1 stop above records the original milestone. The subsequent Phase-2 instruction authorized causal features and wall-clock labels only. See [PHASE2_REPORT.md](PHASE2_REPORT.md) for the measured decision and [PHASE2_PROTOCOL.md](PHASE2_PROTOCOL.md) for preregistered definitions. No statistical prediction experiment or model is included. **The current stop is after Phase 2.**

Phase-1 artifacts, reports, source and tests remain frozen. `config.phase1.yaml` preserves the original configuration; `config.yaml` adds a separate `phase2` mapping. Do not rerun the Phase-1 pipeline/report commands above against this frozen working tree. Phase-2 validation checks the original upstream file hashes directly.

The matrix is `data/processed/phase2/msft_20030103_phase2_v2.0.0.parquet`, with lineage in [PHASE2_MANIFEST.json](PHASE2_MANIFEST.json). The file and local real-row audit traces remain excluded from Git under the inherited data policy. Source code and aggregate diagnostics are in `src/lob_research/phase2/` and `reports/phase2/`.

To audit the existing frozen Phase-2 output:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/phase2/test_results.xml 2>&1 | Tee-Object -FilePath reports/phase2/pytest_output.txt
.\.venv\Scripts\python.exe scripts/verify_phase2.py
.\.venv\Scripts\python.exe scripts/build_phase2_report.py
```

On a fresh local copy containing the verified Phase-1 input files, run `scripts/build_phase2.py` once before those audit commands. It creates the protocol freeze, versioned matrix, metadata and descriptive summaries and refuses to overwrite an existing matrix/manifest. Existing saved outputs should be verified, not rebuilt. The same fixed definitions are documented in [CAUSAL_ORDERING.md](CAUSAL_ORDERING.md), [FEATURE_DEFINITIONS.md](FEATURE_DEFINITIONS.md), [LABEL_DEFINITIONS.md](LABEL_DEFINITIONS.md), and [DEPENDENCE_NOTES.md](DEPENDENCE_NOTES.md). Production uses no network and no predictive evaluation.

## Phase 3 — univariate statistical research

The separately authorized Phase 3 is recorded in [PHASE3_REPORT.md](PHASE3_REPORT.md): a fixed 1-second grid, ten preregistered existing features, 1,000 paired block-bootstrap draws at 60 seconds (30/300-second robustness), 1,000 circular-shift null draws, BH/BY adjustment, time/state diagnostics and controls. [PHASE3_PROTOCOL.md](PHASE3_PROTOCOL.md), [PHASE3_FEATURE_SET.md](PHASE3_FEATURE_SET.md) and [RESEARCH_LOG.md](RESEARCH_LOG.md) preserve the hypotheses and actual results. The current stop is **after Phase 3**. No model or trading strategy was built.

Root config and all Phase-2 artifacts remain unchanged. `config.phase3.yaml` is separate; `requirements.phase3.txt` pins the research environment. The grid is `data/processed/phase3/msft_20030103_grid_v3.0.0.parquet`. Aggregate CSVs and eight PNG/SVG figures are under `reports/phase3/`; real grid rows, traces and row-level manual comparisons remain excluded from Git.

To audit the frozen study:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/phase3/test_results.xml 2>&1 | Tee-Object -FilePath reports/phase3/pytest_output.txt
.\.venv\Scripts\python.exe scripts/verify_phase3.py
.\.venv\Scripts\python.exe scripts/build_phase3_report.py
.\.venv\Scripts\python.exe scripts/verify_phase3.py --report-only
```

`scripts/run_phase3.py` is the one-time numerical study entry point and refuses to overwrite existing outputs. `scripts/plot_phase3.py` regenerates figures from frozen aggregate CSVs. A final-console NumPy serialization fix after the successful numerical study is documented in `reports/phase3/console_fix_audit.json`; original executed source is archived and the verifier proves analytic modules and saved outputs were unchanged. Do not rebuild upstream phases or rerun research with altered features, seeds or definitions to improve the results. Statistical associations here concern one historical session and do not establish tradable alpha.
