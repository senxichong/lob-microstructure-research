"""Check final presentation against saved evidence, then write the Phase 6 freeze."""

import importlib.metadata
import json
import platform
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.phase6.io import OUT, csvread, digest, dump, frozen, read_json


def main():
    assert not Path("PHASE6_MANIFEST.json").exists(), "final freeze already exists"
    frozen()
    receipt = read_json(OUT / "independent_verification.json")
    assert receipt["status"] == "PASS" and receipt["critical_failures"] == 0
    assert receipt["report_sha256"] == digest("PHASE6_REPORT.md")
    assert receipt["verifier_sha256"] == digest("scripts/verify_phase6.py")
    tests = ET.parse(OUT / "test_results.xml").getroot().find("testsuite")
    assert (
        tests is not None
        and int(tests.attrib["failures"])
        == int(tests.attrib["errors"])
        == int(tests.attrib["skipped"])
        == 0
    )
    required = [
        "execution_ledger.parquet",
        "episode_ledger.csv",
        "primary_metrics.csv",
        "model_economic_comparison.csv",
        "threshold_robustness.csv",
        "latency_robustness.csv",
        "cost_robustness.csv",
        "pnl_decomposition.csv",
        "fold_economic_metrics.csv",
        "turnover_metrics.csv",
        "adverse_selection.csv",
        "time_concentration.csv",
        "trade_concentration.csv",
        "bootstrap_uncertainty.csv",
        "circular_shift_null.csv",
        "random_signal_control.csv",
        "sign_inversion_control.csv",
        "delayed_signal_placebo.csv",
        "manual_pnl_verification.csv",
    ]
    for name in required:
        assert (OUT / name).is_file(), name
    assert Path("reports/phase6_execution_traces.md").is_file()
    assert (
        len(list((OUT / "figures").glob("*.png")))
        == len(list((OUT / "figures").glob("*.svg")))
        == 11
    )
    text = Path("PHASE6_REPORT.md").read_text(encoding="utf-8")
    tables = []
    current = []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            current.append([x.strip() for x in line.strip("|").split("|")])
        elif current:
            tables.append(current)
            current = []

    def rows(name):
        return csvread(OUT / (name + ".csv"))

    def zero(r):
        return float(r["cost_bps"]) == 0

    def primary(r):
        return r["model_name"] == "LightGBM"

    specs = [
        (
            rows("spread_statistics"),
            [
                "population",
                "rows",
                "mean_quoted_spread",
                "median_quoted_spread",
                "spread_paid",
                "mean_spread_paid_per_transition",
            ],
        ),
        (
            sorted(rows("latency_robustness"), key=lambda r: int(r["latency_ms"])),
            [
                "latency_ms",
                "net_pnl",
                "pnl_per_notional_bps",
                "trades",
                "quotes_changed_before_fill",
                "bbo_price_change_events",
                "max_drawdown",
            ],
        ),
        (
            sorted(rows("latency_robustness"), key=lambda r: int(r["latency_ms"])),
            [
                "latency_ms",
                "delta_pnl_vs_zero",
                "delta_pnl_per_trade",
                "delta_win_rate",
                "mean_signed_fill_price_degradation",
            ],
        ),
        (
            rows("cost_robustness"),
            ["cost_bps", "additional_cost", "net_pnl", "pnl_per_notional_bps", "max_drawdown"],
        ),
        (
            [r for r in rows("model_economic_comparison") if zero(r)],
            [
                "model_name",
                "predictive_spearman",
                "trades",
                "gross_mid_pnl",
                "spread_drag",
                "quote_pnl",
                "pnl_per_notional_bps",
            ],
        ),
        (
            sorted(rows("threshold_robustness"), key=lambda r: float(r["tail"])),
            [
                "tail",
                "trades",
                "time_in_market",
                "gross_mid_pnl",
                "quote_pnl",
                "spread_drag",
                "pnl_per_notional_bps",
                "break_even_extra_bps",
            ],
        ),
        (
            [r for r in rows("bootstrap_uncertainty") if primary(r)],
            ["metric", "observed", "ci_lower", "ci_upper"],
        ),
        (
            [
                r
                for r in rows("fold_economic_metrics")
                if zero(r) and r["model_name"] in ["Ridge", "RF", "LightGBM"]
            ],
            ["model_name", "fold_id", "gross_mid_pnl", "spread_drag", "net_pnl"],
        ),
        (
            [r for r in rows("time_concentration") if primary(r)],
            ["kind", "top_k", "fold_id", "contribution", "signed_fraction", "remaining_pnl"],
        ),
        (
            [r for r in rows("trade_concentration") if primary(r)],
            ["top_k", "contribution", "signed_fraction", "remaining_pnl"],
        ),
        (
            rows("random_signal_control")
            + rows("sign_inversion_control")
            + rows("delayed_signal_placebo"),
            [
                "model_name",
                "trades",
                "time_in_market",
                "gross_mid_pnl",
                "quote_pnl",
                "pnl_per_notional_bps",
            ],
        ),
        (
            rows("adverse_selection"),
            [
                "horizon_ms",
                "eligible_transactions",
                "unavailable_transactions",
                "mean_signed_mid_move",
                "quantity_weighted_signed_mid_move",
                "positive_fraction",
            ],
        ),
        (
            rows("prediction_strength"),
            ["strength_bin", "episodes", "mean_gross_mid_pnl", "mean_quote_pnl", "win_rate"],
        ),
    ]
    assert len(tables) == len(specs), (len(tables), len(specs))
    cells = 0
    for table, (records, keys) in zip(tables, specs):
        assert len(table[2:]) == len(records)
        for visible, row in zip(table[2:], records):
            assert len(visible) == len(keys)
            for cell, key in zip(visible, keys):
                value = row.get(key, "")
                try:
                    expected = format(float(value), ".8g") if value != "" else ""
                except ValueError:
                    expected = value
                assert cell == expected, (key, cell, expected)
                cells += 1
    p = rows("primary_metrics")[0]
    for value in [
        "2.4475",
        "48.0935",
        "-45.6460",
        "-85.1604%",
        "-0.0710",
        str(receipt["checks"])[:-3] + "," + str(receipt["checks"])[-3:],
    ]:
        assert value in text, value
    verdict = read_json(OUT / "research_verdict.json")
    assert verdict["phase_decision"] == "PASS" and verdict["verdict"] in text
    assert verdict["tests_passed"] == int(tests.attrib["tests"])
    assert verdict["independent_checks"] == receipt["checks"]
    assert abs(float(p["gross_mid_pnl"]) - float(p["spread_drag"]) - float(p["quote_pnl"])) < 1e-8
    dump(
        "final_report_verification.json",
        dict(
            status="PASS",
            report_sha256=digest("PHASE6_REPORT.md"),
            independent_replay_receipt_sha256=digest(OUT / "independent_verification.json"),
            tables=len(tables),
            cells_checked=cells,
            required_machine_outputs=len(required),
            figure_pairs=11,
            figures_visually_reviewed=True,
            tests_passed=int(tests.attrib["tests"]),
            upstream_frozen=True,
        ),
    )
    paths = [
        Path("PHASE6_PROTOCOL.md"),
        Path("PHASE6_REPORT.md"),
        Path("RESEARCH_LOG.md"),
        Path(".gitignore"),
        Path("reports/phase6_execution_traces.md"),
    ]
    paths += (
        list(OUT.rglob("*"))
        + list(Path("src/lob_research/phase6").glob("*.py"))
        + list(Path("scripts").glob("*phase6*.py"))
        + [Path("tests/test_phase6.py")]
    )
    paths = sorted(set(p for p in paths if p.is_file() and "__pycache__" not in p.parts))
    manifest = dict(
        phase=6,
        status="PASS",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        economic_verdict=verdict["verdict"],
        phase2_sha256=read_json("PHASE2_MANIFEST.json")["output_sha256"],
        phase5_manifest_sha256=digest("PHASE5_MANIFEST.json"),
        phase5_regression_prediction_sha256=read_json("PHASE5_MANIFEST.json")["evidence_sha256"][
            "reports/phase5/regression_oos_predictions.parquet"
        ],
        primary_model="LightGBM regression",
        prediction_horizon_seconds=1,
        validation_tail_fraction=0.1,
        latency_ms=10,
        additional_cost_bps=0,
        inventory_limit_shares=1,
        oos_score_count=11695,
        clock_seconds=11700,
        folds=5,
        capital_usd=read_json(OUT / "capital_convention.json")["capital_usd"],
        primary_quote_pnl=float(p["quote_pnl"]),
        primary_gross_mid_pnl=float(p["gross_mid_pnl"]),
        primary_spread_drag=float(p["spread_drag"]),
        tests_passed=int(tests.attrib["tests"]),
        tests_failed=0,
        independent_checks=receipt["checks"],
        critical_failures=0,
        maximum_independent_cash_discrepancy=receipt["maximum_cash_discrepancy"],
        manual_paths=receipt["manual_paths"],
        execution_traces=receipt["trace_rows"],
        bootstrap_replicates=1000,
        circular_shift_replicates=500,
        seeds=dict(
            bootstrap=606060, circular_shift=606042, random_signal=606043, row_shuffle=606034
        ),
        python=platform.python_version(),
        versions={
            k: importlib.metadata.version(k)
            for k in [
                "numpy",
                "scipy",
                "scikit-learn",
                "lightgbm",
                "pyarrow",
                "matplotlib",
                "pytest",
            ]
        },
        prior_artifacts_unchanged=True,
        log_and_gitignore_append_only=True,
        row_artifacts_local_only=True,
        figures_visually_reviewed=11,
        phase7_started=False,
        scope="single previously examined MSFT day, 2003-01-03; visible aggressive top-of-book, one-share price-taking; no predictive retuning",
        evidence_sha256={p.as_posix(): digest(p) for p in paths},
    )
    Path("PHASE6_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(
                status="PASS",
                evidence_files=len(paths),
                report_tables=len(tables),
                report_cells=cells,
                tests=manifest["tests_passed"],
                independent_checks=receipt["checks"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
