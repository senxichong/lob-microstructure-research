"""Reproduce the frozen Phase 7 audit without rerunning or optimizing research."""

import argparse
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lob_research.phase7 import (  # noqa: E402
    data_audit,
    execution_audit,
    redteam,
    reporting,
    statistical_audit,
)
from lob_research.phase7.common import CHECKS, OUT, arrays, dump, js, sha  # noqa: E402


def tests():
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--junitxml=reports/phase7/test_results.xml"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    (OUT / "pytest_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    root = ET.parse(OUT / "test_results.xml").getroot()
    suites = list(root.iter("testsuite"))

    def count(k):
        return sum(int(s.attrib.get(k, 0)) for s in suites)

    warnings = re.findall(r"(\d+) warnings?", proc.stdout)
    result = dict(
        passed=count("tests") - count("failures") - count("errors") - count("skipped"),
        failed=count("failures"),
        errors=count("errors"),
        skipped=count("skipped"),
        warnings=int(warnings[-1]) if warnings else 0,
        runtime_seconds=time.perf_counter() - started,
        exit_code=proc.returncode,
    )
    dump("test_summary.json", result)
    if proc.returncode:
        raise RuntimeError("Full pytest failed; see reports/phase7/pytest_output.txt")
    return result


def main():
    import os

    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "data", "statistics", "execution", "redteam", "tests", "summarize"],
        default="all",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = (
        ["data", "statistics", "execution", "redteam", "tests", "summarize"]
        if args.stage == "all"
        else [args.stage]
    )
    started = time.perf_counter()
    runtimes = {}
    for stage in selected:
        CHECKS.clear()
        begin = time.perf_counter()
        print("Phase 7: " + stage, flush=True)
        if stage == "data":
            lineage = data_audit.integrity()
            dump("integrity_checks.json", CHECKS)
            b, e = data_audit.raw_replay(lineage)
            data_audit.features_grid(b, e)
            dump("data_checks.json", CHECKS)
        elif stage == "statistics":
            statistical_audit.run(
                arrays(js("PHASE3_MANIFEST.json")["output_path"]),
                yaml.safe_load(Path("config.phase3.yaml").read_text()),
            )
        elif stage == "execution":
            execution_audit.run(
                arrays(js("data/processed/lineage.json")["artifacts"]["books"]["path"]),
                arrays(js("PHASE3_MANIFEST.json")["output_path"]),
            )
        elif stage == "redteam":
            redteam.run()
        elif stage == "tests":
            tests()
        else:
            data_audit.integrity()
            dump("integrity_checks.json", CHECKS)
            history = subprocess.run(
                ["git", "log", "--all", "--format=%h %ad %s", "--date=iso-strict"],
                capture_output=True,
                text=True,
                check=True,
            )
            (OUT / "git_history.txt").write_text(history.stdout, encoding="utf-8")
            summary = reporting.summarize(js(OUT / "test_summary.json"))
            if summary["status"] != "PASS":
                raise RuntimeError("Phase 7 gate not satisfied")
        failed = [r for r in CHECKS if not r["passed"]]
        if failed:
            raise AssertionError(json.dumps(failed, indent=2))
        runtimes[stage] = time.perf_counter() - begin
        print(f"{stage}: PASS ({runtimes[stage]:.1f}s)", flush=True)
    if args.stage == "all":
        dump(
            "audit_run.json",
            dict(
                stage_seconds=runtimes,
                total_seconds=time.perf_counter() - started,
                protocol_sha256=sha("PHASE7_AUDIT_PROTOCOL.md"),
            ),
        )
    if "summarize" in selected:
        files = (
            [
                Path("PHASE7_AUDIT_PROTOCOL.md"),
                Path("QUANT_AUDIT.md"),
                Path("scripts/audit_phase7.py"),
                Path("tests/test_phase7_audit.py"),
            ]
            + list(Path("src/lob_research/phase7").glob("*.py"))
            + [p for p in OUT.rglob("*") if p.is_file()]
        )
        manifest = dict(
            phase=7,
            status="PASS",
            evidence_sha256={p.as_posix(): sha(p) for p in sorted(files)},
            upstream_modified=False,
            stop_after_phase7=True,
        )
        Path("PHASE7_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
