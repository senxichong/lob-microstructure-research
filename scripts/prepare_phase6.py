"""Integrity + validation-only thresholds, strictly no strategy returns."""

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
from scipy.stats import spearmanr

from lob_research.phase4.core import folds, partitions
from lob_research.phase5.core import impute, predict
from lob_research.phase6.io import (
    OUT,
    arrays,
    csvread,
    csvwrite,
    digest,
    dump,
    evidence_path,
    parquet,
    predictions,
    read_json,
)


def main():
    assert not (OUT / "protocol_freeze.json").exists(), "already frozen"
    OUT.mkdir(parents=True, exist_ok=True)
    old = read_json("reports/phase5/upstream_freeze.json")
    inspected = []
    for phase in [4, 5]:
        m = read_json(f"PHASE{phase}_MANIFEST.json")
        for p, h in m["evidence_sha256"].items():
            b = Path(p).read_bytes()
            if p == "RESEARCH_LOG.md" and phase == 4:
                b = b[: old["log_bytes"]]
            assert hashlib.sha256(b).hexdigest() == h, p
            if p.endswith(".json"):
                read_json(p)
            elif p.endswith(".csv"):
                csvread(p)
            elif p.endswith(".parquet"):
                arrays(p)
            inspected.append(p)
    models = read_json(evidence_path(5, "/model_artifacts.json"))
    for r in models:
        assert digest(r["path"]) == r["sha256"]
    p2, p3 = [read_json(f"PHASE{i}_MANIFEST.json") for i in (2, 3)]
    assert (
        digest(p2["output_path"])
        == p2["output_sha256"]
        == "a5a493636d9f6d592e2d229921533940ec7cfffcdc5da8f1c9fb636445d34e96"
    )
    assert digest(p3["output_path"]) == p3["output_sha256"]
    rm = {r["model_name"]: r for r in csvread(evidence_path(5, "/regression_metrics.csv"))}
    cm = {r["model_name"]: r for r in csvread(evidence_path(5, "/classification_metrics.csv"))}
    for name, expected in [("Ridge", 0.050747), ("RF", 0.097540), ("LightGBM", 0.098637)]:
        assert abs(float(rm[name]["spearman"]) - expected) < 5e-7
        s = predictions(name)
        assert len(s["prediction"]) == 11695
        assert (
            abs(spearmanr(s["prediction"], s["target"]).statistic - float(rm[name]["spearman"]))
            < 1e-12
        )
    for name, expected in [("Logistic", 0.302183), ("RF", 0.345043), ("LightGBM", 0.384455)]:
        assert abs(float(cm[name]["macro_f1"]) - expected) < 5e-7
    verdict = read_json(evidence_path(5, "/research_verdict.json"))
    assert verdict["regression"] == "RELIABLE NONLINEAR IMPROVEMENT"
    assert verdict["classification"] == "RELIABLE NONLINEAR DIRECTIONAL IMPROVEMENT"
    paired = csvread(evidence_path(5, "/nonlinear_vs_ridge_bootstrap.csv"))
    print("Frozen paired comparison:", paired)
    paths = set(inspected + [r["path"] for r in models] + [p2["output_path"], p3["output_path"]])
    paths.update(p.as_posix() for p in Path(".").glob("*.md"))
    paths.update(p.as_posix() for p in Path(".").glob("*MANIFEST.json"))
    for root in ["src", "scripts", "tests", "config"]:
        paths.update(
            p.as_posix()
            for p in Path(root).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and "phase6" not in p.as_posix()
        )
    paths.discard("PHASE6_PROTOCOL.md")
    paths.add(".gitignore")
    dump(
        "upstream_freeze.json",
        dict(
            files={p: digest(p) for p in sorted(paths)},
            prefix_bytes={p: Path(p).stat().st_size for p in ["RESEARCH_LOG.md", ".gitignore"]},
            phase2_path=p2["output_path"],
            grid_path=p3["output_path"],
            evidence_files_checked=len(inspected),
            models_checked=len(models),
        ),
    )
    with Path("RESEARCH_LOG.md").open("a", encoding="utf-8") as f:
        f.write(
            "\n## Phase 6 preregistration — before execution PnL\n\n"
            "H1: The LightGBM mid-price predictive improvement may be substantially reduced after aggressive bid/ask execution.\n\n"
            "H2: Higher prediction strength should be associated with better economic outcomes, but spread may dominate weaker signals.\n\n"
            "H3: Latency should degrade the short-horizon signal.\n\n"
            "H4: LightGBM should economically outperform Ridge if its predictive improvement is genuinely useful.\n\n"
            "H5: The signal may remain statistically predictive while failing to be executable after spread and costs.\n\n"
            "Protocol 6.0.0 fixes validation-only 10% tails, stateful one share, 10ms aggressive BBO execution and all robustness/control assumptions. No real PnL viewed.\n"
        )
    with Path(".gitignore").open("a", encoding="utf-8") as f:
        f.write(
            "\n# Phase 6 local-only row-level market/signal/execution artifacts\nreports/phase6/*.parquet\nreports/phase6/episode_ledger.csv\nreports/phase6/manual_pnl_verification.csv\nreports/phase6_execution_traces.md\n"
        )
    dump(
        "protocol_freeze.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            protocol_sha256=digest("PHASE6_PROTOCOL.md"),
            log_bytes=Path("RESEARCH_LOG.md").stat().st_size,
            log_sha256=digest("RESEARCH_LOG.md"),
        ),
    )
    g = arrays(p3["output_path"])
    linear = read_json(evidence_path(4, "/fitted_models.json"))
    valrows, thresholdrows, provenance = [], [], []
    baseline = predictions("LightGBM")["grid_timestamp"]
    for name in ["Ridge", "RF", "LightGBM"]:
        s = predictions(name)
        assert np.array_equal(s["grid_timestamp"], baseline)
        for fold in folds():
            fid = fold["fold_id"]
            parts, _ = partitions(g, fold)
            vi, ti = parts["validation"], parts["test"]
            if name == "Ridge":
                m = next(
                    r
                    for r in linear
                    if r["model_name"] == name and r["fold_id"] == fid and r["embargo_seconds"] == 0
                )

                def infer(idx):
                    x = np.column_stack([g[f][idx] for f in m["features"]])
                    z = (np.where(np.isfinite(x), x, m["median"]) - m["mean"]) / m["scale"]
                    return z @ np.asarray(m["coefficient"]) + m["intercept"]

                vp, tp = infer(vi), infer(ti)
                path = evidence_path(4, "/fitted_models.json")
            else:
                rec = next(
                    r
                    for r in models
                    if r["task"] == "regression" and r["model_name"] == name and r["fold_id"] == fid
                )
                path = rec["path"]
                bundle = joblib.load(path)
                m = bundle["metadata"]

                def infer(idx):
                    x = np.column_stack([g[f][idx] for f in m["features"]])
                    return predict(bundle["model"], impute(x, m["median"]), "regression")

                vp, tp = infer(vi), infer(ti)
            saved = s["prediction"][s["fold_id"] == fid]
            error = float(np.max(np.abs(tp - saved)))
            assert error < 1e-12
            provenance.append(
                dict(
                    model_name=name,
                    fold_id=fid,
                    model_path=path,
                    model_sha256=digest(path),
                    validation_n=len(vi),
                    validation_start=int(g["grid_timestamp"][vi[0]]),
                    validation_last=int(g["grid_timestamp"][vi[-1]]),
                    test_prediction_max_error=error,
                )
            )
            valrows.extend(
                dict(
                    model_name=name,
                    fold_id=fid,
                    grid_timestamp=int(g["grid_timestamp"][j]),
                    prediction=float(v),
                )
                for j, v in zip(vi, vp)
            )
            for tail in [0.05, 0.10, 0.20, 0.30]:
                lo, hi = np.quantile(vp, [tail, 1 - tail], method="linear")
                assert lo < hi
                row = dict(
                    model_name=name,
                    fold_id=fid,
                    tail=tail,
                    q_low=float(lo),
                    q_high=float(hi),
                    validation_n=len(vp),
                )
                row.update(
                    {
                        f"q{int(q * 100):02d}": float(np.quantile(vp, q))
                        for q in [0.01, 0.05, 0.10, 0.90, 0.95, 0.99]
                    }
                )
                thresholdrows.append(row)
    parquet("validation_predictions", valrows)
    csvwrite("thresholds", thresholdrows)
    csvwrite("prediction_provenance", provenance)
    dump(
        "upstream_verification.json",
        dict(
            status="PASS",
            phase2_sha256=p2["output_sha256"],
            evidence_files_checked=len(inspected),
            models_checked=len(models),
            metrics_verified=True,
            identical_oos=11695,
            paired_comparison=paired,
        ),
    )
    files = [
        OUT / (n + e)
        for n, e in [
            ("validation_predictions", ".parquet"),
            ("thresholds", ".csv"),
            ("prediction_provenance", ".csv"),
        ]
    ]
    dump(
        "signal_freeze.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            files={p.as_posix(): digest(p) for p in files},
            no_model_refitting=True,
            no_real_pnl_calculated=True,
        ),
    )
    print(
        "PASS: integrity, protocol and validation-only thresholds frozen; no strategy PnL calculated."
    )


if __name__ == "__main__":
    main()
