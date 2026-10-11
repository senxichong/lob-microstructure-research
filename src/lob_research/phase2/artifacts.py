"""Phase-2 lineage, summaries and typed storage; never writes Phase-1 artifacts."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import yaml

from lob_research.download import sha256

DOCS = [
    "PHASE2_PROTOCOL.md",
    "CAUSAL_ORDERING.md",
    "FEATURE_DEFINITIONS.md",
    "LABEL_DEFINITIONS.md",
    "DEPENDENCE_NOTES.md",
]


def configuration(path="config.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    p = cfg["phase2"]
    if (
        p["primary_horizon_ms"] != 1000
        or p["secondary_horizons_ms"] != [100, 5000, 10000]
        or p["history_windows_ms"] != [100, 1000, 5000, 10000]
        or p["book_depths"] != [1, 3, 5, 10]
        or p["weighting"] != "reciprocal_rank"
    ):
        raise ValueError("configuration differs from preregistered protocol")
    if p["tick_size_usd"] is not None:
        raise ValueError("real source tick remains unverified; do not impute a tick")
    original = yaml.safe_load(Path("config.phase1.yaml").read_text())
    if {k: v for k, v in cfg.items() if k != "phase2"} != original:
        raise ValueError("Phase 1 settings changed")
    return cfg


def upstream_check():
    frozen = json.loads(Path("reports/phase2/upstream_freeze.json").read_text())
    for path, digest in frozen["files"].items():
        if path not in frozen["authorized_edits"] and sha256(path) != digest:
            raise ValueError(f"Frozen upstream changed: {path}")
    if sha256("config.phase1.yaml") != frozen["files"]["config.yaml"]:
        raise ValueError("original Phase 1 config not preserved")
    return frozen


def code_hash():
    paths = sorted(Path("src/lob_research/phase2").glob("*.py")) + [Path("scripts/build_phase2.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("empty diagnostic table")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(name, values):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    out = {
        "name": name,
        "count": len(a),
        "non_null_count": int(np.sum(~np.isnan(a))),
        "availability_rate": float(np.mean(~np.isnan(a))),
        "mean": None,
        "std": None,
        "min": None,
        **{p: None for p in ("p01", "p05", "p25", "median", "p75", "p95", "p99")},
        "max": None,
        "infinite_count": int(np.isinf(a).sum()),
    }
    if len(finite):
        out.update(
            mean=float(finite.mean()),
            std=float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            min=float(finite.min()),
            max=float(finite.max()),
        )
        for p, q in zip(
            ("p01", "p05", "p25", "median", "p75", "p95", "p99"),
            (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99),
        ):
            out[p] = float(np.quantile(finite, q, method="inverted_cdf"))
    return out


def arrow_array(name, values):
    if isinstance(values, tuple):
        data, mask = values
        kind = pa.timestamp("ns", tz="UTC") if "timestamp" in name else pa.int64()
        return pa.array(data, mask=mask, type=kind)
    if name == "timestamp" or name.startswith("target_time_requested_"):
        return pa.array(values, type=pa.timestamp("ns", tz="UTC"))
    a = np.asarray(values)
    if a.dtype.kind == "f":
        return pa.array(a, mask=np.isnan(a), type=pa.float64())
    return pa.array(a)
