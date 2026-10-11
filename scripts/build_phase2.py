"""Build one immutable Phase-2 matrix and descriptive-only summaries."""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.download import sha256
from lob_research.phase2.artifacts import (
    DOCS,
    arrow_array,
    code_hash,
    configuration,
    summary,
    upstream_check,
    write_csv,
)
from lob_research.phase2.common import load_stream, suffix
from lob_research.phase2.features import build_features
from lob_research.phase2.labels import build_labels
from lob_research.timestamps import iso_utc


def main():
    cfg = configuration()
    p = cfg["phase2"]
    frozen = upstream_check()
    reports = Path(p["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    output = Path(p["output_dir"]) / f"msft_20030103_phase2_v{p['protocol_version']}.parquet"
    if output.exists() or Path("PHASE2_MANIFEST.json").exists():
        raise FileExistsError(
            "Phase 2 is frozen; use verify_phase2.py. Never overwrite the matrix."
        )
    prereg = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": p["protocol_version"],
        "config_sha256": sha256("config.yaml"),
        "documents": {d: sha256(d) for d in DOCS},
    }
    freeze_path = reports / "protocol_freeze.json"
    if freeze_path.exists():
        prior = json.loads(freeze_path.read_text())
        if (
            prior["config_sha256"] != prereg["config_sha256"]
            or prior["documents"] != prereg["documents"]
        ):
            raise ValueError("preregistered definitions changed")
    else:
        freeze_path.write_text(json.dumps(prereg, indent=2) + "\n")
    (reports / "build_status.json").write_text(json.dumps({"status": "incomplete"}) + "\n")
    started = time.perf_counter()
    s = load_stream(cfg)
    f, meta = build_features(s, p)
    labels, label_names = build_labels(s, p)
    n = len(s.time)
    upstream = json.loads(Path("reports/data_quality.json").read_text())
    raw = upstream["lineage"]["raw_sha256"]
    columns = {
        "session_date": np.repeat(s.date, n),
        "symbol": np.repeat(s.symbol, n),
        "timestamp": s.time,
        "source_order": s.order,
        "event_index": np.arange(n, dtype=np.int64),
        "event_type": s.kind,
        "resting_side": s.side,
        "event_size": s.size,
        "best_bid_price": s.bid[:, 0] / 10000,
        "best_bid_size": s.bsize[:, 0],
        "best_ask_price": s.ask[:, 0] / 10000,
        "best_ask_size": s.asize[:, 0],
        "midprice": f["MIDPRICE"],
        "spread": f["SPREAD"],
        "mid_sum_price_units": (
            np.where(f["book_valid"], s.bid[:, 0] + s.ask[:, 0], 0).astype(np.int64),
            ~f["book_valid"],
        ),
        **f,
        **labels,
        "source_raw_hash": np.repeat(raw, n),
        "source_books_hash": np.repeat(sha256(p["input_books"]), n),
        "source_events_hash": np.repeat(sha256(p["input_events"]), n),
        "feature_version": np.repeat(p["feature_version"], n),
        "label_version": np.repeat(p["label_version"], n),
    }
    lineage = {
        "phase": 2,
        "protocol_version": p["protocol_version"],
        "phase1_git_commit": frozen["git_commit"],
        "input_hashes": {k: sha256(p[k]) for k in ("input_books", "input_events")},
        "raw_sha256": raw,
        "config_sha256": sha256("config.yaml"),
        "protocol_sha256": sha256("PHASE2_PROTOCOL.md"),
        "code_sha256": code_hash(),
    }
    table = pa.table(
        {name: arrow_array(name, values) for name, values in columns.items()}
    ).replace_schema_metadata({b"lineage": json.dumps(lineage, sort_keys=True).encode()})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.part")
    pq.write_table(table, temporary, compression="zstd", row_group_size=p["row_group_size"])
    temporary.rename(output)
    runtime = time.perf_counter() - started
    rss = psutil.Process().memory_info()
    manifest = {
        **lineage,
        "output_path": output.as_posix(),
        "output_sha256": sha256(output),
        "row_count": n,
        "column_count": len(columns),
        "timestamp_range": [iso_utc(int(s.time[0])), iso_utc(int(s.time[-1]))],
        "feature_list": list(meta),
        "label_list": label_names,
        "columns": list(columns),
        "availability_flags": [k for k in f if k not in meta],
        "feature_version": p["feature_version"],
        "label_version": p["label_version"],
        "tick_size_usd": None,
        "source_raw_messages": upstream["source_messages"],
        "performance": {
            "input_observations": n,
            "output_observations": n,
            "runtime_seconds": runtime,
            "observations_per_second": n / runtime,
            "peak_memory_bytes": getattr(rss, "peak_wset", rss.rss),
            "peak_memory_method": "OS peak working set"
            if hasattr(rss, "peak_wset")
            else "final RSS",
            "parquet_bytes": output.stat().st_size,
            "scope": "load + features + labels + Parquet; summaries excluded",
        },
    }
    (reports / "feature_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_csv(reports / "feature_summary.csv", [summary(k, f[k]) for k in meta])
    write_csv(
        reports / "feature_availability.csv",
        [
            {
                "flag": k,
                "available": int(v.sum()),
                "unavailable": int(n - v.sum()),
                "availability_rate": float(v.mean()),
            }
            for k, v in f.items()
            if k not in meta
        ],
    )
    label_summary, ages = [], []
    for h in sorted([p["primary_horizon_ms"], *p["secondary_horizons_ms"]]):
        tag, flag = suffix(h), suffix(h).lower()
        good = labels[f"target_available_{flag}"]
        direction = labels[f"DIRECTION_{tag}"]
        counts = {d: int(np.sum(direction == d)) for d in ("UP", "FLAT", "DOWN")}
        ret = labels[f"FWD_MID_RET_{tag}"][good]
        label_summary.append(
            {
                "horizon_ms": h,
                "valid_count": int(good.sum()),
                "unavailable_count": int((~good).sum()),
                **{d.lower() + "_count": v for d, v in counts.items()},
                **{
                    d.lower() + "_proportion": v / int(good.sum()) if good.any() else None
                    for d, v in counts.items()
                },
                "mean_future_mid_return": float(ret.mean()) if len(ret) else None,
                "std_future_mid_return": float(ret.std(ddof=1)) if len(ret) > 1 else None,
                "mid_move_ticks": "unavailable: historical tick not established",
                "tick_non_null_count": 0,
                "earliest_feature_timestamp": iso_utc(int(s.time[0])),
                "latest_valid_feature_timestamp": iso_utc(int(s.time[np.flatnonzero(good)[-1]]))
                if good.any()
                else None,
            }
        )
        ages.append(summary(flag, labels[f"target_age_ms_{flag}"]))
    write_csv(reports / "label_summary.csv", label_summary)
    write_csv(reports / "target_age_summary.csv", ages)
    write_csv(
        reports / "mid_move_summary.csv",
        [
            summary(f"MID_MOVE_{suffix(h)}", labels[f"MID_MOVE_{suffix(h)}"])
            for h in sorted([p["primary_horizon_ms"], *p["secondary_horizons_ms"]])
        ],
    )
    write_csv(
        reports / "event_type_summary.csv",
        [{"event_type": k, "count": int(np.sum(s.kind == k))} for k in sorted(set(s.kind))],
    )
    Path("PHASE2_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (reports / "build_status.json").write_text(
        json.dumps(
            {"status": "complete", "manifest_sha256": sha256("PHASE2_MANIFEST.json")}, indent=2
        )
        + "\n"
    )
    upstream_check()
    print(
        json.dumps(
            {k: manifest[k] for k in ("row_count", "column_count", "output_sha256", "performance")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
