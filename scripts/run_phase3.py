"""Run the single frozen Phase-3 study; refuse overwriting completed research."""

import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.download import sha256
from lob_research.phase2.artifacts import summary, write_csv
from lob_research.phase3.artifacts import (
    check_frozen,
    code_hash,
    load_phase2,
    save_tables,
    write_grid,
)
from lob_research.phase3.grid import build_grid, epoch_ns
from lob_research.phase3.study import run_study


def main():
    cfg = check_frozen()
    reports = Path(cfg["reports_dir"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "msft_20030103_grid_v3.0.0.parquet"
    if path.exists() or Path("PHASE3_MANIFEST.json").exists():
        raise FileExistsError("Phase 3 is frozen; verify existing outputs, never overwrite")
    begun = {
        "status": "incomplete",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": code_hash(),
        "protocol_freeze_sha256": sha256(reports / "protocol_freeze.json"),
    }
    (reports / "run_status.json").write_text(json.dumps(begun, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    source = load_phase2(cfg)
    start, end = epoch_ns(cfg["session_start_utc"]), epoch_ns(cfg["session_end_utc"])
    grid = build_grid(source, cfg["features"], start, end)
    tables = run_study(grid, source, cfg, start, end, lambda text: print(text, flush=True))
    save_tables(reports, tables)
    write_csv(
        reports / "grid_staleness.csv", [summary("grid_state_age_ms", grid["grid_state_age_ms"])]
    )
    labelrows = []
    for tag in ["100MS", "1S", "5S", "10S"]:
        valid = grid[f"target_available_{tag}"]
        direction = grid[f"GRID_DIRECTION_{tag}"]
        labelrows.append(
            {
                "horizon": tag,
                "grid_rows": len(valid),
                "valid": int(valid.sum()),
                "unavailable": int((~valid).sum()),
                **{k.lower(): int(np.sum(direction == k)) for k in ["UP", "FLAT", "DOWN"]},
            }
        )
    write_csv(reports / "grid_label_summary.csv", labelrows)
    write_csv(
        reports / "price_scale_context.csv",
        [
            {
                "sample": "available 1s grid targets",
                "n": int(np.isfinite(grid["GRID_MID_MOVE_1S"]).sum()),
                "median_absolute_mid_move_usd": float(
                    np.nanmedian(np.abs(grid["GRID_MID_MOVE_1S"]))
                ),
                "median_spread_usd": float(
                    np.nanmedian(grid["spread"][grid["target_available_1S"]])
                ),
                "interpretation": "descriptive price scale only; no execution profit calculation",
            }
        ],
    )
    lineage = {
        "phase": 3,
        "version": cfg["version"],
        "source_sha256": cfg["input_sha256"],
        "code_sha256": code_hash(),
        "config_sha256": sha256("config.phase3.yaml"),
        "protocol_freeze_sha256": sha256(reports / "protocol_freeze.json"),
    }
    write_grid(path, grid, lineage)
    manifest = {
        **lineage,
        "output_path": path.as_posix(),
        "output_sha256": sha256(path),
        "event_rows": len(source["timestamp"]),
        "grid_rows": len(grid["grid_timestamp"]),
        "grid_columns": len(grid),
        "valid_primary_targets": int(grid["target_available_1S"].sum()),
        "features": cfg["features"],
        "primary_horizon_ms": 1000,
        "bootstrap_replicates": cfg["bootstrap_replicates"],
        "permutation_replicates": cfg["permutation_replicates"],
        "seeds": {k: v for k, v in cfg.items() if k.endswith("_seed")},
        "runtime_seconds": time.perf_counter() - started,
        "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", None),
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "scipy", "pyarrow", "matplotlib", "pytest", "PyYAML", "psutil"]
        },
        "csv_sha256": {p.as_posix(): sha256(p) for p in sorted(reports.glob("*.csv"))},
        "chronology": [
            "protocol frozen",
            "synthetic tests",
            "primary 1s statistics and inference",
            "secondary diagnostics",
            "event descriptive estimates",
        ],
        "model_fits": 0,
        "execution_simulations": 0,
    }
    Path("PHASE3_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (reports / "run_status.json").write_text(
        json.dumps(
            {
                **begun,
                "status": "complete",
                "manifest_sha256": sha256("PHASE3_MANIFEST.json"),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    check_frozen()
    print(
        json.dumps(
            {
                "grid_rows": manifest["grid_rows"],
                "valid_primary_targets": manifest["valid_primary_targets"],
                "runtime_seconds": manifest["runtime_seconds"],
                "random_control": tables["random_control"],
            },
            indent=2,
            default=lambda value: value.item(),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
