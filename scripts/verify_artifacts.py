"""Verify persisted Parquet, lineage, real traces and independently recomputed diagnostics."""

import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.download import fetch, sha256
from lob_research.schemas import parse_line


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    report = json.loads(Path("reports/data_quality.json").read_text())
    status = json.loads(Path("reports/run_status.json").read_text())
    require(status["status"] == "complete", "latest run incomplete")
    require(
        status["quality_sha256"] == sha256("reports/data_quality.json"), "quality report changed"
    )
    require(status["config_sha256"] == sha256("config.yaml"), "config changed since run")
    for entry in json.loads(Path("DATA_MANIFEST.json").read_text())["artifacts"]:
        # Also restore the documented source specification on a clean checkout.
        fetch(entry["source"], entry["filename"], "DATA_MANIFEST.json")
        require(sha256(entry["filename"]) == entry["sha256"], "raw manifest mismatch")
        require(Path(entry["filename"]).stat().st_size == entry["size_bytes"], "raw size mismatch")
    for name, artifact in report["artifacts"].items():
        require(sha256(artifact["path"]) == artifact["sha256"], f"{name} hash mismatch")
        parquet = pq.ParquetFile(artifact["path"])
        require(parquet.metadata.num_rows == artifact["rows"], f"{name} count mismatch")
        require(
            json.loads(parquet.schema_arrow.metadata[b"lineage"]) == report["lineage"],
            f"{name} lineage mismatch",
        )
    # Only selected diagnostic columns are read; raw input is never materialized.
    depth = 10
    columns = [
        "source_row",
        "timestamp",
        "source_time_ms",
        "best_bid_price",
        "best_ask_price",
        "spread",
        "midprice",
        "research_eligible",
    ]
    columns += [
        f"{side}_{field}_{level}"
        for side in ("bid", "ask")
        for field in ("price_units", "size")
        for level in range(1, depth + 1)
    ]
    previous_row = -1
    previous_ns = -1
    checked = 0
    for batch in pq.ParquetFile(report["artifacts"]["books"]["path"]).iter_batches(
        batch_size=10000, columns=columns
    ):
        values = batch.to_pydict()
        rows = np.array(values["source_row"], dtype=np.int64)
        ns = batch.column(batch.schema.get_field_index("timestamp")).cast("int64").to_numpy()
        require(rows[0] > previous_row and bool(np.all(np.diff(rows) > 0)), "row order failed")
        require(ns[0] >= previous_ns and bool(np.all(np.diff(ns) >= 0)), "timestamp order failed")
        require(bool(np.all(ns % 1000000 == 0)), "false sub-ms precision")
        bids = np.array(
            [values[f"bid_price_units_{i}"] for i in range(1, depth + 1)], dtype=np.int64
        ).T
        asks = np.array(
            [values[f"ask_price_units_{i}"] for i in range(1, depth + 1)], dtype=np.int64
        ).T
        sizes = np.array(
            [values[f"{side}_size_{i}"] for side in ("bid", "ask") for i in range(1, depth + 1)],
            dtype=np.int64,
        )
        require(bool(np.all(bids > 0) and np.all(asks > 0) and np.all(sizes > 0)), "invalid level")
        require(
            bool(np.all(np.diff(bids, axis=1) < 0) and np.all(np.diff(asks, axis=1) > 0)),
            "level ordering",
        )
        require(bool(np.all(bids[:, 0] < asks[:, 0])), "locked/crossed")
        require(
            np.allclose(values["spread"], (asks[:, 0] - bids[:, 0]) / 10000, rtol=0, atol=1e-12),
            "spread units",
        )
        require(
            np.allclose(values["midprice"], (asks[:, 0] + bids[:, 0]) / 20000, rtol=0, atol=1e-12),
            "mid units",
        )
        require(
            np.allclose(values["best_bid_price"], bids[:, 0] / 10000, rtol=0, atol=1e-12),
            "BBO extraction",
        )
        require(
            np.allclose(values["best_ask_price"], asks[:, 0] / 10000, rtol=0, atol=1e-12),
            "BBO extraction",
        )
        require(all(values["research_eligible"]), "unexpected ineligible selected observation")
        previous_row, previous_ns = rows[-1], ns[-1]
        checked += batch.num_rows
    events = pq.read_table(
        report["artifacts"]["events"]["path"], columns=["source_row", "event_type"]
    )
    books = pq.read_table(
        report["artifacts"]["books"]["path"], columns=["source_row", "event_type"]
    )
    require(events.equals(books), "events and snapshots not one-to-one")
    trades = pq.read_table(
        report["artifacts"]["trades"]["path"],
        columns=["source_row", "event_type", "aggressor_side"],
    )
    require(trades["aggressor_side"].null_count == trades.num_rows, "invented aggressor side")
    require(set(trades["event_type"].to_pylist()) <= {"E", "P"}, "nontrade in trades")
    require(
        trades.num_rows
        == report["rth_event_types"].get("E", 0) + report["rth_event_types"].get("P", 0),
        "trade count",
    )
    traces = [
        json.loads(block)
        for block in re.findall(
            r"```json\n(.*?)\n```", Path("reports/event_trace_examples.md").read_text(), re.S
        )
    ]
    require(len(traces) >= 10, "fewer than ten real traces")
    wanted = {t["event"]["source_row"]: t for t in traces}
    with zipfile.ZipFile("data/raw/S010303-v2.zip") as z:
        with io.BufferedReader(
            z.open(report["lineage"]["member"]), buffer_size=1024 * 1024
        ) as stream:
            for row, line in enumerate(stream, 1):
                if row in wanted:
                    parsed = parse_line(line, row)
                    trace = wanted[row]
                    event = trace["event"]
                    require(
                        parsed.kind == event["event_type"]
                        and parsed.order_id == event["order_id"]
                        and parsed.size == event["size"],
                        "trace does not match raw event",
                    )
                    require(
                        parsed.time_ms == event["source_time_ms"],
                        "trace timestamp differs from source",
                    )
                    if parsed.kind in {"A", "P"}:
                        require(
                            parsed.price_units == event["price_units"]
                            and parsed.side == event["side"]
                            and parsed.symbol == event["symbol"],
                            "trace price/side/symbol differs from source",
                        )
                    before_size, after_size = trace["order_size_before"], trace["order_size_after"]
                    if parsed.kind in {"X", "E"}:
                        require(
                            before_size - (after_size or 0) == parsed.size, "trace decrement wrong"
                        )
                    elif parsed.kind == "A":
                        require(
                            before_size is None and after_size == parsed.size, "trace add wrong"
                        )
                    elif parsed.kind == "P":
                        require(
                            trace["book_before"] == trace["book_after"], "hidden trade changed book"
                        )
                    require(trace["before_source_row"] < row, "pre-trade quote is not causal")
                if row >= max(wanted):
                    break
    code = hashlib.sha256()
    for path in sorted(Path("src").rglob("*.py")):
        code.update(path.as_posix().encode() + b"\0" + path.read_bytes())
    result = {
        "status": "pass",
        "book_rows_independently_checked": checked,
        "real_traces_crosschecked_to_raw": len(traces),
        "raw_and_output_hashes": "verified",
        "event_snapshot_alignment": "one-to-one",
        "source_code_sha256": code.hexdigest(),
        "requirements_sha256": sha256("requirements.txt"),
        "quality_sha256": sha256("reports/data_quality.json"),
    }
    Path("reports/artifact_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
