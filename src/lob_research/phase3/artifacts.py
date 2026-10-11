"""Phase-3-only storage and upstream immutability checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from lob_research.download import sha256
from lob_research.phase2.artifacts import write_csv


def configuration():
    return yaml.safe_load(Path("config.phase3.yaml").read_text(encoding="utf-8"))


def check_frozen():
    cfg = configuration()
    if sha256(cfg["input_matrix"]) != cfg["input_sha256"]:
        raise ValueError("STOP: frozen Phase 2 matrix hash mismatch; never rebuild it")
    frozen = json.loads(Path("reports/phase3/upstream_freeze.json").read_text(encoding="utf-8"))
    for path, digest in frozen["files"].items():
        if path not in frozen["authorized_edits"] and sha256(path) != digest:
            raise ValueError(f"Frozen upstream changed: {path}")
    protocol = json.loads(Path("reports/phase3/protocol_freeze.json").read_text(encoding="utf-8"))
    for path, digest in protocol["documents"].items():
        if sha256(path) != digest:
            raise ValueError(f"Preregistered definition changed: {path}")
    if (
        not Path("RESEARCH_LOG.md")
        .read_text(encoding="utf-8")
        .startswith(protocol["research_log_prefix"])
    ):
        raise ValueError("Original research hypotheses changed")
    return cfg


def code_hash():
    paths = sorted(Path("src/lob_research/phase3").glob("*.py")) + [Path("scripts/run_phase3.py")]
    h = hashlib.sha256()
    for path in paths:
        h.update(path.as_posix().encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def load_phase2(cfg):
    names = list(
        dict.fromkeys(
            [
                "timestamp",
                "source_order",
                "event_index",
                "book_valid",
                "mid_sum_price_units",
                "midprice",
                "best_bid_price",
                "best_ask_price",
                "best_bid_size",
                "best_ask_size",
                "spread",
                "FWD_MID_RET_1S",
                *cfg["features"],
            ]
        )
    )
    table = pq.read_table(cfg["input_matrix"], columns=names)
    output = {}
    for name in names:
        col = table[name]
        output[name] = (
            col.cast(pa.int64()).to_numpy() if pa.types.is_timestamp(col.type) else col.to_numpy()
        )
    return output


def write_grid(path, data, lineage):
    columns = {}
    for name, values in data.items():
        a = np.asarray(values)
        if "timestamp" in name:
            columns[name] = pa.array(a, mask=a == 0, type=pa.timestamp("ns", tz="UTC"))
        elif a.dtype.kind == "f":
            columns[name] = pa.array(a, mask=np.isnan(a))
        else:
            columns[name] = pa.array(a)
    table = pa.table(columns).replace_schema_metadata(
        {b"lineage": json.dumps(lineage, sort_keys=True).encode()}
    )
    pq.write_table(table, path, compression="zstd")


def read_grid(path):
    table = pq.read_table(path)
    return {
        name: (
            table[name].cast(pa.int64()).fill_null(0).to_numpy()
            if pa.types.is_timestamp(table[name].type)
            else table[name].to_numpy()
        )
        for name in table.column_names
    }


def save_tables(directory, tables):
    for name, rows in tables.items():
        write_csv(Path(directory) / (name + ".csv"), rows)
