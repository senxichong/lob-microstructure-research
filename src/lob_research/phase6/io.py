import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path("reports/phase6")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(value, indent=2, default=lambda x: x.item()), encoding="utf-8"
    )


def csvwrite(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def csvread(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def arrays(path, columns=None):
    t = pq.read_table(path, columns=columns)
    return {
        k: t[k].cast(pa.int64()).to_numpy() if pa.types.is_timestamp(t[k].type) else t[k].to_numpy()
        for k in t.column_names
    }


def parquet(name, rows):
    pq.write_table(pa.Table.from_pylist(rows), OUT / (name + ".parquet"), compression="zstd")


def evidence_path(phase, suffix):
    paths = [
        p for p in read_json(f"PHASE{phase}_MANIFEST.json")["evidence_sha256"] if p.endswith(suffix)
    ]
    assert len(paths) == 1, (phase, suffix, paths)
    return paths[0]


def predictions(model):
    phase = 4 if model == "Ridge" else 5
    a = arrays(evidence_path(phase, "/regression_oos_predictions.parquet"))
    mask = a["model_name"] == model
    if phase == 4:
        mask &= a["embargo_seconds"] == 0
    i = np.flatnonzero(mask)
    i = i[np.argsort(a["grid_timestamp"][i])]
    return {k: v[i] for k, v in a.items()}


def frozen():
    u = read_json(OUT / "upstream_freeze.json")
    for f, h in u["files"].items():
        b = Path(f).read_bytes()
        if f in u["prefix_bytes"]:
            b = b[: u["prefix_bytes"][f]]
        assert hashlib.sha256(b).hexdigest() == h, f
    p = read_json(OUT / "protocol_freeze.json")
    assert digest("PHASE6_PROTOCOL.md") == p["protocol_sha256"]
    assert (
        hashlib.sha256(Path("RESEARCH_LOG.md").read_bytes()[: p["log_bytes"]]).hexdigest()
        == p["log_sha256"]
    )
    if (OUT / "signal_freeze.json").exists():
        for f, h in read_json(OUT / "signal_freeze.json")["files"].items():
            assert digest(f) == h, f
    return u
