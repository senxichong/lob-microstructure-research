import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

OUT = Path("reports/phase7")
LOCAL = Path("data/processed/phase7")
CHECKS = []


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, obj):
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=lambda a: a.item()), encoding="utf-8")


def readcsv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    OUT.mkdir(exist_ok=True, parents=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def arrays(path, columns=None):
    t = pq.read_table(path, columns=columns)
    return {
        k: t[k].cast(pa.int64()).fill_null(0).to_numpy()
        if pa.types.is_timestamp(t[k].type)
        else t[k].to_numpy()
        for k in t.column_names
    }


def record(section, name, condition, error=0.0, detail=""):
    row = dict(
        section=section,
        check=name,
        passed=bool(condition),
        max_discrepancy=float(error),
        detail=detail,
    )
    CHECKS.append(row)
    return row


def close(section, name, a, b, tol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        return record(section, name, np.array_equal(a, b))
    if a.shape != b.shape:
        return record(section, name, False, detail=f"{a.shape} != {b.shape}")
    if a.dtype.kind == "b":
        return record(section, name, np.array_equal(a, b))
    diff = np.abs(a - b)
    finite = diff[np.isfinite(diff)]
    error = float(finite.max()) if len(finite.ravel()) else 0.0
    return record(section, name, np.allclose(a, b, atol=tol, rtol=0, equal_nan=True), error)


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    a, b = rankdata(np.asarray(x)[m]), rankdata(np.asarray(y)[m])
    return float(np.corrcoef(a, b)[0, 1])


def blocks(n, length, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / length)))
    return np.concatenate([(int(s) + np.arange(length)) % n for s in starts])[:n]


def path(phase, suffix):
    m = js(f"PHASE{phase}_MANIFEST.json")
    p = [p for p in m.get("evidence_sha256", m.get("csv_sha256", {})) if p.endswith(suffix)]
    if len(p) != 1:
        raise ValueError((phase, suffix, p))
    return p[0]


def predictions(phase, task, name, embargo=0):
    if phase == 5 and name in ["Ridge", "Logistic"]:
        return predictions(4, task, name, embargo)
    a = arrays(path(phase, f"/{task}_oos_predictions.parquet"))
    m = a["model_name"] == name
    if "embargo_seconds" in a:
        m &= a["embargo_seconds"] == embargo
    i = np.flatnonzero(m)
    i = i[np.argsort(a["grid_timestamp"][i])]
    if not len(i):
        raise ValueError(f"No saved predictions for phase {phase}, {task}, {name}")
    return {k: v[i] for k, v in a.items()}
