import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lob_research.phase4.core import CLASSES

OUT = Path("reports/phase5")
ART = Path("artifacts/phase5")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=lambda a: a.item()), encoding="utf-8")


def csvwrite(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def csvread(name):
    with (OUT / (name + ".csv")).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frozen():
    u = json.loads((OUT / "upstream_freeze.json").read_text())
    for p, h in u["files"].items():
        b = Path(p).read_bytes()
        if p == "RESEARCH_LOG.md":
            b = b[: u["log_bytes"]]
        assert hashlib.sha256(b).hexdigest() == h, p
    assert digest(u["phase2_path"]) == u["phase2_hash"]
    p = json.loads((OUT / "protocol_freeze.json").read_text())
    for f, h in p["documents"].items():
        assert digest(f) == h, f
    assert (
        hashlib.sha256(Path("RESEARCH_LOG.md").read_bytes()[: p["log_bytes"]]).hexdigest()
        == p["log_hash"]
    )
    return u


def persist(g, results):
    ART.mkdir(parents=True, exist_ok=True)
    rows = {"regression": [], "classification": []}
    selection = []
    meta = []
    purge = []
    model_paths = []
    versions = {
        n: importlib.metadata.version(n) for n in ["numpy", "scipy", "scikit-learn", "lightgbm"]
    }
    for result in results:
        fold = result["fold"]
        te = result["partitions"]["test"]
        purge.extend(result["purge"])
        for task, entries in result["results"].items():
            for name, e in entries.items():
                tag = dict(task=task, model_name=name, fold_id=fold["fold_id"])
                metadata = {**tag, **e["metadata"], "versions": versions, **fold}
                meta.append(metadata)
                for candidate in e["candidates"]:
                    selection.append(
                        dict(**tag, **candidate, selected_family=e["metadata"]["family"])
                    )
                path = ART / f"{task}_{name}_fold{fold['fold_id']}.joblib"
                joblib.dump(dict(model=e["model"], metadata=metadata), path, compress=3)
                model_paths.append(dict(**tag, path=path.as_posix(), sha256=digest(path)))
                pred = e["prediction"]
                for j, i in enumerate(te):
                    row = dict(
                        **fold,
                        model_name=name,
                        selected_family=e["metadata"]["family"],
                        selected_candidate=e["metadata"]["candidate_id"],
                        feature_version="5.0.0",
                        grid_timestamp=int(g["grid_timestamp"][i]),
                        source_state_timestamp=int(g["source_state_timestamp"][i]),
                        target_timestamp=int(g["target_timestamp_1S"][i]),
                        target=float(g["GRID_FWD_MID_RET_1S"][i]),
                    )
                    if task == "regression":
                        row["prediction"] = float(pred[j])
                    else:
                        row.update(
                            true_class=g["GRID_DIRECTION_1S"][i],
                            predicted_class=CLASSES[pred[j].argmax()],
                            p_down=float(pred[j, 0]),
                            p_flat=float(pred[j, 1]),
                            p_up=float(pred[j, 2]),
                        )
                    rows[task].append(row)
    for task, items in rows.items():
        t = pa.Table.from_pylist(items)
        for name in t.column_names:
            if name.endswith(("_timestamp", "_start", "_end")):
                t = t.set_column(
                    t.schema.get_field_index(name), name, t[name].cast(pa.timestamp("ns", tz="UTC"))
                )
        pq.write_table(t, OUT / f"{task}_oos_predictions.parquet", compression="zstd")
    dump("model_metadata.json", meta)
    dump("model_artifacts.json", model_paths)
    csvwrite("model_selection", selection)
    csvwrite("purge_summary", purge)


def groups(task):
    t = pq.read_table(OUT / f"{task}_oos_predictions.parquet")
    a = {
        k: t[k].cast(pa.int64()).to_numpy() if pa.types.is_timestamp(t[k].type) else t[k].to_numpy()
        for k in t.column_names
    }
    output = {}
    for name in np.unique(a["model_name"]):
        i = np.flatnonzero(a["model_name"] == name)
        i = i[np.argsort(a["grid_timestamp"][i])]
        output[str(name)] = {k: v[i] for k, v in a.items()}
    return output
