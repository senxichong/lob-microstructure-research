"""Lineage, row-level persistence and immutable preregistration."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .core import CLASSES, GRID, OUT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=lambda a: a.item()), encoding="utf-8")


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(name):
    with (OUT / (name + ".csv")).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_frozen():
    frozen = json.loads((OUT / "upstream_freeze.json").read_text())
    for path, h in frozen["files"].items():
        if path == ".gitignore":
            continue
        data = Path(path).read_bytes()
        if path == "RESEARCH_LOG.md":
            data = data[: frozen["log_bytes"]]
        assert hashlib.sha256(data).hexdigest() == h, path
    p2 = "data/processed/phase2/msft_20030103_phase2_v2.0.0.parquet"
    assert digest(p2) == frozen["phase2_hash"]
    assert digest(GRID) == frozen["grid_hash"]
    protocol = json.loads((OUT / "protocol_freeze.json").read_text())
    for path, h in protocol["documents"].items():
        assert digest(path) == h, path
    assert (
        hashlib.sha256(Path("RESEARCH_LOG.md").read_bytes()[: protocol["log_bytes"]]).hexdigest()
        == protocol["log_sha256"]
    )
    return frozen


def persist(g, results):
    reg, cls, coefs, tuning, preprocessing, purges, fitted = [], [], [], [], [], [], []
    for result in results:
        fold, embargo = result["fold"], result["embargo"]
        common = dict(
            **fold,
            embargo_seconds=embargo,
            feature_version="4.0.0",
            purge_rule="requested_target_timestamp < partition_end minus gap",
        )
        te = result["test_indices"]
        purges.extend(result["purge"])
        for name, a in result["artifacts"].items():
            tag = dict(model_name=name, fold_id=fold["fold_id"], embargo_seconds=embargo)
            fitted.append(dict(**tag, **a))
            tuning.extend([dict(**tag, **v) for v in a["validation_scores"]])
            for j, f in enumerate(a["features"]):
                preprocessing.append(
                    dict(
                        **tag,
                        feature=f,
                        median=a["median"][j],
                        mean=a["mean"][j],
                        scale=a["scale"][j],
                    )
                )
            b = np.atleast_2d(a["coefficient"])
            inter = np.atleast_1d(a["intercept"])
            for ci, bc in enumerate(b):
                category = CLASSES[ci] if name == "Logistic" else "return"
                for f, v in zip(a["features"], bc):
                    coefs.append(
                        dict(
                            **tag,
                            feature=f,
                            class_name=category,
                            standardized_coefficient=float(v),
                            intercept=float(inter[ci]),
                        )
                    )
        for task, store in [("regression", reg), ("classification", cls)]:
            for name, predictions in result[task].items():
                a = result["artifacts"].get(name, {})
                metadata = dict(
                    **common,
                    model_name=name,
                    selected_hyperparameter=a.get("parameter"),
                    train_purged=result["purge"][0]["purged"],
                    validation_purged=result["purge"][1]["purged"],
                    test_purged=result["purge"][2]["purged"],
                )
                for j, i in enumerate(te):
                    row = dict(
                        **metadata,
                        grid_timestamp=int(g["grid_timestamp"][i]),
                        source_state_timestamp=int(g["source_state_timestamp"][i]),
                        target_timestamp=int(g["target_timestamp_1S"][i]),
                        target=float(g["GRID_FWD_MID_RET_1S"][i]),
                    )
                    if task == "regression":
                        row["prediction"] = float(predictions[j])
                    else:
                        row.update(
                            true_class=g["GRID_DIRECTION_1S"][i],
                            p_down=float(predictions[j, 0]),
                            p_flat=float(predictions[j, 1]),
                            p_up=float(predictions[j, 2]),
                            predicted_class=CLASSES[predictions[j].argmax()],
                        )
                    store.append(row)
    for name, rows in [("regression", reg), ("classification", cls)]:
        table = pa.Table.from_pylist(rows)
        for col in table.column_names:
            if col.endswith(("_timestamp", "_start", "_end")):
                table = table.set_column(
                    table.schema.get_field_index(col),
                    col,
                    table[col].cast(pa.timestamp("ns", tz="UTC")),
                )
        pq.write_table(table, OUT / f"{name}_oos_predictions.parquet", compression="zstd")
    for name, rows in [
        ("model_coefficients", coefs),
        ("hyperparameter_selection", tuning),
        ("preprocessing_parameters", preprocessing),
        ("purge_summary", purges),
    ]:
        write_csv(name, rows)
    dump("fitted_models.json", fitted)


def prediction_groups(task):
    table = pq.read_table(OUT / f"{task}_oos_predictions.parquet")
    data = {
        k: (
            table[k].cast(pa.int64()).to_numpy()
            if pa.types.is_timestamp(table[k].type)
            else table[k].to_numpy()
        )
        for k in table.column_names
    }
    groups = {}
    for embargo in np.unique(data["embargo_seconds"]):
        for model in np.unique(data["model_name"]):
            mask = (data["embargo_seconds"] == embargo) & (data["model_name"] == model)
            if mask.any():
                idx = np.flatnonzero(mask)
                idx = idx[np.argsort(data["grid_timestamp"][idx])]
                groups[(int(embargo), str(model))] = {k: v[idx] for k, v in data.items()}
    return groups
