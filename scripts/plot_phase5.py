import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.phase5.storage import OUT, csvread

DEST = OUT / "figures"
SCOPE = "MSFT · 2003-01-03 · single historical trading day"


def save(fig, name, title):
    fig.suptitle(title + "\n" + SCOPE, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(DEST / (name + ".png"), dpi=150)
    fig.savefig(DEST / (name + ".svg"))
    plt.close(fig)


def main():
    DEST.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    reg = {r["model_name"]: r for r in csvread("regression_metrics")}
    cla = {r["model_name"]: r for r in csvread("classification_metrics")}
    for task, rows, field, names, num, title in [
        (
            "regression",
            reg,
            "spearman",
            ["OBI_L1", "Ridge", "RF", "LightGBM", "ValidationSelected"],
            "01",
            "OOS rank association",
        ),
        (
            "classification",
            cla,
            "macro_f1",
            ["AlwaysFLAT", "Logistic", "RF", "LightGBM", "ValidationSelected"],
            "03",
            "OOS Macro F1 — natural class distribution",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        values = [float(rows[n][field]) for n in names]
        ax.bar(names, values, color=["#8197ac", "#668d82", "#b39268", "#8e7ea7", "#a1a1a1"])
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_ylabel(field)
        for i, v in enumerate(values):
            ax.annotate(
                f"{v:.4f}",
                (i, v),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set_ylim(min(0, min(values) * 1.2), max(values) * 1.2)
        save(fig, num + "_" + task, title)
    for task, field, names, num, title in [
        (
            "regression",
            "spearman",
            ["Ridge", "RF", "LightGBM", "ValidationSelected"],
            "02",
            "Fold-by-fold OOS Spearman",
        ),
        (
            "classification",
            "macro_f1",
            ["Logistic", "RF", "LightGBM", "ValidationSelected"],
            "04",
            "Fold-by-fold OOS Macro F1",
        ),
    ]:
        rows = csvread(task + "_fold_metrics")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for name in names:
            r = sorted(
                [r for r in rows if r["model_name"] == name], key=lambda r: int(r["fold_id"])
            )
            ax.plot(
                range(1, 6),
                [float(x[field]) for x in r],
                marker="o",
                label=name,
                ls="--" if name == "ValidationSelected" else "-",
            )
        ax.set(xticks=range(1, 6), xlabel="Chronological test fold", ylabel=field)
        ax.axhline(0, color="gray", lw=0.6)
        ax.legend(fontsize=9)
        save(fig, num + "_fold_" + task, title)
    rows = csvread("bootstrap_draws")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name in ["RF", "LightGBM", "ValidationSelected"]:
        delta = [float(r["reg_" + name]) - float(r["reg_Ridge"]) for r in rows]
        ax.hist(delta, bins=35, alpha=0.38, label=name, density=True)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Spearman difference versus frozen Ridge", ylabel="Bootstrap density")
    ax.legend()
    save(fig, "05_delta_bootstrap", "Paired 60-second temporal bootstrap — 1,000 replicates")
    imp = csvread("feature_importance")
    per = csvread("permutation_importance")
    for filename, data, key, num, title in [
        (
            "native",
            imp,
            "normalized_importance",
            "06",
            "LightGBM native gain — interpretation, not causality",
        ),
        (
            "permutation",
            per,
            "degradation",
            "07",
            "Held-out temporal permutation importance — interpretation only",
        ),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
        for ax, task in zip(axes, ["regression", "classification"]):
            rs = [r for r in data if r["model_name"] == "LightGBM" and r["task"] == task]
            names = list(dict.fromkeys(r["feature"] for r in rs))
            values = [np.mean([float(r[key]) for r in rs if r["feature"] == n]) for n in names]
            order = np.argsort(values)
            ax.barh(np.array(names)[order], [values[i] for i in order])
            ax.set_title(task)
            ax.set_xlabel(
                "Mean normalized gain" if filename == "native" else "Mean OOS metric degradation"
            )
            ax.axvline(0, color="gray", lw=0.6)
        save(fig, num + "_" + filename, title)
    names = ["Ridge", "RF", "LightGBM"]
    rows = csvread("prediction_correlation")
    matrix = np.array(
        [
            [
                float(next(r["spearman"] for r in rows if r["model_a"] == a and r["model_b"] == b))
                for b in names
            ]
            for a in names
        ]
    )
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if abs(matrix[i, j]) > 0.6 else "black",
            )
    ax.set(xticks=range(3), xticklabels=names, yticks=range(3), yticklabels=names)
    save(fig, "08_prediction_correlation", "OOS prediction correlation")
    rows = csvread("response_shapes")
    names = ["OBI_L1", "MICROPRICE_MINUS_MID", "OFI_1S", "OFI_100MS"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, name in zip(axes.flat, names):
        for fold in range(1, 6):
            r = [r for r in rows if r["feature"] == name and int(r["fold_id"]) == fold]
            ax.plot(
                [float(x["value"]) for x in r],
                [float(x["mean_prediction"]) * 1e4 for x in r],
                marker=".",
                label="Fold " + str(fold),
            )
        ax.set(xlabel=name, ylabel="Partial dependence (return bp)")
        ax.axhline(0, color="gray", lw=0.5)
    axes[0, 0].legend(fontsize=8)
    save(
        fig,
        "09_response_shapes",
        "LightGBM response shapes — train-quantile grids; no causal interpretation",
    )
    rows = csvread("train_validation_test_gap")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, task in zip(axes, ["regression", "classification"]):
        for name in ["RF", "LightGBM", "ValidationSelected"]:
            r = [r for r in rows if r["task"] == task and r["model_name"] == name]
            ax.plot(
                ["Train", "Validation", "Test"],
                [
                    np.mean([float(v[k]) for v in r])
                    for k in ["train_metric", "validation_metric", "test_metric"]
                ],
                marker="o",
                label=name,
            )
        ax.set(
            title=task,
            ylabel="Mean fold Spearman" if task == "regression" else "Mean fold Macro F1",
        )
        ax.legend(fontsize=8)
    save(fig, "10_fit_gap", "Training / validation / test gap — same selected fold models")
    print("Ten PNG/SVG figure pairs created.")


if __name__ == "__main__":
    main()
