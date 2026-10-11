"""Aggregate figures from persisted OOS data; no model fitting."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.phase4.core import FEATURES, OUT
from lob_research.phase4.storage import prediction_groups, read_csv

SCOPE = "MSFT · 2003-01-03 · single historical trading day"
DEST = OUT / "figures"


def save(fig, name, title):
    fig.suptitle(title + "\n" + SCOPE, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(DEST / (name + ".png"), dpi=160)
    fig.savefig(DEST / (name + ".svg"))
    plt.close(fig)


def main():
    DEST.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    d = prediction_groups("regression")[(0, "Ridge")]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot = ax.hexbin(
        d["prediction"] * 1e4, d["target"] * 1e4, gridsize=45, mincnt=1, bins="log", cmap="viridis"
    )
    ax.set(xlabel="Predicted 1s mid return (bp)", ylabel="Realized 1s mid return (bp)")
    fig.colorbar(plot, ax=ax, label="Count per density bin (log color)")
    save(fig, "01_prediction_realized", "Ridge OOS prediction / realization — binned density")
    rf = [r for r in read_csv("regression_fold_metrics") if r["embargo_seconds"] == "0"]
    cf = [r for r in read_csv("classification_fold_metrics") if r["embargo_seconds"] == "0"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name in ("OBI_L1", "OLS", "Ridge"):
        rows = [r for r in rf if r["model_name"] == name]
        ax.plot(
            [int(r["fold_id"]) for r in rows],
            [float(r["spearman"]) for r in rows],
            marker="o",
            label=name,
            linestyle="--" if name == "OLS" else "-",
        )
    ax.axhline(0, color="gray", lw=0.7)
    ax.set(
        xticks=range(1, 6),
        xlabel="Chronological test fold (39 minutes each)",
        ylabel="OOS Spearman rho",
    )
    ax.legend()
    save(fig, "02_fold_spearman", "Test-fold rank association — OLS and Ridge nearly coincide")
    co = [
        r
        for r in read_csv("model_coefficients")
        if r["model_name"] == "Ridge" and r["embargo_seconds"] == "0"
    ]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for name in FEATURES:
        rows = [r for r in co if r["feature"] == name]
        ax.plot(
            [int(r["fold_id"]) for r in rows],
            [float(r["standardized_coefficient"]) * 1e4 for r in rows],
            marker="o",
            label=name,
        )
    ax.axhline(0, color="gray", lw=0.7)
    ax.set(xticks=range(1, 6), xlabel="Fold", ylabel="Return coefficient (bp per training SD)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    save(fig, "03_ridge_coefficients", "Training-standardized Ridge coefficient stability")
    rows = [
        r
        for r in read_csv("confusion_matrix")
        if r["model_name"] == "Logistic" and r["embargo_seconds"] == "0" and r["fold_id"] == "0"
    ]
    cats = ["DOWN", "FLAT", "UP"]
    cm = np.zeros((3, 3), int)
    for r in rows:
        cm[cats.index(r["true_class"]), cats.index(r["predicted_class"])] = int(r["count"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="OOS sample count")
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set(
        xticks=range(3),
        xticklabels=cats,
        yticks=range(3),
        yticklabels=cats,
        xlabel="Predicted class",
        ylabel="True class",
    )
    save(fig, "04_logistic_confusion", "Multinomial Logistic — FLAT remains dominant")
    rows = read_csv("calibration")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, c in zip(axes, ["DOWN", "UP"]):
        rs = [r for r in rows if r["class_name"] == c and int(r["n"]) > 0]
        x = [float(r["mean_probability"]) for r in rs]
        y = [float(r["observed_frequency"]) for r in rs]
        n = [int(r["n"]) for r in rs]
        ax.plot([0, 1], [0, 1], color="gray", ls="--")
        ax.scatter(x, y, s=20 + np.log1p(n) * 12)
        for a, b, nn in zip(x, y, n):
            if nn < 100:
                ax.annotate(
                    "n=" + str(nn), (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points"
                )
        ax.set(
            xlim=(0, 1.05),
            ylim=(0, 1.08),
            xlabel="Mean predicted probability",
            ylabel="Observed frequency",
            title=c + " · fixed 0.1 bins",
        )
    save(fig, "05_calibration", "OOS probability reliability — sparse tails remain uncertain")
    rows = {r["model_name"]: r for r in read_csv("regression_inference")}
    names = ["Static", "Flow", "Ridge"]
    val = np.array([float(rows[n]["spearman"]) for n in names])
    lo = np.array([float(rows[n]["ci_lower"]) for n in names])
    hi = np.array([float(rows[n]["ci_upper"]) for n in names])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(["Static", "Flow", "Full Ridge"], val, color=["#537895", "#65a49b", "#bd875b"])
    ax.errorbar(range(3), val, yerr=[val - lo, hi - val], fmt="none", color="black", capsize=5)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set(ylabel="OOS Spearman rho with paired 60s-block 95% CI")
    save(fig, "06_static_flow_full", "Fixed feature-family comparison")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name in ("Ridge", "Static", "Flow"):
        rows = [r for r in rf if r["model_name"] == name]
        axes[0].plot(range(1, 6), [float(r["spearman"]) for r in rows], marker="o", label=name)
    for name in ("Logistic", "AlwaysFLAT"):
        rows = [r for r in cf if r["model_name"] == name]
        axes[1].plot(range(1, 6), [float(r["macro_f1"]) for r in rows], marker="o", label=name)
    labels = ["12:45", "13:24", "14:03", "14:42", "15:21"]
    for ax in axes:
        ax.set(xticks=range(1, 6), xticklabels=labels, xlabel="Test block start (Eastern)")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("OOS Spearman")
    axes[0].axhline(0, color="gray", lw=0.7)
    axes[1].set_ylabel("Macro F1")
    save(fig, "07_intraday_stability", "Intraday stability — OOS covers 12:45–16:00 Eastern only")
    print("Created seven PNG/SVG figure pairs.")


if __name__ == "__main__":
    main()
