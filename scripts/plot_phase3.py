"""Eight descriptive research figures, with explicit single-session scope."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path("reports/phase3")
    out = directory / "figures"
    out.mkdir(exist_ok=True)

    def rows(name):
        with (directory / (name + ".csv")).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    primary = rows("primary_feature_statistics")
    quintiles = rows("primary_feature_quintiles")
    names = [r["feature"] for r in primary]
    scope = "MSFT — single historical trading day — 2003-01-03"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )

    def save(fig, filename, title, note):
        fig.suptitle(title + "\n" + scope, fontsize=13, y=0.99)
        fig.text(0.02, 0.015, note, fontsize=8, color="#444444")
        fig.tight_layout(rect=(0, 0.06, 1, 0.92))
        fig.savefig(out / (filename + ".png"), dpi=180)
        fig.savefig(out / (filename + ".svg"))
        plt.close(fig)

    def bins(name):
        return [r for r in quintiles if r["feature"] == name]

    for name, filename in [
        ("OBI_L1", "01_obi_quintile_return"),
        ("OFI_1S", "02_ofi_quintile_return"),
    ]:
        data = bins(name)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        q = np.arange(1, 6)
        ax.plot(
            q,
            [float(r["mean_return"]) * 10000 for r in data],
            marker="o",
            color="#24748b",
            linewidth=2,
        )
        for x, r in zip(q, data):
            if int(r["n"]) == 0:
                ax.text(x, 0, "empty", ha="center", color="#777777")
        ax.axhline(0, color="#aaaaaa", linewidth=0.8)
        ax.set(
            xticks=q,
            xticklabels=["Q1", "Q2", "Q3", "Q4", "Q5"],
            ylabel="Mean future 1s mid return (basis points)",
            xlabel=name + " quintile",
        )
        save(
            fig,
            filename,
            name + ": quintile mean future return",
            "Ex-post descriptive bins; tied values are not split. Empty bins remain empty. No strategy return.",
        )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = bins("MICROPRICE_MINUS_MID")
    ax.bar(np.arange(1, 6), [float(r["direction_edge"]) * 100 for r in data], color="#24748b")
    ax.axhline(0, color="#aaaaaa", linewidth=0.8)
    ax.set(
        xticks=np.arange(1, 6),
        xticklabels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        ylabel="P(UP) − P(DOWN), percentage points",
        xlabel="Microprice deviation quintile",
    )
    save(
        fig,
        "03_microprice_direction_edge",
        "Microprice deviation: 1s direction difference",
        "Ex-post descriptive bins. No classifier, execution assumptions or trading profit.",
    )
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(names))
    rho = np.array([float(r["spearman_rho"]) for r in primary])
    lo = np.array([float(r["bootstrap_ci_low"]) for r in primary])
    hi = np.array([float(r["bootstrap_ci_high"]) for r in primary])
    ax.errorbar(rho, y, xerr=[rho - lo, hi - rho], fmt="o", color="#24748b", capsize=3)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    ax.set(yticks=y, yticklabels=names, xlabel="Spearman rho, primary 1s grid target")
    ax.invert_yaxis()
    save(
        fig,
        "04_primary_spearman",
        "Primary feature associations and 95% intervals",
        "1,000 paired circular bootstrap replicates; 60s blocks. Features shown in preregistered order.",
    )
    decay = rows("horizon_decay")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for name in names[:7]:
        data = sorted(
            [r for r in decay if r["feature"] == name], key=lambda r: int(r["horizon_ms"])
        )
        ax.plot(range(4), [float(r["spearman_rho"]) for r in data], marker="o", label=name)
    ax.axhline(0, color="#aaaaaa", linewidth=0.8)
    ax.set(
        xticks=range(4),
        xticklabels=["100ms", "1s (primary)", "5s", "10s"],
        ylabel="Spearman rho",
        xlabel="Wall-clock grid target horizon",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    save(
        fig,
        "05_horizon_decay",
        "Horizon persistence / decay of directional features",
        "Secondary horizons are descriptive. The primary horizon remains 1s regardless of magnitude.",
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    for ax, name in zip(axes, ["OBI_L1", "OFI_1S", "MICROPRICE_MINUS_MID"]):
        data = bins(name)
        bottom = np.zeros(5)
        for key, color, label in [
            ("p_down", "#c77c76", "DOWN"),
            ("p_flat", "#cbd2d6", "FLAT"),
            ("p_up", "#24748b", "UP"),
        ]:
            values = np.array([float(r[key]) * 100 for r in data])
            ax.bar(range(1, 6), values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set(
            title=name,
            xticks=range(1, 6),
            xticklabels=["Q1", "Q2", "Q3", "Q4", "Q5"],
            ylim=(0, 100),
        )
        for q, r in enumerate(data, 1):
            if not int(r["n"]):
                ax.text(q, 50, "empty", rotation=90, ha="center")
    axes[0].set_ylabel("1s direction probability (%)")
    axes[-1].legend(loc="lower right", fontsize=8)
    save(
        fig,
        "06_direction_probabilities",
        "UP / FLAT / DOWN probabilities by feature quintile",
        "Ex-post descriptive bins; FLAT observations retained. Empty tied bins are not manufactured.",
    )
    tod = rows("time_of_day_stability")
    matrix = np.array(
        [
            [
                float(
                    next(r for r in tod if r["feature"] == name and r["block"] == b)["spearman_rho"]
                )
                for b in "ABCDE"
            ]
            for name in names
        ]
    )
    fig, ax = plt.subplots(figsize=(9, 6.5))
    bound = max(0.05, np.max(abs(matrix)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set(
        yticks=range(10),
        yticklabels=names,
        xticks=range(5),
        xticklabels=[
            "A\n09:30–10:48",
            "B\n10:48–12:06",
            "C\n12:06–13:24",
            "D\n13:24–14:42",
            "E\n14:42–16:00",
        ],
    )
    for i in range(10):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(matrix[i, j]) > 0.65 * bound else "#222222",
            )
    fig.colorbar(im, ax=ax, label="Spearman rho")
    save(
        fig,
        "07_time_of_day",
        "Fixed session-block stability",
        "Times America/New_York. Five equal-duration blocks; no performance-selected boundaries.",
    )
    corr = rows("feature_correlation")
    matrix = np.array(
        [
            [
                float(
                    next(r for r in corr if r["feature_a"] == a and r["feature_b"] == b)[
                        "spearman_rho"
                    ]
                )
                for b in names
            ]
            for a in names
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(10), xticklabels=names, yticks=range(10), yticklabels=names)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(matrix[i, j]) > 0.65 else "#222222",
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pairwise Spearman rho")
    save(
        fig,
        "08_feature_correlation",
        "Primary-feature redundancy",
        "Pairwise finite grid observations. Correlated features are not independent economic discoveries.",
    )
    print("Eight research figures saved as PNG and SVG")


if __name__ == "__main__":
    main()
