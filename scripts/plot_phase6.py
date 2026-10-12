import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lob_research.phase6.io import OUT, arrays, csvread

DEST = OUT / "figures"
DEST.mkdir(exist_ok=True)
SCOPE = "MSFT | 2003-01-03 | single historical trading day\naggressive top-of-book simulation | price-taking assumption"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)


def read(name):
    return csvread(OUT / (name + ".csv"))


def f(r, k):
    return float(r[k])


def fig(title):
    figure, axis = plt.subplots(figsize=(10.4, 6.2))
    figure.suptitle(title, fontsize=15, y=0.98)
    figure.text(0.5, 0.90, SCOPE, ha="center", va="center", fontsize=10)
    figure.subplots_adjust(top=0.80, bottom=0.18, left=0.12, right=0.96)
    axis.grid(alpha=0.18)
    return figure, axis


def save(figure, name):
    figure.savefig(DEST / (name + ".png"), dpi=150)
    figure.savefig(DEST / (name + ".svg"))
    plt.close(figure)


def main():
    clock = arrays(OUT / "clock_pnl.parquet")
    m = clock["model_name"] == "LightGBM"
    x = np.arange(m.sum()) / 60
    figure, ax = fig("Midprice edge is consumed by quote crossing")
    ax.plot(
        x,
        clock["gross_equity"][m],
        label="MIDPRICE GROSS: non-executable upper bound",
        color="#20877d",
    )
    ax.plot(x, clock["equity"][m], label="QUOTE-EXECUTABLE: 10ms, zero extra fees", color="#bd4644")
    ax.set(xlabel="Minutes from OOS start (12:45 Eastern)", ylabel="Cumulative one-share PnL (USD)")
    ax.legend(loc="lower left", fontsize=9)
    save(figure, "01_mid_vs_quote")
    figure, ax = fig("Better ranking does not produce positive economic PnL")
    for name in ["Ridge", "RF", "LightGBM"]:
        z = clock["model_name"] == name
        ax.plot(x, clock["equity"][z], label=name)
    ax.set(xlabel="Minutes from OOS start", ylabel="10ms quote PnL (USD), zero extra fees")
    ax.legend()
    save(figure, "02_models")
    row = read("primary_metrics")[0]
    figure, ax = fig("LightGBM PnL decomposition at 10ms")
    values = [f(row, "gross_mid_pnl"), -f(row, "spread_drag"), f(row, "net_pnl")]
    ax.bar(
        [
            "Midprice gross\n(non-executable)",
            "Spread crossing\n(at delayed quotes)",
            "Quote-executable\nPnL",
        ],
        values,
        color=["#20877d", "#d68a37", "#bd4644"],
    )
    for j, v in enumerate(values):
        ax.text(j, v + (1.5 if v >= 0 else -2.5), f"${v:.4f}", ha="center")
    ax.set(ylabel="USD", ylim=(-55, 10))
    save(figure, "03_decomposition")
    rows = sorted(read("latency_robustness"), key=lambda r: f(r, "latency_ms"))
    figure, ax = fig("Latency sensitivity is small and nonmonotonic here")
    ax.plot(
        [f(r, "latency_ms") for r in rows], [f(r, "net_pnl") for r in rows], "o-", color="#bd4644"
    )
    ax.set(
        xlabel="Latency (ms); 0ms is an idealized upper-bound assumption",
        ylabel="Quote PnL (USD), 10% tails / zero extra fees",
    )
    save(figure, "04_latency")
    rows = read("cost_robustness")
    figure, ax = fig("Additional one-way costs compound the loss")
    ax.plot(
        [f(r, "cost_bps") for r in rows], [f(r, "net_pnl") for r in rows], "o-", color="#bd4644"
    )
    ax.set(
        xlabel="Generic extra cost (bps of one-way traded notional)",
        ylabel="Net PnL (USD), 10ms / 10% tails",
    )
    figure.text(
        0.5,
        0.05,
        "Spread already included. These are not verified historical exchange fees.",
        ha="center",
        fontsize=9,
    )
    save(figure, "05_costs")
    rows = sorted(read("threshold_robustness"), key=lambda r: f(r, "tail"))
    figure, ax = fig("All predeclared tail thresholds lose after spread")
    xx = np.arange(len(rows))
    ax.bar(
        xx - 0.18,
        [f(r, "gross_mid_pnl") for r in rows],
        0.36,
        label="Mid gross: non-executable",
        color="#20877d",
    )
    ax.bar(
        xx + 0.18, [f(r, "net_pnl") for r in rows], 0.36, label="10ms quote PnL", color="#bd4644"
    )
    ax.set_xticks(
        xx,
        [f"{100 * f(r, 'tail'):.0f}%" + (" PRIMARY" if f(r, "tail") == 0.1 else "") for r in rows],
    )
    ax.set(xlabel="Symmetric validation-derived tails (no threshold optimization)", ylabel="USD")
    ax.legend()
    save(figure, "06_thresholds")
    rows = read("fold_economic_metrics")
    figure, ax = fig("Losses persist across all five OOS folds")
    for j, name in enumerate(["Ridge", "RF", "LightGBM"]):
        rr = [r for r in rows if r["model_name"] == name and f(r, "cost_bps") == 0]
        ax.bar(np.arange(5) + (j - 1) * 0.25, [f(r, "net_pnl") for r in rr], 0.25, label=name)
    ax.set_xticks(np.arange(5), [f"Fold {i}" for i in range(1, 6)])
    ax.set(ylabel="10ms quote PnL (USD)")
    ax.legend(loc="lower left")
    save(figure, "07_folds")
    rows = read("prediction_strength")
    figure, ax = fig("Stronger scores improve gross edge, still below spread")
    xx = np.arange(len(rows))
    ax.bar(
        xx - 0.19,
        [100 * f(r, "mean_gross_mid_pnl") for r in rows],
        0.38,
        label="Mid gross",
        color="#20877d",
    )
    ax.bar(
        xx + 0.19,
        [100 * f(r, "mean_quote_pnl") for r in rows],
        0.38,
        label="Executable",
        color="#bd4644",
    )
    ax.set_xticks(
        xx,
        [
            "Short\n0–1%",
            "Short\n1–5%",
            "Short\n5–10%",
            "Long\n90–95%",
            "Long\n95–99%",
            "Long\n99–100%",
        ],
    )
    ax.set(
        xlabel="Validation percentile of episode-entry score",
        ylabel="Mean PnL per episode (US cents)",
    )
    ax.legend()
    save(figure, "08_strength")
    rows = read("adverse_selection")
    figure, ax = fig("Post-fill signed mid moves are small relative to half-spread")
    ax.bar(
        np.arange(3),
        [100 * f(r, "mean_signed_mid_move") for r in rows],
        color="#20877d",
        label="Mean signed post-fill mid move",
    )
    ax.axhline(
        100 * f(row, "mean_execution_half_spread"),
        color="#bd4644",
        linestyle="--",
        label="Mean execution half-spread",
    )
    ax.set_xticks(np.arange(3), ["10ms", "100ms", "1s"])
    ax.set(
        ylabel="US cents per share",
        xlabel="Diagnostic horizon; positive is directionally favorable, not causal",
    )
    ax.legend()
    save(figure, "09_adverse_selection")
    rows = read("circular_shift_null")
    figure, ax = fig("Observed executable PnL does not exceed the shift null")
    ax.hist(
        [f(r, "quote_pnl") for r in rows],
        bins=30,
        color="#9ab5c7",
        edgecolor="white",
        label="500 within-fold large signal rotations",
    )
    ax.axvline(float(row["net_pnl"]), color="#bd4644", linewidth=2, label="Observed LightGBM")
    ax.set(xlabel="Quote PnL (USD), 10ms / frozen validation thresholds", ylabel="Null draws")
    ax.legend(fontsize=9)
    save(figure, "10_shift_null")
    figure, ax = fig("Equity and drawdown: fixed one-share normalization")
    eq = clock["equity"][m]
    dd = np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq
    ax.plot(x, eq, label="Quote equity / PnL (USD)", color="#bd4644")
    ax.fill_between(x, 0, -dd, alpha=0.18, color="#bd4644", label="Negative drawdown (USD)")
    ax.set(
        xlabel="Minutes from OOS start; displayed at 1s, audit marks every source event",
        ylabel="USD",
    )
    ax.legend(loc="lower left")
    save(figure, "11_equity_drawdown")
    print("11 PNG/SVG figure pairs saved.")


if __name__ == "__main__":
    main()
