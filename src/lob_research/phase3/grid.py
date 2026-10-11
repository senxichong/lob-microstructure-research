"""Clock-anchored labels from immutable event states, never copied event targets."""

from datetime import datetime

import numpy as np

from lob_research.phase2.common import suffix


def epoch_ns(iso):
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()) * 1000000000


def time_blocks(times, start, end, count=5):
    times = np.asarray(times)
    if np.any(times < start) or np.any(times >= end):
        raise ValueError("time outside half-open session")
    return ((times - start) * count // (end - start)).astype(int)


def build_grid(source, features, start, end, horizons=(100, 1000, 5000, 10000)):
    t = source["timestamp"]
    order = source["source_order"]
    if len(t) == 0 or np.any(np.diff(t) < 0) or np.any(np.diff(order) <= 0):
        raise ValueError("require nonempty authoritative source order")
    if np.any(t < start) or np.any(t >= end):
        raise ValueError("cross-session source forbidden")
    g = np.arange(start, end, 1000000000, dtype=np.int64)
    i = np.searchsorted(t, g, side="right") - 1
    safe = np.maximum(i, 0)
    valid = np.asarray(source["book_valid"], bool)
    current = (i >= 0) & valid[safe]
    sums = source["mid_sum_price_units"]
    out = {
        "grid_timestamp": g,
        "source_event_index": np.where(current, i, -1),
        "source_state_timestamp": np.where(current, t[safe], 0),
        "source_state_order": np.where(current, order[safe], -1),
        "grid_state_available": current,
        "grid_state_age_ms": np.where(current, (g - t[safe]) / 1000000, np.nan),
        "mid_sum_price_units": np.where(current, sums[safe], 0),
    }
    for name in [
        "midprice",
        "best_bid_price",
        "best_ask_price",
        "best_bid_size",
        "best_ask_size",
        "spread",
        *features,
    ]:
        out[name] = np.where(current, source[name][safe], np.nan)
    bad = np.r_[0, np.cumsum(~valid)]
    for h in horizons:
        tag = suffix(h)
        tau = g + h * 1000000
        j = np.searchsorted(t, tau, side="right") - 1
        target = np.maximum(j, 0)
        uninterrupted = bad[target + 1] == bad[safe]
        available = current & (tau < end) & (j >= i) & uninterrupted
        ret = np.divide(sums[target], sums[safe], out=np.zeros(len(g)), where=sums[safe] != 0) - 1
        direction = np.where(
            sums[target] > sums[safe], "UP", np.where(sums[target] < sums[safe], "DOWN", "FLAT")
        ).astype(object)
        direction[~available] = None
        reason = np.where(
            ~current,
            "current_state_unavailable",
            np.where(
                tau >= end,
                "session_boundary",
                np.where(~uninterrupted, "invalid_book_interval", "available"),
            ),
        )
        out.update(
            {
                f"target_available_{tag}": available,
                f"target_timestamp_{tag}": tau,
                f"target_state_timestamp_{tag}": np.where(available, t[target], 0),
                f"target_state_order_{tag}": np.where(available, order[target], -1),
                f"target_event_index_{tag}": np.where(available, j, -1),
                f"target_age_ms_{tag}": np.where(available, (tau - t[target]) / 1000000, np.nan),
                f"target_reason_{tag}": reason,
                f"GRID_FWD_MID_RET_{tag}": np.where(available, ret, np.nan),
                f"GRID_MID_MOVE_{tag}": np.where(
                    available, (sums[target] - sums[safe]) / 20000, np.nan
                ),
                f"GRID_DIRECTION_{tag}": direction,
            }
        )
    return out
