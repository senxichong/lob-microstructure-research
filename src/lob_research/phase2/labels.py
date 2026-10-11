"""Wall-clock as-of mid labels; never imported by the feature engine."""

import numpy as np

from .common import safe_ratio, suffix, valid_depth


def build_labels(s, cfg):
    s.validate()
    valid = valid_depth(s, 1)
    n = len(s.time)
    indices = np.arange(n)
    sums = np.where(valid, s.bid[:, 0] + s.ask[:, 0], 0).astype(np.int64)
    bad = np.r_[0, np.cumsum(~valid)]
    output, label_names = {}, []
    for h in sorted([cfg["primary_horizon_ms"], *cfg["secondary_horizons_ms"]]):
        tag, flag = suffix(h), suffix(h).lower()
        tau = s.time + h * 1000000
        j = np.searchsorted(s.time, tau, side="right") - 1
        good = valid & (tau < s.end) & (j >= indices)
        uninterrupted = bad[j + 1] - bad[indices] == 0
        good &= uninterrupted
        reason = np.where(
            ~valid,
            "current_book_invalid",
            np.where(
                tau >= s.end,
                "session_boundary",
                np.where(~uninterrupted, "invalid_book_interval", "available"),
            ),
        )
        age = (tau - s.time[j]) / 1000000
        if np.any(age[good] < 0):
            raise AssertionError("negative target age")
        delta = sums[j] - sums
        output.update(
            {
                f"target_available_{flag}": good,
                f"target_time_requested_{flag}": tau,
                f"target_state_timestamp_{flag}": (s.time[j], ~good),
                f"target_state_source_order_{flag}": (s.order[j], ~good),
                f"target_age_ms_{flag}": np.where(good, age, np.nan),
                f"target_midprice_{flag}": np.where(good, sums[j] / 20000, np.nan),
                f"target_unavailable_reason_{flag}": reason,
            }
        )
        output[f"FWD_MID_RET_{tag}"] = np.where(good, safe_ratio(sums[j], sums) - 1, np.nan)
        output[f"MID_MOVE_{tag}"] = np.where(good, delta / 20000, np.nan)
        tick = cfg["tick_size_usd"]
        output[f"MID_MOVE_TICKS_{tag}"] = (
            output[f"MID_MOVE_{tag}"] / tick if tick is not None else np.full(n, np.nan)
        )
        direction = np.where(delta > 0, "UP", np.where(delta < 0, "DOWN", "FLAT")).astype(object)
        direction[~good] = None
        output[f"DIRECTION_{tag}"] = direction
        label_names += [
            f"{name}_{tag}" for name in ("FWD_MID_RET", "MID_MOVE", "MID_MOVE_TICKS", "DIRECTION")
        ]
    return output, label_names
