"""Consecutive-book OFI, with no trade-sign inference."""

import numpy as np


def ofi_event(s, valid):
    n = len(s.time)
    out = np.full(n, np.nan)
    b, a, qb, qa = s.bid[:, 0], s.ask[:, 0], s.bsize[:, 0], s.asize[:, 0]
    value = (
        (b[1:] >= b[:-1]) * qb[1:]
        - (b[1:] <= b[:-1]) * qb[:-1]
        - (a[1:] <= a[:-1]) * qa[1:]
        + (a[1:] >= a[:-1]) * qa[:-1]
    )
    out[1:] = np.where(valid[1:] & valid[:-1], value, np.nan)
    return out


def activity_inputs(s):
    result = {"EVENT_COUNT": np.ones(len(s.time))}
    for family, types in (
        ("ADD", ["A"]),
        ("CANCEL", ["X"]),
        ("EXECUTE", ["E", "P"]),
        ("VISIBLE_EXECUTE", ["E"]),
        ("HIDDEN_EXECUTE", ["P"]),
    ):
        mask = np.isin(s.kind, types)
        result[f"{family}_COUNT"] = mask.astype(float)
        result[f"{family}_SIZE"] = np.where(mask, s.size, 0.0)
        if family in {"ADD", "CANCEL", "EXECUTE"}:
            for side, code in (("BID", "B"), ("ASK", "S")):
                sided = mask & (s.side == code)
                result[f"{family}_{side}_SIDE_COUNT"] = sided.astype(float)
                result[f"{family}_{side}_SIDE_SIZE"] = np.where(sided, s.size, 0.0)
    return result


def previous_age(s, valid, qualifies):
    index = np.arange(len(s.time))
    last = np.maximum.accumulate(np.where(qualifies & valid, index, -1))
    previous = np.r_[-1, last[:-1]]
    last_invalid = np.maximum.accumulate(np.where(~valid, index, -1))
    good = valid & (previous >= 0) & (previous > last_invalid)
    return np.where(good, (s.time - s.time[np.maximum(previous, 0)]) / 1000000, np.nan)
