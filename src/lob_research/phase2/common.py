"""Validated single-session arrays and integer-time lookup primitives."""

from dataclasses import dataclass, replace

import numpy as np
import pyarrow.parquet as pq

from lob_research.timestamps import clock_ms, day_epoch_ns


def suffix(ms):
    return f"{ms // 1000}S" if ms % 1000 == 0 else f"{ms}MS"


@dataclass
class Stream:
    time: np.ndarray
    order: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bsize: np.ndarray
    asize: np.ndarray
    kind: np.ndarray
    side: np.ndarray
    size: np.ndarray
    eligible: np.ndarray
    start: int
    end: int
    date: str = "2003-01-03"
    symbol: str = "MSFT"

    def validate(self):
        n = len(self.time)
        if not n or any(
            len(getattr(self, name)) != n
            for name in (
                "order",
                "bid",
                "ask",
                "bsize",
                "asize",
                "kind",
                "side",
                "size",
                "eligible",
            )
        ):
            raise ValueError("empty or unaligned stream")
        if np.any(np.diff(self.time) < 0) or np.any(np.diff(self.order) <= 0):
            raise ValueError("unordered input: require source order, never timestamp-only sorting")
        if np.any(self.time < self.start) or np.any(self.time >= self.end):
            raise ValueError("cross-session input is forbidden")
        if self.bid.ndim != 2 or not (
            self.bid.shape == self.ask.shape == self.bsize.shape == self.asize.shape
        ):
            raise ValueError("inconsistent depth arrays")
        if not set(self.kind) <= {"A", "X", "E", "P", "B"}:
            raise ValueError("unverified event semantics")
        active = self.kind != "B"
        if (
            np.any(~np.isin(self.side[active], ["B", "S"]))
            or np.any(~np.isfinite(self.size[active]))
            or np.any(self.size[active] <= 0)
        ):
            raise ValueError("invalid event side or quantity")
        for arr in (self.bid, self.ask, self.bsize, self.asize):
            finite = np.isfinite(arr)
            if np.any(arr[finite] != np.floor(arr[finite])):
                raise ValueError("price/size source units must be integers")
        return self

    def copy(self):
        return replace(
            self,
            **{
                name: value.copy()
                for name, value in vars(self).items()
                if isinstance(value, np.ndarray)
            },
        )

    def take(self, indices):
        return replace(
            self,
            **{
                name: value[indices].copy()
                for name, value in vars(self).items()
                if isinstance(value, np.ndarray)
            },
        )


def load_stream(cfg):
    p = cfg["phase2"]
    book = pq.read_table(p["input_books"])
    events = pq.read_table(p["input_events"])
    for name in ("source_row", "timestamp", "event_type"):
        if not book[name].equals(events[name]):
            raise ValueError(f"Phase 1 alignment differs: {name}")
    if set(events["symbol"].to_pylist()) != {cfg["symbol"]}:
        raise ValueError("multiple or incorrect symbols")
    depth = cfg["depth"]

    def matrix(side, field):
        return np.column_stack(
            [book[f"{side}_{field}_{k}"].to_numpy().astype(float) for k in range(1, depth + 1)]
        )

    epoch = day_epoch_ns(cfg["date"], cfg["timezone"])
    return Stream(
        book["timestamp"].cast("int64").to_numpy(),
        book["source_row"].to_numpy(),
        matrix("bid", "price_units"),
        matrix("ask", "price_units"),
        matrix("bid", "size"),
        matrix("ask", "size"),
        np.asarray(events["event_type"].to_pylist()),
        np.asarray(events["side"].to_pylist()),
        events["size"].to_numpy().astype(float),
        book["research_eligible"].to_numpy(),
        epoch + clock_ms(cfg["session_start"]) * 1000000,
        epoch + clock_ms(cfg["session_end"]) * 1000000,
        cfg["date"],
        cfg["symbol"],
    ).validate()


def safe_ratio(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=np.isfinite(b) & (b != 0))


def valid_depth(s, k):
    if s.bid.shape[1] < k:
        return np.zeros(len(s.time), dtype=bool)
    result = s.eligible.copy()
    for prices, sizes, sign in (
        (s.bid[:, :k], s.bsize[:, :k], -1),
        (s.ask[:, :k], s.asize[:, :k], 1),
    ):
        result &= np.all(
            np.isfinite(prices) & (prices > 0) & np.isfinite(sizes) & (sizes > 0), axis=1
        )
        result &= np.all(sign * np.diff(prices, axis=1) > 0, axis=1)
    return result & (s.bid[:, 0] < s.ask[:, 0])


def historical_indices(s, valid, window_ms):
    cutoff = s.time - window_ms * 1000000
    j = np.searchsorted(s.time, cutoff, side="right") - 1
    bad = np.r_[0, np.cumsum(~valid)]
    available = (j >= 0) & (cutoff >= s.time[0])
    available &= bad[np.arange(len(s.time)) + 1] - bad[np.maximum(j, 0)] == 0
    return j, available


def window_sum(values, times, window_ms):
    """Left timestamp-exclusive; right source-row-inclusive, not timestamp-group-inclusive."""
    left = np.searchsorted(times, times - window_ms * 1000000, side="right")
    prefix = np.r_[0.0, np.cumsum(np.nan_to_num(values, nan=0.0))]
    return prefix[np.arange(len(times)) + 1] - prefix[left]
