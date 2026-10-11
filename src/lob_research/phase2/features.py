"""Causal feature engine, entirely independent of label availability."""

import numpy as np

from .common import historical_indices, safe_ratio, suffix, window_sum
from .order_flow import activity_inputs, ofi_event, previous_age
from .static_book import static_features


def build_features(s, cfg):
    s.validate()
    output, metadata = {}, {}

    def emit(name, values, formula, fields, lookback, unit, missing):
        output[name] = values
        metadata[name] = {
            "name": name,
            "formula": formula,
            "required_fields": fields,
            "lookback_ms": lookback,
            "unit": unit,
            "timestamp_semantics": "post-event (timestamp, source_order); prefix only",
            "missing_rule": missing,
        }

    valid, static = static_features(s, cfg, emit)
    output.update(
        book_valid=valid,
        feature_static_available=static,
        tick_size_available=np.full(len(s.time), cfg["tick_size_usd"] is not None),
    )
    ofi = ofi_event(s, valid)
    emit(
        "OFI_EVENT",
        ofi,
        "Cont-style consecutive BBO increment; see FEATURE_DEFINITIONS.md",
        ["current and immediately preceding BBO"],
        0,
        "shares",
        "first row or invalid transition -> null",
    )
    activity = activity_inputs(s)
    for window in cfg["history_windows_ms"]:
        tag = suffix(window)
        j, available = historical_indices(s, valid, window)
        available &= static
        output[f"feature_{tag.lower()}_available"] = available
        missing = "full observed window and uninterrupted valid book required; else null"
        emit(
            f"OFI_{tag}",
            np.where(available, window_sum(ofi, s.time, window), np.nan),
            "sum OFI_EVENT in (t-w,t], source-row capped",
            ["OFI_EVENT", "timestamp", "source_order"],
            window,
            "shares",
            missing,
        )
        for name, values in activity.items():
            emit(
                f"{name}_{tag}",
                np.where(available, window_sum(values, s.time, window), np.nan),
                f"sum {name} indicators/quantities in (t-w,t]",
                ["event_type", "resting side", "size", "causal key"],
                window,
                "shares" if "SIZE" in name else "events",
                missing,
            )
        for rate, count in (
            ("EVENT", "EVENT"),
            ("ADD", "ADD"),
            ("CANCEL", "CANCEL"),
            ("EXECUTION", "EXECUTE"),
        ):
            emit(
                f"{rate}_RATE_{tag}",
                output[f"{count}_COUNT_{tag}"] / (window / 1000),
                f"{count}_COUNT_{tag}/window_seconds",
                [f"{count}_COUNT_{tag}"],
                window,
                "events/second",
                missing,
            )
        hist = output["MIDPRICE"][np.maximum(j, 0)]
        emit(
            f"MID_RETURN_{tag}",
            np.where(available, safe_ratio(output["MIDPRICE"], hist) - 1, np.nan),
            "current_mid / mid_asof(t-w) - 1",
            ["current mid", "as-of historical mid"],
            window,
            "return",
            missing,
        )
        if window == 1000:
            for feature, source in (("SPREAD_CHANGE_1S", "SPREAD"), ("OBI_CHANGE_1S", "OBI_L1")):
                emit(
                    feature,
                    np.where(available, output[source] - output[source][np.maximum(j, 0)], np.nan),
                    f"current {source} - {source} asof(t-1s)",
                    [source, "causal key"],
                    window,
                    "USD" if source == "SPREAD" else "ratio",
                    missing,
                )
    price_change = np.r_[
        False,
        ((s.bid[1:, 0] != s.bid[:-1, 0]) | (s.ask[1:, 0] != s.ask[:-1, 0]))
        & valid[1:]
        & valid[:-1],
    ]
    for name, qualifies in (
        ("EVENT", np.ones(len(s.time), bool)),
        ("TRADE", np.isin(s.kind, ["E", "P"])),
        ("BOOK_PRICE_CHANGE", price_change),
    ):
        emit(
            f"TIME_SINCE_PREVIOUS_{name}",
            previous_age(s, valid, qualifies),
            f"t - strictly prior {name} timestamp",
            ["causal key", "event type or BBO prices"],
            None,
            "milliseconds",
            "no prior qualifying event in current valid segment -> null",
        )
    return output, metadata
