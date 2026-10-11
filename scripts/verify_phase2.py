"""Reload and independently audit the frozen matrix; no predictive calculations.

Scalar reference arithmetic and bisect lookups below do not call production
feature/label functions. Those functions are used only for mutation experiments.
"""

import bisect
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lob_research.download import sha256
from lob_research.phase2.artifacts import DOCS, code_hash, configuration, upstream_check, write_csv
from lob_research.phase2.common import load_stream, suffix
from lob_research.phase2.features import build_features
from lob_research.phase2.labels import build_labels
from lob_research.timestamps import iso_utc


def scalar_ofi(s, i):
    if i == 0:
        return float("nan")
    b, a, qb, qa = map(float, (s.bid[i, 0], s.ask[i, 0], s.bsize[i, 0], s.asize[i, 0]))
    bp, ap, qbp, qap = map(
        float, (s.bid[i - 1, 0], s.ask[i - 1, 0], s.bsize[i - 1, 0], s.asize[i - 1, 0])
    )
    bid_term = qb if b > bp else -qbp if b < bp else qb - qbp
    ask_term = -qa if a < ap else qap if a > ap else qap - qa
    return bid_term + ask_term


def main():
    started = time.perf_counter()
    cfg = configuration()
    p = cfg["phase2"]
    report_dir = Path(p["reports_dir"])
    manifest = json.loads(Path("PHASE2_MANIFEST.json").read_text())
    checks = []

    def check(name, violations, detail=""):
        count = (
            int(np.count_nonzero(violations))
            if isinstance(violations, np.ndarray)
            else int(violations)
        )
        checks.append({"check": name, "passed": count == 0, "violations": count, "detail": detail})

    def equal(name, actual, expected, detail="", approximate=False):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            check(name, 1, f"shape mismatch {actual.shape} != {expected.shape}")
        elif approximate:
            check(
                name, ~np.isclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True), detail
            )
        elif actual.dtype.kind == "f" or expected.dtype.kind == "f":
            check(name, ~((actual == expected) | (np.isnan(actual) & np.isnan(expected))), detail)
        else:
            check(name, actual != expected, detail)

    frozen = upstream_check()
    check(
        "upstream_freeze",
        0,
        f"{len(frozen['files'])} baseline files; only recorded config/README/gitignore edits allowed",
    )
    prereg = json.loads((report_dir / "protocol_freeze.json").read_text())
    for doc in DOCS:
        check(f"preregistered_{doc}", sha256(doc) != prereg["documents"][doc])
    check(
        "configuration_hash",
        sha256("config.yaml") != manifest["config_sha256"]
        or sha256("config.yaml") != prereg["config_sha256"],
    )
    check("production_code_hash", code_hash() != manifest["code_sha256"])
    check("output_sha256", sha256(manifest["output_path"]) != manifest["output_sha256"])
    status = json.loads((report_dir / "build_status.json").read_text())
    check(
        "complete_build",
        status["status"] != "complete"
        or status["manifest_sha256"] != sha256("PHASE2_MANIFEST.json"),
    )
    for key, digest in manifest["input_hashes"].items():
        check(f"input_hash_{key}", sha256(p[key]) != digest)
    table = pq.read_table(manifest["output_path"])
    s = load_stream(cfg)
    n = len(s.time)

    def arr(name):
        col = table[name]
        if pa.types.is_timestamp(col.type):
            return col.cast(pa.int64()).to_numpy()
        return col.to_numpy()

    check("row_count", table.num_rows != n or n != manifest["row_count"])
    check("column_count", table.num_columns != manifest["column_count"])
    check("column_names", table.column_names != manifest["columns"])
    equal("source_orders_match_phase1", arr("source_order"), s.order)
    equal("timestamps_match_phase1", arr("timestamp"), s.time)
    equal("event_index", arr("event_index"), np.arange(n))
    equal("event_types_match_phase1", arr("event_type"), s.kind)
    equal("event_sides_match_phase1", arr("resting_side"), s.side)
    equal("event_sizes_match_phase1", arr("event_size"), s.size)
    check("strict_source_order", np.diff(arr("source_order")) <= 0)
    check("nondecreasing_timestamps", np.diff(s.time) < 0)
    check("unique_causal_keys", len(set(zip(s.time.tolist(), s.order.tolist()))) != n)
    check("session_containment", (s.time < s.start) | (s.time >= s.end))
    check(
        "one_symbol_and_session",
        set(arr("symbol")) != {s.symbol} or set(arr("session_date")) != {s.date},
    )
    lineage = json.loads(table.schema.metadata[b"lineage"])
    check("embedded_lineage", any(lineage[k] != manifest[k] for k in lineage))
    for name, expected in (
        ("source_raw_hash", manifest["raw_sha256"]),
        ("source_books_hash", manifest["input_hashes"]["input_books"]),
        ("source_events_hash", manifest["input_hashes"]["input_events"]),
        ("feature_version", p["feature_version"]),
        ("label_version", p["label_version"]),
    ):
        check(f"row_lineage_{name}", arr(name) != expected)
    for side, prices, sizes in (("bid", s.bid, s.bsize), ("ask", s.ask, s.asize)):
        equal(f"bbo_{side}_price", arr(f"best_{side}_price"), prices[:, 0] / 10000)
        equal(f"bbo_{side}_size", arr(f"best_{side}_size"), sizes[:, 0])

    # Reconstruct validity directly from the Phase-1 arrays, without valid_depth().
    valid = s.eligible & np.isfinite(s.bid[:, 0]) & np.isfinite(s.ask[:, 0])
    valid &= (s.bid[:, 0] > 0) & (s.ask[:, 0] > s.bid[:, 0])
    valid &= (
        np.isfinite(s.bsize[:, 0])
        & np.isfinite(s.asize[:, 0])
        & (s.bsize[:, 0] > 0)
        & (s.asize[:, 0] > 0)
    )
    static = valid.copy()
    for prices, sizes, sign in ((s.bid, s.bsize, -1), (s.ask, s.asize, 1)):
        static &= np.all(
            np.isfinite(prices) & np.isfinite(sizes) & (prices > 0) & (sizes > 0), axis=1
        )
        static &= np.all(sign * np.diff(prices, axis=1) > 0, axis=1)
    equal("book_valid_independent", arr("book_valid"), valid)
    equal("static_available_independent", arr("feature_static_available"), static)
    sums = (s.bid[:, 0] + s.ask[:, 0]).astype(np.int64)
    equal("mid_integer_encoding", arr("mid_sum_price_units"), sums)
    equal("midprice", arr("MIDPRICE"), sums / 20000)
    equal("midprice_alias", arr("midprice"), arr("MIDPRICE"))
    equal("spread_alias", arr("spread"), arr("SPREAD"))
    metadata = json.loads((report_dir / "feature_metadata.json").read_text())
    check("feature_metadata_names", list(metadata) != manifest["feature_list"])
    for name, meta in metadata.items():
        check(
            f"metadata_{name}",
            any(
                key not in meta
                for key in (
                    "name",
                    "formula",
                    "required_fields",
                    "lookback_ms",
                    "unit",
                    "timestamp_semantics",
                    "missing_rule",
                )
            ),
        )
        values = arr(name)
        check(f"finite_{name}", np.isinf(values))
        if name.startswith(("OBI_L", "WOBI_")):
            check(f"bounds_{name}", np.isfinite(values) & ((values < -1) | (values > 1)))
        if (
            name.startswith(("BID_DEPTH", "ASK_DEPTH", "TOTAL_DEPTH", "DEPTH_RATIO", "TIME_SINCE"))
            or name in ("SPREAD", "RELATIVE_SPREAD")
            or "_COUNT_" in name
            or "_SIZE_" in name
            or "_RATE_" in name
        ):
            check(f"nonnegative_{name}", values < 0)
        if meta["lookback_ms"] in p["history_windows_ms"]:
            equal(
                f"availability_{name}",
                np.isfinite(values),
                arr(f"feature_{suffix(meta['lookback_ms']).lower()}_available"),
            )
    check(
        "microprice_inside_bbo",
        valid
        & (
            (arr("MICROPRICE") < s.bid[:, 0] / 10000 - 1e-12)
            | (arr("MICROPRICE") > s.ask[:, 0] / 10000 + 1e-12)
        ),
    )
    check("no_next_feature", any("NEXT" in name.upper() for name in manifest["feature_list"]))
    check("unknown_tick_flag", arr("tick_size_available"))
    check("unknown_tick_microprice", table["MICROPRICE_DEVIATION_TICKS"].null_count != n)
    times = s.time.tolist()
    invalid_prefix = np.r_[0, np.cumsum(~valid)]
    for window in p["history_windows_ms"]:
        historical = np.array([bisect.bisect_right(times, t - window * 1000000) - 1 for t in times])
        available = (
            (historical >= 0)
            & static
            & (invalid_prefix[np.arange(n) + 1] == invalid_prefix[np.maximum(historical, 0)])
        )
        equal(
            f"history_asof_availability_{window}",
            arr(f"feature_{suffix(window).lower()}_available"),
            available,
        )
        expected = np.where(available, sums / sums[np.maximum(historical, 0)] - 1, np.nan)
        equal(
            f"history_asof_return_{window}",
            arr(f"MID_RETURN_{suffix(window)}"),
            expected,
            approximate=True,
        )

    # Independent per-row wall-clock resolver: Python bisect, not the label engine.
    for h in sorted([p["primary_horizon_ms"], *p["secondary_horizons_ms"]]):
        tag, flag = suffix(h), suffix(h).lower()
        tau = s.time + h * 1000000
        j = np.array([bisect.bisect_right(times, int(t)) - 1 for t in tau])
        good = valid & (tau < s.end) & (invalid_prefix[j + 1] == invalid_prefix[np.arange(n)])
        equal(f"target_availability_{flag}", arr(f"target_available_{flag}"), good)
        equal(f"requested_time_{flag}", arr(f"target_time_requested_{flag}"), tau)
        for name, expected in (
            (f"target_state_timestamp_{flag}", s.time[j]),
            (f"target_state_source_order_{flag}", s.order[j]),
        ):
            # Keep nanoseconds in int64: never compare float-converted epoch times.
            col = table[name]
            equal(f"target_nullmask_{name}", col.is_null().to_numpy(), ~good)
            actual = col.cast(pa.int64()).fill_null(0).to_numpy()
            equal(f"target_resolution_{name}", actual[good], expected[good])
        expected_direction = np.where(
            sums[j] > sums, "UP", np.where(sums[j] < sums, "DOWN", "FLAT")
        ).astype(object)
        expected_direction[~good] = None
        equal(f"direction_exact_{flag}", arr(f"DIRECTION_{tag}"), expected_direction)
        for name, expected in (
            (f"target_midprice_{flag}", sums[j] / 20000),
            (f"FWD_MID_RET_{tag}", sums[j] / sums - 1),
            (f"MID_MOVE_{tag}", (sums[j] - sums) / 20000),
            (f"target_age_ms_{flag}", (tau - s.time[j]) / 1000000),
        ):
            equal(f"label_arithmetic_{name}", arr(name), np.where(good, expected, np.nan))
        check(f"nonnegative_target_age_{flag}", arr(f"target_age_ms_{flag}") < 0)
        check(f"unknown_tick_label_{flag}", table[f"MID_MOVE_TICKS_{tag}"].null_count != n)
        expected_reason = np.where(
            ~valid,
            "current_book_invalid",
            np.where(
                tau >= s.end,
                "session_boundary",
                np.where(
                    invalid_prefix[j + 1] != invalid_prefix[np.arange(n)],
                    "invalid_book_interval",
                    "available",
                ),
            ),
        )
        equal(f"target_reason_{flag}", arr(f"target_unavailable_reason_{flag}"), expected_reason)

    # Deterministic descriptive trace selection; no feature/label association used.
    indices = set(
        np.minimum(
            n - 1,
            np.searchsorted(
                s.time, np.linspace(s.time[0] + 11000000000, s.time[-1], 20).astype(np.int64)
            ),
        ).tolist()
    )
    counts = arr("EVENT_COUNT_1S")
    high, low = int(np.nanargmax(counts)), int(np.nanargmin(counts))
    ties = np.flatnonzero(np.diff(s.time) == 0)
    tie = int(ties[len(ties) // 2])
    indices.update((high, low, tie, tie + 1, n // 2, n - 1))
    selected = sorted(indices)
    comparisons, traces = (
        [],
        [
            "# Phase 2 real alignment traces",
            "",
            "Local restricted derivatives; do not redistribute. Deterministic time-spaced coverage plus activity extremes, midpoint, timestamp tie and session end. All prices USD; sizes shares; times UTC.",
            "",
        ],
    )

    def spot(i, name, expected):
        actual = table[name][i].as_py()
        if isinstance(expected, (float, int, np.number)):
            matched = actual is not None and bool(
                np.isclose(actual, expected, rtol=1e-12, atol=1e-12)
            )
        else:
            matched = actual == expected
        comparisons.append(
            {
                "event_index": i,
                "timestamp": iso_utc(int(s.time[i])),
                "source_order": int(s.order[i]),
                "field": name,
                "independent_expected": expected,
                "stored_actual": actual,
                "passed": matched,
                "method": "scalar Phase-1 arithmetic / Python bisect; no production feature function",
            }
        )

    for i in selected:
        b, a, qb, qa = s.bid[i, 0], s.ask[i, 0], s.bsize[i, 0], s.asize[i, 0]
        for depth in (1, 3, 5, 10):
            bd = sum(float(x) for x in s.bsize[i, :depth])
            ad = sum(float(x) for x in s.asize[i, :depth])
            spot(i, f"OBI_L{depth}", (bd - ad) / (bd + ad))
        spot(i, "MICROPRICE", (a * qb + b * qa) / (qb + qa) / 10000)
        spot(i, "OFI_EVENT", scalar_ofi(s, i))
        left = bisect.bisect_right(times, int(s.time[i]) - 1000000000)
        spot(i, "EVENT_COUNT_1S", i - left + 1)
        spot(i, "OFI_1S", sum(scalar_ofi(s, k) for k in range(left, i + 1)))
        j = bisect.bisect_right(times, int(s.time[i]) + 1000000000) - 1
        available = s.time[i] + 1000000000 < s.end
        spot(i, "target_available_1s", bool(available))
        spot(i, "FWD_MID_RET_1S", float(sums[j] / sums[i] - 1) if available else None)
        spot(i, "MID_MOVE_1S", float((sums[j] - sums[i]) / 20000) if available else None)
        spot(
            i,
            "DIRECTION_1S",
            ("UP" if sums[j] > sums[i] else "DOWN" if sums[j] < sums[i] else "FLAT")
            if available
            else None,
        )
        if available:
            spot(i, "target_state_source_order_1s", int(s.order[j]))
        tags = [
            label
            for condition, label in (
                (i == high, "highest 1s event count"),
                (i == low, "lowest available 1s event count"),
                (i in (tie, tie + 1), "same-timestamp pair"),
                (i == n // 2, "midpoint"),
                (i == n - 1, "last session row"),
            )
            if condition
        ]
        obi_text = ", ".join(format(arr("OBI_L" + str(k))[i], ".12g") for k in (1, 3, 5, 10))
        traces.extend(
            [
                f"## Row {i}: {', '.join(tags) or 'time-spaced sample'}",
                "",
                f"Feature key: ({iso_utc(int(s.time[i]))}, {int(s.order[i])}); event {s.kind[i]}, resting side {s.side[i]}, size {s.size[i]:g}. Current state is post-event.",
                "",
                "| Level | Bid USD | Bid shares | Ask USD | Ask shares |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for depth in range(10):
            traces.append(
                f"| {depth + 1} | {s.bid[i, depth] / 10000:.4f} | {s.bsize[i, depth]:g} | {s.ask[i, depth] / 10000:.4f} | {s.asize[i, depth]:g} |"
            )
        traces.extend(
            [
                "",
                f"Previous BBO (source {s.order[i - 1]}): bid {s.bid[i - 1, 0] / 10000:.4f} x {s.bsize[i - 1, 0]:g}; ask {s.ask[i - 1, 0] / 10000:.4f} x {s.asize[i - 1, 0]:g}.",
                f"OBI L1/L3/L5/L10: {obi_text}. Microprice: {arr('MICROPRICE')[i]:.12g}. OFI_EVENT: {arr('OFI_EVENT')[i]:g}.",
                f"1s event interval: ({iso_utc(int(s.time[i]) - 1000000000)}, current key]; rows {left}..{i}, source orders {s.order[left]}..{s.order[i]}. Exact scalar OFI sum: {sum(scalar_ofi(s, k) for k in range(left, i + 1)):g}.",
                "",
                "| Window | OFI shares | Event count | Add count | Cancel count | Execute count | Event rate /s |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for window in p["history_windows_ms"]:
            tag = suffix(window)
            traces.append(
                f"| {tag} | "
                + " | ".join(
                    str(table[f"{name}_{tag}"][i].as_py())
                    for name in (
                        "OFI",
                        "EVENT_COUNT",
                        "ADD_COUNT",
                        "CANCEL_COUNT",
                        "EXECUTE_COUNT",
                        "EVENT_RATE",
                    )
                )
                + " |"
            )
        traces.extend(
            [
                "",
                "| Horizon | Requested UTC | Available | Resolved UTC | Source order | Future mid | Direction | Age ms |",
                "|---|---|---|---|---:|---:|---|---:|",
            ]
        )
        for h in sorted([p["primary_horizon_ms"], *p["secondary_horizons_ms"]]):
            tag, flag = suffix(h), suffix(h).lower()
            good = bool(arr(f"target_available_{flag}")[i])
            target_index = bisect.bisect_right(times, int(s.time[i]) + h * 1000000) - 1
            resolved = (
                iso_utc(int(s.time[target_index])) if good else "unavailable: session boundary"
            )
            fields = [
                tag,
                iso_utc(int(s.time[i]) + h * 1000000),
                str(good),
                resolved,
                str(table[f"target_state_source_order_{flag}"][i].as_py()),
                str(table[f"target_midprice_{flag}"][i].as_py()),
                str(table[f"DIRECTION_{tag}"][i].as_py()),
                str(table[f"target_age_ms_{flag}"][i].as_py()),
            ]
            traces.append("| " + " | ".join(fields) + " |")
        traces.append("")
    write_csv(report_dir / "manual_verification.csv", comparisons)
    Path("reports/phase2_alignment_traces.md").write_text("\n".join(traces), encoding="utf-8")
    (report_dir / "trace_indices.json").write_text(
        json.dumps(
            {
                "indices": selected,
                "high_activity": high,
                "low_activity": low,
                "same_timestamp_pair": [tie, tie + 1],
            },
            indent=2,
        )
        + "\n"
    )
    check("real_trace_count", len(selected) < 20, str(len(selected)))
    check(
        "independent_manual_comparisons",
        sum(not row["passed"] for row in comparisons),
        f"{len(comparisons)} comparisons across {len(selected)} real rows",
    )

    # Rerun production only to test invariance under destructive future mutation.
    # Causal keys stay fixed; every future book, size, type and side is changed.
    cuts = {
        "future_mutation": bisect.bisect_right(times, times[n // 2]) - 1,
        "same_timestamp_mutation": tie,
    }
    mutations = {}
    for title, cut in cuts.items():
        changed = s.copy()
        changed.bid[cut + 1 :] += 10000
        changed.ask[cut + 1 :] += 20000
        changed.bsize[cut + 1 :] *= 3
        changed.asize[cut + 1 :] *= 2
        changed.size[cut + 1 :] *= 7
        changed.kind[cut + 1 :] = np.where(s.kind[cut + 1 :] == "A", "X", "A")
        changed.side[cut + 1 :] = np.where(s.side[cut + 1 :] == "B", "S", "B")
        altered, _ = build_features(changed, p)
        first_check = len(checks)
        for name, values in altered.items():
            equal(
                f"{title}_{name}",
                values[: cut + 1],
                arr(name)[: cut + 1],
                "exact prefix equality, NaN masks included",
            )
        del altered
        gc.collect()
        original_labels, _ = build_labels(s, p)
        altered_labels, _ = build_labels(changed, p)
        settled_counts = {}
        for h in sorted([p["primary_horizon_ms"], *p["secondary_horizons_ms"]]):
            flag = suffix(h).lower()
            settled = (original_labels[f"target_time_requested_{flag}"] <= s.time[cut]) & (
                original_labels[f"target_state_source_order_{flag}"][0] <= s.order[cut]
            )
            settled_counts[flag] = int(settled.sum())
            for name, original in original_labels.items():
                if name.lower().endswith("_" + flag):
                    altered_value = altered_labels[name]
                    if isinstance(original, tuple):
                        for component, (a, b) in enumerate(zip(original, altered_value)):
                            equal(f"{title}_settled_{name}_{component}", a[settled], b[settled])
                    else:
                        equal(f"{title}_settled_{name}", original[settled], altered_value[settled])
        mutations[title] = {
            "cut_event_index": cut,
            "cut_timestamp": iso_utc(int(s.time[cut])),
            "cut_source_order": int(s.order[cut]),
            "mutated_rows": n - cut - 1,
            "unchanged_prefix_rows": cut + 1,
            "feature_and_flag_columns": len(manifest["feature_list"])
            + len(manifest["availability_flags"]),
            "settled_label_counts": settled_counts,
            "passed": all(row["passed"] for row in checks[first_check:]),
        }
        del changed, original_labels, altered_labels
        gc.collect()

    write_csv(report_dir / "invariant_checks.csv", checks)
    failed = [row for row in checks if not row["passed"]]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "check_count": len(checks),
        "failed_checks": failed,
        "violation_count": sum(row["violations"] for row in checks),
        "manual_trace_rows": len(selected),
        "manual_comparisons": len(comparisons),
        "manual_failed_comparisons": sum(not row["passed"] for row in comparisons),
        "mutations": mutations,
        "output_sha256": sha256(manifest["output_path"]),
        "verifier_sha256": sha256(__file__),
        "runtime_seconds": time.perf_counter() - started,
    }
    (report_dir / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
