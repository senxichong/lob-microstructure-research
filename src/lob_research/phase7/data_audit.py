import bisect
import hashlib
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .common import CHECKS, LOCAL, OUT, arrays, close, dump, js, record, sha, write


def integrity():
    rows = []

    def verify(p, h, origin, prefix=False):
        content = Path(p).read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        status = "PASS"
        used = len(content)
        if actual != h and prefix:
            candidates = [i + 1 for i, b in enumerate(content) if b == 10]
            found = next(
                (i for i in candidates if hashlib.sha256(content[:i]).hexdigest() == h), None
            )
            if found is not None:
                actual = h
                used = found
                status = "PASS_APPEND_ONLY_PREFIX"
        if actual != h:
            status = "FAIL"
        rows.append(
            dict(
                file=p,
                manifest=origin,
                expected_hash=h,
                actual_hash=actual,
                status=status,
                verified_bytes=used,
            )
        )

    for phase in [4, 5, 6]:
        m = js(f"PHASE{phase}_MANIFEST.json")
        for p, h in m["evidence_sha256"].items():
            verify(p, h, f"PHASE{phase}_MANIFEST.json", p in ["RESEARCH_LOG.md", ".gitignore"])
    for origin in ["reports/phase3/final_evidence.json", "reports/phase6/upstream_freeze.json"]:
        m = js(origin)
        for p, h in m.get("evidence_sha256", m.get("files", {})).items():
            verify(p, h, origin, p in ["RESEARCH_LOG.md", ".gitignore"])
    for phase in [2, 3]:
        m = js(f"PHASE{phase}_MANIFEST.json")
        verify(m["output_path"], m["output_sha256"], f"PHASE{phase}_MANIFEST.json")
        for p, h in m.get("csv_sha256", {}).items():
            verify(p, h, f"PHASE{phase}_MANIFEST.json")
        verify(
            "config.yaml" if phase == 2 else "config.phase3.yaml",
            m["config_sha256"],
            f"PHASE{phase}_MANIFEST.json",
        )
        if phase == 2:
            verify("PHASE2_PROTOCOL.md", m["protocol_sha256"], "PHASE2_MANIFEST.json")
        else:
            verify(
                "reports/phase3/protocol_freeze.json",
                m["protocol_freeze_sha256"],
                "PHASE3_MANIFEST.json",
            )
        digest = hashlib.sha256()
        paths = sorted(Path(f"src/lob_research/phase{phase}").glob("*.py")) + [
            Path("scripts/build_phase2.py" if phase == 2 else "scripts/run_phase3.py")
        ]
        for p in paths:
            archived = (
                Path("reports/phase3/run_phase3_at_execution.py")
                if phase == 3 and p.name == "run_phase3.py"
                else p
            )
            digest.update(p.as_posix().encode() + b"\0" + archived.read_bytes())
        rows.append(
            dict(
                file=f"Phase {phase} executed code composite",
                manifest=f"PHASE{phase}_MANIFEST.json",
                expected_hash=m["code_sha256"],
                actual_hash=digest.hexdigest(),
                status="PASS" if digest.hexdigest() == m["code_sha256"] else "FAIL",
                verified_bytes=sum(p.stat().st_size for p in paths),
            )
        )
    for phase in range(2, 7):
        origin = f"reports/phase{phase}/protocol_freeze.json"
        m = js(origin)
        for p, h in m.get("documents", {}).items():
            verify(p, h, origin)
        if "log_hash" in m:
            verify("RESEARCH_LOG.md", m["log_hash"], origin, True)
    for phase in [4, 5, 6]:
        origin = f"reports/phase{phase}/execution_manifest.json"
        m = js(origin)
        for p, h in m.get("code_sha256", {}).items():
            if phase == 4 and p == "src/lob_research/phase4/core.py":
                p = "reports/phase4/core_at_execution.py"
            verify(p, h, origin)
    for r in js("DATA_MANIFEST.json")["artifacts"]:
        verify(r["filename"], r["sha256"], "DATA_MANIFEST.json")
    lineage = js("data/processed/lineage.json")
    for r in lineage["artifacts"].values():
        verify(r["path"], r["sha256"], "data/processed/lineage.json")
    for r in js("reports/phase5/model_artifacts.json"):
        verify(r["path"], r["sha256"], "reports/phase5/model_artifacts.json")
    write("artifact_integrity", rows)
    record(
        "integrity",
        "all authoritative file hashes",
        all(r["status"].startswith("PASS") for r in rows),
        detail=f"{len(rows)} checks",
    )
    if any(r["status"] == "FAIL" for r in rows):
        raise ValueError("unexpected frozen artifact mutation")
    # Freeze the complete starting state, including every document/script/test read.
    inventory = []
    for root in ["data", "reports", "artifacts", "src", "scripts", "tests"]:
        for p in Path(root).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts or "phase7" in p.as_posix():
                continue
            row = dict(
                path=p.as_posix(), bytes=p.stat().st_size, sha256=sha(p), inspection="bytes hashed"
            )
            if p.suffix == ".json":
                js(p)
                row["inspection"] = "JSON parsed"
            elif p.suffix == ".csv":
                import csv

                with p.open(encoding="utf-8-sig", newline="") as f:
                    row["rows"] = sum(1 for _ in csv.DictReader(f))
                row["inspection"] = "all CSV records parsed"
            elif p.suffix == ".parquet":
                pf = pq.ParquetFile(p)
                row["rows"] = sum(batch.num_rows for batch in pf.iter_batches())
                row["inspection"] = "all Parquet batches decoded"
            elif p.suffix in [".py", ".md", ".yaml", ".txt"]:
                text = p.read_text(encoding="utf-8")
                row["lines"] = len(text.splitlines())
                row["inspection"] = "full text read and indexed"
                if p.suffix == ".py":
                    import ast

                    ast.parse(text)
                    row["inspection"] = "full Python AST parsed and indexed"
            inventory.append(row)
    for p in Path(".").glob("*.md"):
        if "PHASE7" not in p.name:
            text = p.read_text(encoding="utf-8")
            inventory.append(
                dict(
                    path=p.as_posix(),
                    bytes=p.stat().st_size,
                    sha256=sha(p),
                    inspection="full document read",
                    lines=len(text.splitlines()),
                )
            )
    write("review_inventory", inventory)
    snapshot = {r["path"]: r["sha256"] for r in inventory}
    snapshot.update(
        {f"PHASE{i}_MANIFEST.json": sha(f"PHASE{i}_MANIFEST.json") for i in range(2, 7)}
    )
    snapshot[".gitignore"] = sha(".gitignore")
    if not (OUT / "upstream_freeze.json").exists():
        dump(
            "upstream_freeze.json",
            dict(files=snapshot, protocol_sha256=sha("PHASE7_AUDIT_PROTOCOL.md")),
        )
    return lineage


def raw_replay(lineage):
    cfg = yaml.safe_load(Path("config.phase1.yaml").read_text())
    books = arrays(lineage["artifacts"]["books"]["path"])
    events = arrays(lineage["artifacts"]["events"]["path"])
    columns = [
        f"{s}_{v}_{k}"
        for s in ["bid", "ask"]
        for k in range(1, 11)
        for v in ["price_units", "size"]
    ]
    reconstructed = np.empty((len(books["timestamp"]), 40), dtype=np.int64)
    orders = {}
    levels = {"B": {}, "S": {}}
    prices = {"B": [], "S": []}
    j = 0
    types = {}
    counts = {}
    windows = []
    member_hash = hashlib.sha256()
    last_add = 0
    last_time = -1
    empty = 0

    def update(side, price, quantity):
        before = levels[side].get(price, 0)
        after = before + quantity
        if after < 0:
            raise ValueError("negative reconstructed queue")
        if after == 0:
            del levels[side][price]
            prices[side].remove(price)
        else:
            if before == 0:
                bisect.insort(prices[side], price)
            levels[side][price] = after

    with zipfile.ZipFile(cfg["raw_path"]) as z:
        with z.open(lineage["member"]) as f:
            for row, line in enumerate(f, 1):
                member_hash.update(line)
                raw = line.rstrip(b"\r\n")
                if not raw:
                    empty += 1
                    continue
                typ = chr(raw[8])
                ms = int(raw[:8])
                types[typ] = types.get(typ, 0) + 1
                if ms < last_time:
                    raise ValueError("raw clock reversal")
                last_time = ms
                selected = False
                size = 0
                side = ""
                price = 0
                if typ in ["A", "P"]:
                    oid = int(raw[9:18])
                    selected = raw[25:31].strip() == b"MSFT"
                    if typ == "A":
                        if oid <= last_add:
                            raise ValueError("nonincreasing add reference")
                        last_add = oid
                    if selected:
                        side = chr(raw[18])
                        size = int(raw[19:25])
                        price = int(raw[31:41])
                        if typ == "A":
                            if raw[41:42] != b"Y" or oid in orders:
                                raise ValueError("bad selected add")
                            orders[oid] = (side, price, size)
                            update(side, price, size)
                elif typ in ["X", "E"]:
                    oid = int(raw[9:18])
                    selected = oid in orders
                    if selected:
                        side, price, remaining = orders[oid]
                        size = int(raw[18:24])
                        if size <= 0 or size > remaining:
                            raise ValueError("over-reduction")
                        update(side, price, -size)
                        if remaining == size:
                            del orders[oid]
                        else:
                            orders[oid] = (side, price, remaining - size)
                        counts["full_removal" if remaining == size else "partial_reduction"] = (
                            counts.get(
                                "full_removal" if remaining == size else "partial_reduction", 0
                            )
                            + 1
                        )
                elif typ not in ["S", "B"]:
                    raise ValueError("unsupported message")
                if selected and 34200000 <= ms < 57600000:
                    if int(books["source_row"][j]) != row or events["event_type"][j] != typ:
                        raise ValueError("raw/saved row mismatch")
                    counts[typ] = counts.get(typ, 0) + 1
                    vals = []
                    for s in ["B", "S"]:
                        keys = prices[s][-10:][::-1] if s == "B" else prices[s][:10]
                        if len(keys) < 10:
                            raise ValueError("unexpected missing depth")
                        for p in keys:
                            vals.extend([p, levels[s][p]])
                    reconstructed[j] = vals
                    if j % 30000 < 12:
                        windows.append(
                            dict(
                                event_index=j,
                                source_row=row,
                                event_type=typ,
                                source_time_ms=ms,
                                side=side,
                                event_size=size,
                                price_units=price,
                                best_bid=vals[0],
                                best_ask=vals[20],
                            )
                        )
                    j += 1
    close("raw", "retained states", j, len(books["timestamp"]), 0)
    record("raw", "member hash", member_hash.hexdigest() == lineage["member_sha256"])
    record("raw", "selected final orders exhausted", len(orders) == 0)
    for k, col in enumerate(columns):
        close("raw", col, reconstructed[:, k], books[col], 0)
    b = reconstructed[:, 0:20:2]
    a = reconstructed[:, 20::2]
    record("invariants", "strict uncrossed BBO", np.all(b[:, 0] < a[:, 0]))
    record("invariants", "positive prices/queues", np.all(reconstructed > 0))
    record(
        "invariants",
        "sorted unique levels",
        np.all(np.diff(b, axis=1) < 0) and np.all(np.diff(a, axis=1) > 0),
    )
    LOCAL.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(windows), LOCAL / "raw_windows.parquet")
    dump(
        "raw_reconstruction.json",
        dict(
            states=j,
            raw_messages=sum(types.values()),
            raw_types=types,
            selected_counts=counts,
            raw_window_rows=len(windows),
            deterministic_windows=6,
            price_scale=10000,
            size_unit="share",
            member_hash=member_hash.hexdigest(),
            max_depth_error=max(r["max_discrepancy"] for r in CHECKS if r["section"] == "raw"),
            unsupported_semantics="No standalone DELETE or REPLACE in ITCH 2.0; full X removes orders; P hidden print leaves book unchanged; no B messages observed",
            transport_completeness_proven=False,
        ),
    )
    return books, events


def independently_derive(books, events, end):
    t = books["timestamp"]
    n = len(t)
    b = np.column_stack([books[f"bid_price_units_{k}"] for k in range(1, 11)])
    a = np.column_stack([books[f"ask_price_units_{k}"] for k in range(1, 11)])
    qb = np.column_stack([books[f"bid_size_{k}"] for k in range(1, 11)]).astype(float)
    qa = np.column_stack([books[f"ask_size_{k}"] for k in range(1, 11)]).astype(float)
    sums = b[:, 0] + a[:, 0]
    mid = sums / 20000
    spread = (a[:, 0] - b[:, 0]) / 10000
    micro = (a[:, 0] / 10000 * qb[:, 0] + b[:, 0] / 10000 * qa[:, 0]) / (qb[:, 0] + qa[:, 0])
    out = dict(
        MIDPRICE=mid,
        SPREAD=spread,
        RELATIVE_SPREAD=spread / mid,
        MICROPRICE=micro,
        MICROPRICE_MINUS_MID=micro - mid,
    )
    for k in [1, 3, 5, 10]:
        out[f"OBI_L{k}"] = (qb[:, :k].sum(1) - qa[:, :k].sum(1)) / (
            qb[:, :k].sum(1) + qa[:, :k].sum(1)
        )
    out["WOBI_L10"] = ((qb - qa) / np.arange(1, 11)).sum(1) / ((qb + qa) / np.arange(1, 11)).sum(1)
    ofi = np.full(n, np.nan)
    for i in range(1, n):
        ofi[i] = (
            (qb[i, 0] if b[i, 0] >= b[i - 1, 0] else 0)
            - (qb[i - 1, 0] if b[i, 0] <= b[i - 1, 0] else 0)
            - (qa[i, 0] if a[i, 0] <= a[i - 1, 0] else 0)
            + (qa[i - 1, 0] if a[i, 0] >= a[i - 1, 0] else 0)
        )
    out["OFI_EVENT"] = ofi
    for h, tag in [(100, "100MS"), (1000, "1S"), (5000, "5S"), (10000, "10S")]:
        left = np.searchsorted(t, t - h * 10**6, side="right")
        hist = left - 1
        valid = hist >= 0
        cum = np.r_[0.0, np.cumsum(np.nan_to_num(ofi))]
        out["OFI_" + tag] = np.where(valid, cum[np.arange(n) + 1] - cum[left], np.nan)
        out["EVENT_COUNT_" + tag] = np.where(valid, np.arange(n) - left + 1, np.nan)
        out["EVENT_RATE_" + tag] = out["EVENT_COUNT_" + tag] / (h / 1000)
        for kind, types in [("ADD", ["A"]), ("CANCEL", ["X"]), ("EXECUTE", ["E", "P"])]:
            indicator = np.isin(events["event_type"], types)
            c = np.r_[0, np.cumsum(indicator)]
            out[kind + "_COUNT_" + tag] = np.where(valid, c[np.arange(n) + 1] - c[left], np.nan)
        out["MID_RETURN_" + tag] = np.where(valid, mid / mid[np.maximum(hist, 0)] - 1, np.nan)
        if h == 1000:
            out["SPREAD_CHANGE_1S"] = np.where(valid, spread - spread[np.maximum(hist, 0)], np.nan)
            out["OBI_CHANGE_1S"] = np.where(
                valid, out["OBI_L1"] - out["OBI_L1"][np.maximum(hist, 0)], np.nan
            )
        tau = t + h * 10**6
        dest = np.searchsorted(t, tau, side="right") - 1
        ok = tau < end
        out["FWD_MID_RET_" + tag] = np.where(ok, sums[dest] / sums - 1, np.nan)
        out["target_state_timestamp_" + tag.lower()] = np.where(ok, t[dest], 0)
    last = -1
    age = np.full(n, np.nan)
    for i in range(n):
        if last >= 0:
            age[i] = (t[i] - t[last]) / 1e6
        if events["event_type"][i] in ["E", "P"]:
            last = i
    out["TIME_SINCE_PREVIOUS_TRADE"] = age
    return out


def features_grid(books, events):
    p2 = arrays(js("PHASE2_MANIFEST.json")["output_path"])
    g = arrays(js("PHASE3_MANIFEST.json")["output_path"])
    cfg = yaml.safe_load(Path("config.phase3.yaml").read_text())
    end = (
        int(datetime.fromisoformat(cfg["session_end_utc"].replace("Z", "+00:00")).timestamp())
        * 10**9
    )
    derived = independently_derive(books, events, end)
    for k, v in derived.items():
        close("features", k, v, p2[k], 1e-10 if "RATE" in k else 1e-12)
    t = books["timestamp"]
    ties = np.flatnonzero(np.diff(t) == 0) + 1
    # Deliberately swap real same-time states while retaining source position keys.
    chosen = ties[np.linspace(0, len(ties) - 1, 8, dtype=int)]
    changed_counts = []
    for i in chosen:
        altered = {k: v.copy() for k, v in books.items() if not k == "issues"}
        for k in altered:
            if k not in ["timestamp", "source_row"]:
                altered[k][[i - 1, i]] = altered[k][[i, i - 1]]
        # Current OBI and microprice explicitly change only from this corrupted prefix.
        q1, q2 = altered["bid_size_1"][i - 1], altered["ask_size_1"][i - 1]
        value = (q1 - q2) / (q1 + q2)
        changed_counts.append(value != derived["OBI_L1"][i - 1])
    record(
        "causality",
        "same-timestamp corruption observable",
        any(changed_counts),
        detail=f"{sum(changed_counts)}/8 altered first-row OBI values",
    )
    # Multiple aggressive mutations of all future book/event payloads.
    rng = np.random.default_rng(707008)
    cutoffs = np.sort(rng.choice(np.arange(20000, len(t) - 20000), 5, replace=False))
    mutation = []
    for c in cutoffs:
        # A prefix rerun removes all future records, including later same-time rows.
        subset = {k: v[: c + 1] for k, v in books.items()}
        ev = {k: v[: c + 1] for k, v in events.items()}
        prefix = independently_derive(subset, ev, end)
        past_features = [k for k in derived if not k.startswith(("FWD_", "target_"))]
        for k in past_features:
            close("mutation", f"cut{c} drop_future {k}", prefix[k], derived[k][: c + 1])
        altered = {k: v.copy() for k, v in books.items()}
        for k in altered:
            if "_price_units_" in k:
                altered[k][c + 1 :] += rng.integers(100000, 200000)
            elif "_size_" in k:
                altered[k][c + 1 :] = rng.integers(1, 10000, size=len(t) - c - 1)
        ev2 = {k: v.copy() for k, v in events.items()}
        ev2["event_type"][c + 1 :] = "A"
        shifted = independently_derive(altered, ev2, end)
        for k in past_features:
            close(
                "mutation", f"cut{c} corrupt_future {k}", shifted[k][: c + 1], derived[k][: c + 1]
            )
        for h, tag in [(100, "100MS"), (1000, "1S"), (5000, "5S"), (10000, "10S")]:
            settled = t[: c + 1] + h * 10**6 < t[c]
            close(
                "mutation",
                f"cut{c} settled {tag}",
                shifted["FWD_MID_RET_" + tag][: c + 1][settled],
                derived["FWD_MID_RET_" + tag][: c + 1][settled],
            )
        mutation.append(
            dict(
                cutoff_index=int(c),
                cutoff_timestamp=int(t[c]),
                future_rows=len(t) - c - 1,
                drop_future_pass=True,
                corrupt_future_pass=True,
            )
        )
    write("future_mutations", mutation)
    i = np.searchsorted(t, g["grid_timestamp"], side="right") - 1
    safe = np.maximum(i, 0)
    valid = i >= 0
    close("grid", "23400 grid rows", len(i), 23400, 0)
    close("grid", "source timestamp", np.where(valid, t[safe], 0), g["source_state_timestamp"], 0)
    close(
        "grid",
        "source order",
        np.where(valid, books["source_row"][safe], -1),
        g["source_state_order"],
        0,
    )
    close(
        "grid",
        "state age",
        np.where(valid, (g["grid_timestamp"] - t[safe]) / 1e6, np.nan),
        g["grid_state_age_ms"],
    )
    for k in cfg["features"]:
        close("grid", k, np.where(valid, derived[k][safe], np.nan), g[k])
    sums = books["bid_price_units_1"] + books["ask_price_units_1"]
    alignment = []
    for h, tag in [(100, "100MS"), (1000, "1S"), (5000, "5S"), (10000, "10S")]:
        tau = g["grid_timestamp"] + h * 10**6
        j = np.searchsorted(t, tau, side="right") - 1
        ok = valid & (tau < end)
        target = np.where(ok, sums[np.maximum(j, 0)] / sums[safe] - 1, np.nan)
        close("grid", "return " + tag, target, g["GRID_FWD_MID_RET_" + tag])
        close("grid", "available " + tag, ok, g["target_available_" + tag])
        close(
            "grid",
            "target timestamp " + tag,
            np.where(ok, t[np.maximum(j, 0)], 0),
            g["target_state_timestamp_" + tag],
            0,
        )
        for k in np.flatnonzero(ok)[::1000]:
            alignment.append(
                dict(
                    grid_index=int(k),
                    horizon_ms=h,
                    feature_time=int(g["grid_timestamp"][k]),
                    requested_target=int(tau[k]),
                    selected_before=int(t[j[k]]),
                    next_after=int(t[j[k] + 1]) if j[k] + 1 < len(t) else 0,
                    before_not_after=bool(
                        t[j[k]] <= tau[k] and (j[k] + 1 == len(t) or t[j[k] + 1] > tau[k])
                    ),
                )
            )
    record("grid", "23398 valid primary labels", int(g["target_available_1S"].sum()) == 23398)
    write("label_bracketing", alignment)
    (OUT / "label_bracketing.csv").replace(LOCAL / "label_bracketing.csv")
    write(
        "causal_alignment_audit",
        [
            r
            for r in CHECKS
            if r["section"] in ["raw", "invariants", "features", "causality", "mutation", "grid"]
        ],
    )
    dump(
        "data_audit_summary.json",
        dict(
            phase2_features_compared=len(derived),
            feature_rows=len(t),
            grid_rows=len(i),
            valid_targets=23398,
            timestamp_ties=len(ties),
            tie_corruptions_changing_obi=sum(changed_counts),
            future_cutoffs=len(cutoffs),
            tick_fields_unused=True,
        ),
    )
    return p2, g, cfg
