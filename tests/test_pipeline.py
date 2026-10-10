"""End-to-end synthetic day: warm-up, ties, corrections and failure logging."""

import json
import zipfile

import pyarrow.parquet as pq
import pytest
import yaml

from lob_research.download import sha256
from lob_research.pipeline import run


def frame(ms, kind, oid=None, side=None, size=None, price=None, match=None):
    prefix = f"{ms:08d}{kind}"
    if kind == "S":
        return prefix + side + "\n"
    if kind in {"A", "P"}:
        return (
            prefix
            + f"{oid:9d}{side}{size:6d}{'TEST':6s}{price:10d}"
            + ("Y" if kind == "A" else f"{match:9d}")
            + "\n"
        )
    if kind in {"E", "X"}:
        return prefix + f"{oid:9d}{size:6d}" + (f"{match:9d}" if kind == "E" else "") + "\n"
    if kind == "B":
        return prefix + f"{match:9d}\n"


def fixture(tmp_path, monkeypatch, bad=False):
    monkeypatch.chdir(tmp_path)
    # A $100 bid of 100 shares and $101 ask of 200 shares exist before RTH.
    lines = [
        frame(25200000, "S", side="S"),
        frame(34199998, "A", 1, "B", 100, 1000000),
        frame(34199999, "A", 2, "S", 200, 1010000),
        frame(34200000, "E", 1, size=40, match=1),
        frame(34200000, "P", 3, "S", 50, 1005000, 2),
        frame(34200001, "B", match=2),
        frame(34200002, "X", 2, size=50),
        frame(57600000, "X", 1, size=60),
        frame(72000000, "S", side="E"),
        frame(72000001, "X", 2, size=150),
    ]
    if bad:
        lines.insert(4, "34200000Z\n")
    raw = tmp_path / "sample.zip"
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("sample.txt", "".join(lines) + "\n")
    record = {
        "filename": raw.as_posix(),
        "source": "https://example.invalid/sample.zip",
        "sha256": sha256(raw),
        "size_bytes": raw.stat().st_size,
    }
    (tmp_path / "DATA_MANIFEST.json").write_text(json.dumps({"artifacts": [record]}))
    cfg = {
        "source_url": record["source"],
        "raw_path": raw.as_posix(),
        "member": "sample.txt",
        "date": "2003-01-03",
        "symbol": "TEST",
        "timezone": "America/New_York",
        "session_start": "09:30:00",
        "session_end": "16:00:00",
        "depth": 10,
        "batch_rows": 2,
        "output_dir": "processed",
        "reports_dir": "reports",
        "price_grid_units": 1,
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    return raw


def test_full_pipeline_and_deterministic_replay(tmp_path, monkeypatch):
    raw = fixture(tmp_path, monkeypatch)
    digest = sha256(raw)
    stats = run()
    assert stats["source_messages"] == 10
    assert stats["rth_events"] == 4
    assert stats["selected_before_window_events"] == 2
    assert stats["selected_after_window_events"] == 2
    assert stats["remaining_market_wide_orders_after_eof"] == 0
    assert stats["rth_adjacent_duplicate_timestamps"] == 1
    books = pq.read_table("processed/books.parquet").to_pylist()
    assert [b["best_bid_size"] for b in books] == [60, 60, 60, 60]
    assert [b["best_ask_size"] for b in books] == [200, 200, 200, 150]
    assert all(b["midprice"] == 100.5 and b["spread"] == 1 for b in books)
    assert all(b["bid_price_2"] is None for b in books)
    trades = pq.read_table("processed/trades.parquet").to_pylist()
    assert len(trades) == 2 and [t["price"] for t in trades] == [100, 100.5]
    assert trades[0]["pre_quote_source_row"] == 3  # warm-up row, outside output
    assert trades[0]["post_quote_source_row"] == 4
    assert all(t["aggressor_side"] is None for t in trades)
    correction = pq.read_table("processed/trade_breaks.parquet").to_pylist()
    assert correction[0]["original_trade_source_row"] == 5
    before_hashes = {name: value["sha256"] for name, value in stats["artifacts"].items()}
    again = run()
    assert before_hashes == {name: value["sha256"] for name, value in again["artifacts"].items()}
    assert sha256(raw) == digest
    assert json.loads((tmp_path / "reports/run_status.json").read_text())["status"] == "complete"


def test_unknown_event_aborts_and_marks_incomplete(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch, bad=True)
    with pytest.raises(ValueError, match="unknown type"):
        run()
    failure = json.loads((tmp_path / "reports/failures.jsonl").read_text())
    assert failure["source_row"] == 5 and failure["stats"]["parse_failures"] == 1
    assert not (tmp_path / "processed/books.parquet").exists()
    assert json.loads((tmp_path / "reports/run_status.json").read_text())["status"] == "incomplete"
