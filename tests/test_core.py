import json
from dataclasses import replace

import pyarrow.parquet as pq
import pytest

from lob_research.download import fetch, sha256
from lob_research.orderbook import OrderBook, diagnostics
from lob_research.pipeline import Distribution, audit_aggregate
from lob_research.schemas import Event, parse_line, price_dollars
from lob_research.storage import BatchWriter, schemas
from lob_research.timestamps import Ordering, clock_ms, iso_utc, utc_ns
from lob_research.validation import book_issues


def add(oid=1, side="B", size=100, price=1000000, symbol="MSFT"):
    return Event(oid + 1, 34200000, "A", oid, side, size, symbol, price, display="Y")


def started_book():
    book = OrderBook("MSFT")
    book.apply(Event(1, 25200000, "S", system_code="S"))
    return book


@pytest.mark.parametrize(
    "raw,kind",
    [
        (b"25200000SS\n", "S"),
        (b"25201000A        1B     1TEST       10000Y\n", "A"),
        (b"25202000X        1     1\n", "X"),
        (b"25203000E        2   100        1\n", "E"),
        (b"25204000P        3S   100TEST      911900        3\n", "P"),
        (b"25204000B        3\n", "B"),
    ],
)
def test_parse_fields(raw, kind):
    event = parse_line(raw, 1)
    assert event.kind == kind
    assert event.time_ms == int(raw[:8])


def test_price_size_scaling():
    # Hand-calculated $91.19 and 100 shares (not lots).
    e = parse_line(b"25204000P        3S   100TEST      911900        3\n", 1)
    assert e.price_units == 911900
    assert price_dollars(e.price_units) == 91.19
    assert e.size == 100
    assert e.side == "S"


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"12345678Z",
        b"25200000ST",
        b"25202000X        1     0",
        b"25202000X        1    -1",
        b"25202000X        1   1.0",
        b"99999999SS",
        b"25200000SSextra",
    ],
)
def test_bad_frames_fail(raw):
    with pytest.raises(ValueError):
        parse_line(raw, 5)


@pytest.mark.parametrize(
    "day,expected",
    [
        ("2003-01-03", "2003-01-03T14:30:00.000000000Z"),
        ("2012-06-21", "2012-06-21T13:30:00.000000000Z"),
    ],
)
def test_timezone_winter_summer(day, expected):
    assert iso_utc(utc_ns(day, 34200000)) == expected


def test_ms_precision_and_date_transition():
    assert iso_utc(utc_ns("2003-01-03", 86399999)) == "2003-01-04T04:59:59.999000000Z"
    assert utc_ns("2003-01-04", 0) - utc_ns("2003-01-03", 86399999) == 1000000


@pytest.mark.parametrize("day", ["2026-03-08", "2026-11-01"])
def test_dst_transition_day_rejected(day):
    with pytest.raises(ValueError, match="DST"):
        utc_ns(day, 34200000)


def test_session_half_open():
    start, end = clock_ms("09:30:00"), clock_ms("16:00:00")
    assert not start <= 34199999 < end
    assert start <= 34200000 < end
    assert start <= 57599999 < end
    assert not start <= 57600000 < end


def test_duplicate_timestamps_preserve_source_order():
    ordering = Ordering()
    assert ordering.observe(Event(1, 1, "S")) == (False, False)
    assert ordering.observe(Event(2, 1, "A")) == (False, True)
    assert ordering.observe(Event(3, 0, "X")) == (True, False)
    with pytest.raises(ValueError, match="source row"):
        ordering.observe(Event(5, 2, "E"))


def test_price_priority_and_aggregation():
    b = started_book()
    for e in [
        add(1, price=1000000),
        add(2, price=990000),
        add(3, size=50),
        add(4, side="S", price=1020000),
        add(5, side="S", price=1010000, size=50),
    ]:
        b.apply(e)
    s = b.snapshot(10)
    assert s == {"bids": [(1000000, 150), (990000, 100)], "asks": [(1010000, 50), (1020000, 100)]}
    assert book_issues(s) == []
    audit_aggregate(b)


def test_basic_diagnostics():
    d = diagnostics({"bids": [(1000000, 300)], "asks": [(1020000, 100)]})
    assert d == {"midprice": 101, "spread": 2, "relative_spread": 2 / 101, "microprice": 101.5}


@pytest.mark.parametrize("kind", ["X", "E"])
def test_partial_and_full_removal(kind):
    b = started_book()
    b.apply(add())
    b.apply(Event(3, 34200000, kind, 1, size=40, match_id=10 if kind == "E" else None))
    assert b.orders[1].size == 60
    assert b.snapshot(1)["bids"] == [(1000000, 60)]
    b.apply(Event(4, 34200000, kind, 1, size=60, match_id=11 if kind == "E" else None))
    assert not b.orders and b.snapshot(1)["bids"] == []


@pytest.mark.parametrize(
    "e",
    [
        Event(3, 1, "X", 1, size=101),
        Event(3, 1, "E", 1, size=0),
        Event(3, 1, "X", 999, size=1),
        Event(3, 1, "U", 1),
        Event(3, 1, "D", 1),
        add(1),
    ],
)
def test_reject_inconsistent_events_without_mutation(e):
    b = started_book()
    b.apply(add())
    before = b.snapshot(10)
    with pytest.raises(ValueError):
        b.apply(e)
    assert b.snapshot(10) == before and b.orders[1].size == 100


def test_hidden_trade_and_break_do_not_change_book():
    b = started_book()
    b.apply(add())
    before = b.snapshot(10)
    b.apply(replace(add(2), kind="P", match_id=8))
    b.apply(Event(4, 1, "B", match_id=8))
    assert before == b.snapshot(10)


def test_other_symbol_routing_and_unknown_order():
    b = started_book()
    b.apply(add(symbol="AAPL"))
    assert not b.snapshot(10)["bids"]
    assert b.apply(Event(3, 2, "E", 1, size=10)).symbol == "AAPL"
    with pytest.raises(ValueError, match="unknown order"):
        b.apply(Event(4, 2, "E", 2, size=1))


def test_trailing_cancel_after_end_of_day():
    b = started_book()
    b.apply(add())
    b.apply(Event(3, 72000000, "S", system_code="E"))
    b.apply(Event(4, 72000001, "X", 1, size=100))
    assert not b.orders
    with pytest.raises(ValueError):
        b.apply(add(2))


@pytest.mark.parametrize("ask,issue", [(100, "locked"), (99, "crossed")])
def test_locked_crossed_retained(ask, issue):
    s = {"bids": [(100, 1)], "asks": [(ask, 2)]}
    assert issue in book_issues(s)
    assert s["asks"][0][0] == ask


@pytest.mark.parametrize(
    "s,issue",
    [
        ({"bids": [(100, -1)], "asks": [(101, 1)]}, "negative_size"),
        ({"bids": [(100, 0)], "asks": [(101, 1)]}, "zero_size"),
        ({"bids": [(100, 1), (100, 2)], "asks": [(101, 1)]}, "duplicate_level"),
        ({"bids": [(99, 1), (100, 2)], "asks": [(101, 1)]}, "level_order"),
        ({"bids": [(0, 1)], "asks": [(101, 1)]}, "invalid_price"),
        ({"bids": [], "asks": [(101, 1)]}, "empty_bids"),
    ],
)
def test_invariants(s, issue):
    assert issue in book_issues(s)


def test_grid_and_promised_depth():
    s = {"bids": [(100, 1)], "asks": [(101, 2)]}
    assert {"off_grid_price", "sub_grid_spread", "missing_promised_depth"} <= set(
        book_issues(s, 10, 2)
    )


def test_hash_cache_integrity(tmp_path):
    raw = tmp_path / "raw.zip"
    raw.write_bytes(b"known bytes")
    record = {
        "filename": raw.as_posix(),
        "source": "https://example.invalid/a",
        "sha256": sha256(raw),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": [record]}))
    assert fetch(record["source"], raw, manifest) == record
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="refusing overwrite"):
        fetch(record["source"], raw, manifest)
    assert raw.read_bytes() == b"changed"


def test_parquet_round_trip_preserves_ms_and_lineage(tmp_path):
    schema = schemas(10, {"raw_sha256": "abc"})["events"]
    path = tmp_path / "events.parquet"
    writer = BatchWriter(path, schema, 1)
    ns = utc_ns("2003-01-03", 34200001)
    writer.append(
        {"source_row": 10, "timestamp": ns, "event_type": "A", "price_units": 911900, "size": 100}
    )
    writer.append(
        {"source_row": 11, "timestamp": ns, "event_type": "X", "price_units": 911900, "size": 50}
    )
    writer.close()
    table = pq.read_table(path)
    assert table.schema == schema
    assert table["source_row"].to_pylist() == [10, 11]
    assert table["timestamp"].cast("int64").to_pylist() == [ns, ns]
    assert json.loads(table.schema.metadata[b"lineage"])["raw_sha256"] == "abc"


def test_discrete_distribution():
    d = Distribution()
    for value in [1, 2, 2, 100]:
        d.add(value)
    assert d.result()["p50"] == 2
    assert d.result()["mean"] == 26.25
