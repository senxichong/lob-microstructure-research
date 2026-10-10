"""Bounded Parquet row-group buffers with explicit types and raw lineage."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def schemas(depth, lineage):
    common = [
        ("source_row", pa.int64()),
        ("timestamp", pa.timestamp("ns", tz="UTC")),
        ("source_time_ms", pa.int64()),
        ("event_type", pa.string()),
    ]
    events = common + [
        ("order_id", pa.int64()),
        ("side", pa.string()),
        ("price_units", pa.int64()),
        ("price", pa.float64()),
        ("size", pa.int64()),
        ("match_id", pa.int64()),
        ("display", pa.string()),
        ("symbol", pa.string()),
    ]
    books = common + [
        ("bid_levels_available", pa.int64()),
        ("ask_levels_available", pa.int64()),
        ("best_bid_price", pa.float64()),
        ("best_bid_size", pa.int64()),
        ("best_ask_price", pa.float64()),
        ("best_ask_size", pa.int64()),
    ]
    for side in ("bid", "ask"):
        for level in range(1, depth + 1):
            books.extend(
                [
                    (f"{side}_price_units_{level}", pa.int64()),
                    (f"{side}_price_{level}", pa.float64()),
                    (f"{side}_size_{level}", pa.int64()),
                ]
            )
    books += [
        (name, pa.float64()) for name in ("midprice", "spread", "relative_spread", "microprice")
    ]
    books += [("issues", pa.list_(pa.string())), ("research_eligible", pa.bool_())]
    trades = events + [
        ("aggressor_side", pa.string()),
        ("aggressor_basis", pa.string()),
        ("pre_quote_source_row", pa.int64()),
        ("post_quote_source_row", pa.int64()),
        ("pre_bid_units", pa.int64()),
        ("pre_ask_units", pa.int64()),
        ("post_bid_units", pa.int64()),
        ("post_ask_units", pa.int64()),
    ]
    corrections = common + [("match_id", pa.int64()), ("original_trade_source_row", pa.int64())]
    metadata = {b"lineage": json.dumps(lineage, sort_keys=True).encode()}
    return {
        name: pa.schema(fields, metadata=metadata)
        for name, fields in (
            ("events", events),
            ("books", books),
            ("trades", trades),
            ("trade_breaks", corrections),
        )
    }


class BatchWriter:
    def __init__(self, path, schema, batch_rows=10000):
        self.path = Path(path)
        self.schema = schema
        self.rows = []
        self.limit = batch_rows
        self.count = 0
        self.writer = pq.ParquetWriter(self.path, schema, compression="zstd")

    def append(self, row):
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= self.limit:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()
