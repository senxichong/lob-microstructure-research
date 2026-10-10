"""Exact ASCII ITCH 2.0 fields (Nasdaq tv-itch2.pdf, pages 2-5)."""

from dataclasses import dataclass

PRICE_SCALE = 10000
LENGTHS = {"S": 10, "A": 42, "E": 33, "X": 24, "P": 50, "B": 18}


@dataclass(frozen=True, slots=True)
class Event:
    row: int
    time_ms: int
    kind: str
    order_id: int | None = None
    side: str | None = None
    size: int | None = None
    symbol: str | None = None
    price_units: int | None = None
    match_id: int | None = None
    display: str | None = None
    system_code: str | None = None


def parse_line(raw: bytes, row: int) -> Event:
    """Do not strip spaces: they are significant fixed-width padding."""
    raw = raw.removesuffix(b"\n").removesuffix(b"\r")
    raw.decode("ascii", errors="strict")
    if len(raw) < 9:
        raise ValueError(f"row {row}: truncated message")
    kind = chr(raw[8])
    if kind not in LENGTHS or len(raw) != LENGTHS[kind]:
        raise ValueError(f"row {row}: unknown type or length: {kind!r}/{len(raw)}")

    def number(a, b):
        value = raw[a:b].lstrip(b" ")
        if not value or not value.isdigit():
            raise ValueError(f"row {row}: invalid numeric field {a}:{b}")
        return int(value)

    fields = dict(row=row, time_ms=number(0, 8), kind=kind)
    if not 0 <= fields["time_ms"] < 86400000:
        raise ValueError(f"row {row}: timestamp outside source day")
    if kind == "S":
        code = chr(raw[9])
        if code not in {"S", "E"}:
            raise ValueError(f"row {row}: unknown system code {code}")
        fields["system_code"] = code
    elif kind == "B":
        fields["match_id"] = number(9, 18)
    else:
        fields["order_id"] = number(9, 18)
        if fields["order_id"] <= 0:
            raise ValueError(f"row {row}: nonpositive order ID")
        if kind in {"A", "P"}:
            fields.update(
                side=chr(raw[18]),
                size=number(19, 25),
                symbol=raw[25:31].decode().rstrip(),
                price_units=number(31, 41),
            )
            if (
                fields["side"] not in {"B", "S"}
                or not fields["symbol"]
                or fields["price_units"] <= 0
            ):
                raise ValueError(f"row {row}: invalid side/symbol/price")
            if kind == "A":
                fields["display"] = chr(raw[41])
                # Legacy S is retained, but semantics are not silently guessed.
                if fields["display"] not in {"Y", "S"}:
                    raise ValueError(f"row {row}: unknown display code")
            else:
                fields["match_id"] = number(41, 50)
        else:
            fields["size"] = number(18, 24)
            if kind == "E":
                fields["match_id"] = number(24, 33)
        if fields["size"] <= 0:
            raise ValueError(f"row {row}: nonpositive message size")
    return Event(**fields)


def price_dollars(units):
    return units / PRICE_SCALE
