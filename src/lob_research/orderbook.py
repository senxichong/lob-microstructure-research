"""Deterministic order-level replay; no queue or execution simulation."""

from dataclasses import dataclass

from .schemas import price_dollars


@dataclass(slots=True)
class Order:
    symbol: str
    side: str
    price: int
    size: int
    display: str


class OrderBook:
    """Track reference integrity market-wide, aggregate one selected instrument.

    All active IDs are retained so an unknown cancel cannot masquerade as another
    instrument's event. Price levels contain visible shares, never synthetic seeds.
    """

    def __init__(self, symbol):
        self.symbol = symbol
        self.orders = {}
        self.levels = {"B": {}, "S": {}}
        self.started = False
        self.ended = False
        self.max_order_id = 0

    def _change(self, order, amount):
        if order.symbol != self.symbol:
            return
        levels = self.levels[order.side]
        remaining = levels.get(order.price, 0) + amount
        if remaining < 0:
            raise ValueError("negative aggregate size")
        if remaining:
            levels[order.price] = remaining
        else:
            levels.pop(order.price, None)

    def apply(self, e):
        """Return enriched affected order; book is post-event on return.

        X performs partial cancellation or full deletion. This protocol has no
        standalone D or U/REPLACE. P/B have no effect on the displayed book.
        """
        if e.kind == "S":
            if e.system_code == "S":
                if self.started or self.orders:
                    raise ValueError("unexpected repeated start-of-day")
                self.started = True
            elif e.system_code == "E":
                if not self.started or self.ended:
                    raise ValueError("unexpected end-of-day")
                self.ended = True
                # Keep orders to consume trailing X messages. Never reset on E:
                # source specification text and revision history disagree here.
            else:
                raise ValueError("unknown system event")
            return None
        if not self.started:
            raise ValueError("input does not begin with start-of-day")
        if e.kind == "A":
            if self.ended or e.order_id in self.orders or e.order_id <= self.max_order_id:
                raise ValueError("late/duplicate/nonincreasing add order reference")
            if e.side not in {"B", "S"} or e.size <= 0 or e.price_units <= 0:
                raise ValueError("invalid add")
            if e.symbol == self.symbol and e.display != "Y":
                raise ValueError("legacy non-Y display for selected symbol requires research")
            self.max_order_id = e.order_id
            order = Order(e.symbol, e.side, e.price_units, e.size, e.display)
            self._change(order, e.size)
            self.orders[e.order_id] = order
            return Order(order.symbol, order.side, order.price, e.size, order.display)
        if e.kind in {"X", "E"}:
            if e.order_id not in self.orders:
                raise ValueError(f"unknown order reference {e.order_id}")
            order = self.orders[e.order_id]
            if e.size <= 0 or e.size > order.size:
                raise ValueError(f"over-reduction/nonpositive size for {e.order_id}")
            self._change(order, -e.size)
            order.size -= e.size
            result = Order(order.symbol, order.side, order.price, e.size, order.display)
            if not order.size:
                del self.orders[e.order_id]
            return result
        if e.kind == "P":
            if e.size <= 0 or e.price_units <= 0 or e.side not in {"B", "S"}:
                raise ValueError("invalid hidden trade")
            return Order(e.symbol, e.side, e.price_units, e.size, "hidden")
        if e.kind == "B":
            return None
        raise ValueError(f"unsupported event {e.kind}")

    def snapshot(self, depth):
        return {
            "bids": sorted(self.levels["B"].items(), reverse=True)[:depth],
            "asks": sorted(self.levels["S"].items())[:depth],
        }


def diagnostics(snapshot):
    bids, asks = snapshot["bids"], snapshot["asks"]
    if not bids or not asks:
        return {"midprice": None, "spread": None, "relative_spread": None, "microprice": None}
    bid, bid_size = bids[0]
    ask, ask_size = asks[0]
    mid = price_dollars(bid + ask) / 2
    spread = price_dollars(ask - bid)
    return {
        "midprice": mid,
        "spread": spread,
        "relative_spread": spread / mid if mid > 0 else None,
        "microprice": price_dollars(ask * bid_size + bid * ask_size) / (bid_size + ask_size)
        if bid_size + ask_size > 0
        else None,
    }
