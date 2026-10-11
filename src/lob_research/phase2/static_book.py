"""Pure current-state book features."""

import numpy as np

from .common import safe_ratio, valid_depth


def static_features(s, cfg, emit):
    valid = valid_depth(s, 1)
    bid, ask = s.bid[:, 0] / 10000, s.ask[:, 0] / 10000
    qb, qa = s.bsize[:, 0], s.asize[:, 0]
    mid = (s.bid[:, 0] + s.ask[:, 0]) / 20000
    spread = (s.ask[:, 0] - s.bid[:, 0]) / 10000
    micro = safe_ratio(ask * qb + bid * qa, qb + qa)
    for name, values, formula, unit in (
        ("MIDPRICE", mid, "(bid+ask)/2", "USD"),
        ("SPREAD", spread, "ask-bid", "USD"),
        ("RELATIVE_SPREAD", safe_ratio(spread, mid), "spread/mid", "ratio"),
        ("MICROPRICE", micro, "(ask*bid_size+bid*ask_size)/(bid_size+ask_size)", "USD"),
        ("MICROPRICE_MINUS_MID", micro - mid, "microprice-mid", "USD"),
    ):
        emit(
            name,
            np.where(valid, values, np.nan),
            formula,
            ["BBO prices and sizes"],
            0,
            unit,
            "invalid BBO -> null",
        )
    tick = cfg["tick_size_usd"]
    values = (micro - mid) / tick if tick is not None else np.full(len(mid), np.nan)
    emit(
        "MICROPRICE_DEVIATION_TICKS",
        np.where(valid, values, np.nan),
        "(microprice-mid)/verified_tick",
        ["BBO", "verified tick"],
        0,
        "ticks",
        "unknown tick or invalid BBO -> null",
    )
    all_depth = valid.copy()
    for k in cfg["book_depths"]:
        good = valid_depth(s, k)
        all_depth &= good
        bd, ad = s.bsize[:, :k].sum(axis=1), s.asize[:, :k].sum(axis=1)
        for name, values, formula, unit in (
            (f"BID_DEPTH_L{k}", bd, f"sum bid_size[1:{k}]", "shares"),
            (f"ASK_DEPTH_L{k}", ad, f"sum ask_size[1:{k}]", "shares"),
            (f"TOTAL_DEPTH_L{k}", bd + ad, "bid_depth+ask_depth", "shares"),
            (f"DEPTH_RATIO_L{k}", safe_ratio(bd, ad), "bid_depth/ask_depth", "ratio"),
            (
                f"OBI_L{k}",
                safe_ratio(bd - ad, bd + ad),
                "(bid_depth-ask_depth)/(bid_depth+ask_depth)",
                "ratio",
            ),
        ):
            emit(
                name,
                np.where(good, values, np.nan),
                formula,
                [f"prices/sizes levels 1..{k}"],
                0,
                unit,
                "missing/invalid requested levels or zero denominator -> null",
            )
    weights = 1 / np.arange(1, 11)
    if s.bid.shape[1] >= 10:
        wb, wa = s.bsize[:, :10] @ weights, s.asize[:, :10] @ weights
        weighted = safe_ratio(wb - wa, wb + wa)
    else:
        weighted = np.full(len(s.time), np.nan)
    emit(
        "WOBI_L10",
        np.where(valid_depth(s, 10), weighted, np.nan),
        "sum((bid_size-ask_size)/rank)/sum((bid_size+ask_size)/rank)",
        ["levels 1..10"],
        0,
        "ratio",
        "require ten valid levels; zero denominator -> null",
    )
    return valid, all_depth
