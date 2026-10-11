# Frozen Phase 3 primary features — version 3.0.0

Declared before predictive statistics, 2026-10-11 Asia/Shanghai. Exactly these ten existing Phase-2 columns are primary; no replacements, extra features, sign flips or tick imputation. The single seeded random control is not a research hypothesis and is excluded from the ten-test FDR family.

| Feature | Family | Frozen definition | Direction hypothesis | Inclusion reason |
|---|---|---|---|---|
| OBI_L1 | Imbalance | (bid L1 shares - ask L1 shares)/(their sum) | Positive association with future mid return | Immediate displayed pressure |
| OBI_L5 | Imbalance | Same ratio for summed five-level depth | Positive | Near-book depth context |
| OBI_L10 | Imbalance | Same ratio for summed ten-level depth | Positive | Wider visible depth context |
| WOBI_L10 | Imbalance | Weighted ratio, level weights 1/k | Positive | Preregistered proximity weighting |
| MICROPRICE_MINUS_MID | Microprice | (ask*bid_qty + bid*ask_qty)/(bid_qty+ask_qty) - mid, USD | Positive | Quote tilt from displayed size asymmetry |
| OFI_100MS | Order flow | Sum frozen Cont-style BBO increments over (t-100ms,t] | Positive | Immediate pressure |
| OFI_1S | Order flow | Sum frozen Cont-style BBO increments over (t-1s,t] | Positive | Primary-horizon historical flow |
| RELATIVE_SPREAD | Context | (ask-bid)/mid | None; two-sided context diagnostic | Liquidity state |
| EVENT_RATE_1S | Context | Retained events in (t-1s,t] divided by 1 second | None; two-sided context diagnostic | Activity state |
| TIME_SINCE_PREVIOUS_TRADE | Context | Milliseconds since strictly prior E/P event at source state | None; two-sided context diagnostic | Recent trading activity |

All features are copied from the grid's as-of source row. Rolling windows and elapsed-time features remain evaluated at that source event, not recalculated at the later grid clock; their age is reported. This follows the explicit requirement to use the frozen Phase-2 features directly. Redundancy among depths, microprice and imbalance is expected; these are not ten independent economic discoveries.
