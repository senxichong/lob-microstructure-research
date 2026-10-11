# Phase 4 feature set 4.0.0 — frozen before fitting

| Phase 3 feature | Decision/family | Reason from Phase 3; redundancy |
|---|---|---|
| OBI_L1 | Retain, imbalance | rho .04382; named positive benchmark, compact top level |
| OBI_L5 | Exclude, imbalance | rho .03030; redundant depth alternative; no depth search |
| OBI_L10 | Exclude, imbalance | rho .02213; weaker redundant depth variant |
| WOBI_L10 | Exclude, imbalance | rho .03553; alternative depth weighting, avoid family proliferation |
| MICROPRICE_MINUS_MID | Retain, microprice | rho .03246; economically fixed spread-weighted imbalance; rho .945 with OBI, expect unstable conditional attribution |
| OFI_100MS | Retain, OFI | rho -.03349; diagnostic short-window sign is preserved, never flipped |
| OFI_1S | Retain, OFI | rho .03520; dynamic flow, horizon/regime sensitivity; correlated .432 with 100ms |
| RELATIVE_SPREAD | Retain, context | weak marginal rho -.00899; fixed liquidity context, not selected for significance |
| EVENT_RATE_1S | Retain, context | weak marginal rho -.00277; fixed activity context |
| TIME_SINCE_PREVIOUS_TRADE | Retain, context | weak marginal rho .00461; fixed timing context |

Primary ordered features: OBI_L1, MICROPRICE_MINUS_MID, OFI_100MS, OFI_1S,
RELATIVE_SPREAD, EVENT_RATE_1S, TIME_SINCE_PREVIOUS_TRADE. No new transforms.
No Phase 4 test result can alter membership. RANDOM_CONTROL is diagnostic only.
Static/flow membership and four family removals are fixed in PHASE4_PROTOCOL.md.
