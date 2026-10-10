"""Record issues without deleting or repairing observed states."""


def book_issues(snapshot, grid_units=1, promised_depth=None):
    issues = []
    for side, descending in (("bids", True), ("asks", False)):
        levels = snapshot[side]
        prices = [p for p, _ in levels]
        if not levels:
            issues.append(f"empty_{side}")
        if any(p <= 0 for p in prices):
            issues.append("invalid_price")
        if any(p % grid_units for p in prices):
            issues.append("off_grid_price")
        if any(q < 0 for _, q in levels):
            issues.append("negative_size")
        if any(q == 0 for _, q in levels):
            issues.append("zero_size")
        if len(prices) != len(set(prices)):
            issues.append("duplicate_level")
        if prices != sorted(prices, reverse=descending):
            issues.append("level_order")
        if promised_depth and len(levels) != promised_depth:
            issues.append("missing_promised_depth")
    if snapshot["bids"] and snapshot["asks"]:
        spread = snapshot["asks"][0][0] - snapshot["bids"][0][0]
        if spread == 0:
            issues.append("locked")
        elif spread < 0:
            issues.append("crossed")
        elif spread < grid_units:
            issues.append("sub_grid_spread")
    return sorted(set(issues))
