"""Repeat small, read-only access tests; never download multi-GB alternatives."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

URLS = {
    "nasdaq_itch2_selected": "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/S010303-v2.zip",
    "nasdaq_itch5_sample": "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/12302019.NASDAQ_ITCH50.gz",
    "lobster_legacy_sample": "https://data.lobsterdata.com/info/sample/LOBSTER_SampleFile_AAPL_2012-06-21_10.zip",
    "lobster_main_legacy_path": "https://lobsterdata.com/info/sample/LOBSTER_SampleFile_AAPL_2012-06-21_10.zip",
    "fi2010_catalog": "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649",
    "tardis_crypto_free_day": "https://datasets.tardis.dev/v1/binance/book_snapshot_5/2020/01/01/BTCUSDT.csv.gz",
}


def probe(item):
    name, url = item
    result = {
        "name": name,
        "url": url,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "streaming GET; read first 64 bytes only; close response",
    }
    try:
        with requests.get(url, stream=True, timeout=(10, 20)) as response:
            prefix = next(response.iter_content(64), b"")
            result.update(
                status=response.status_code,
                final_url=response.url,
                content_type=response.headers.get("Content-Type"),
                content_length=response.headers.get("Content-Length"),
                first_16_bytes_hex=prefix[:16].hex(),
                is_zip=prefix.startswith(b"PK\x03\x04"),
                is_gzip=prefix.startswith(b"\x1f\x8b"),
            )
    except requests.RequestException as exc:
        result["error"] = str(exc)
    return result


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(probe, URLS.items()))
    Path("reports/source_access_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
