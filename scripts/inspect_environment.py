"""Capture reproducibility context without environment variables or credentials."""

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

Path("reports").mkdir(exist_ok=True)
report = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "cwd": str(Path.cwd()),
    "os": platform.platform(),
    "python": sys.version,
    "git_status": subprocess.run(
        ["git", "status", "--short", "--branch"], capture_output=True, text=True
    ).stdout,
    "disk_free_bytes": shutil.disk_usage(".").free,
    "host_timezone_names": time.tzname,
    "client_timezone": "Asia/Shanghai",
    "network": "See reports/source_access_checks.json",
    "summer_ny_open_utc": datetime(2012, 6, 21, 9, 30, tzinfo=ZoneInfo("America/New_York"))
    .astimezone(timezone.utc)
    .isoformat(),
    "winter_ny_open_utc": datetime(2003, 1, 3, 9, 30, tzinfo=ZoneInfo("America/New_York"))
    .astimezone(timezone.utc)
    .isoformat(),
}
Path("reports/environment.json").write_text(json.dumps(report, indent=2) + "\n")
packages = sorted(
    (
        {"name": d.metadata["Name"], "version": d.version}
        for d in importlib.metadata.distributions()
    ),
    key=lambda x: x["name"].lower(),
)
Path("reports/environment_packages.json").write_text(json.dumps(packages, indent=2) + "\n")
print(json.dumps(report, indent=2))
