"""Fit only the preregistered experiment, twice, then freeze predictions."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from threadpoolctl import threadpool_limits

from lob_research.phase3.artifacts import read_grid
from lob_research.phase4.core import GRID, canonical, folds
from lob_research.phase5.audits import real_audits
from lob_research.phase5.core import compare, run_fold
from lob_research.phase5.storage import OUT, digest, dump, frozen, persist


def main():
    frozen()
    if (OUT / "regression_oos_predictions.parquet").exists():
        raise SystemExit("Refusing overwrite of frozen OOS predictions")
    paths = list(Path("src/lob_research/phase5").glob("*.py")) + [Path(__file__)]
    dump(
        "execution_manifest.json",
        dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            code_sha256={p.as_posix(): digest(p) for p in paths},
            protocol_freeze_sha256=digest(OUT / "protocol_freeze.json"),
        ),
    )
    g = canonical(read_grid(GRID))
    first = []
    with threadpool_limits(limits=1):
        for f in folds():
            first.append(run_fold(g, f))
            print(f"Run 1 fold {f['fold_id']} fitted", flush=True)
        count = 0
        for f, a in zip(folds(), first):
            b = run_fold(g, f)
            count += compare(a, b, g, g)
            print(f"Run 2 fold {f['fold_id']}: identical", flush=True)
        dump(
            "reproducibility_audit.json",
            dict(
                status="PASS",
                runs=2,
                comparisons=count,
                tolerance=1e-12,
                model_signatures_equal=True,
                seed=505,
            ),
        )
        print("Real future/test-label/storage-order mutation audits running", flush=True)
        dump("isolation_audit.json", real_audits(g, first[0]))
        persist(g, first)
    frozen()
    print("Phase 5 fits, audits and OOS persistence complete.", flush=True)


if __name__ == "__main__":
    main()
