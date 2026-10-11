"""Run the preregistered experiment; refuse accidental overwriting."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from threadpoolctl import threadpool_limits

from lob_research.phase3.artifacts import read_grid
from lob_research.phase4.audits import compare, real_audits
from lob_research.phase4.core import GRID, OUT, canonical, folds, run_fold
from lob_research.phase4.evaluate import evaluate, inference
from lob_research.phase4.storage import digest, dump, persist, verify_frozen


def main():
    verify_frozen()
    if (OUT / "regression_oos_predictions.parquet").exists():
        raise SystemExit("Refusing overwrite: preserve completed predictions")
    code = sorted(Path("src/lob_research/phase4").glob("*.py")) + [Path(__file__)]
    dump(
        "execution_manifest.json",
        dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            code_sha256={p.as_posix(): digest(p) for p in code},
            protocol_freeze_sha256=digest(OUT / "protocol_freeze.json"),
        ),
    )
    g = canonical(read_grid(GRID))
    with threadpool_limits(limits=1):
        first, second = [], []
        for run, destination in [(1, first), (2, second)]:
            for embargo in (0, 10):
                for fold in folds():
                    destination.append(run_fold(g, fold, embargo, variants=embargo == 0))
                    print(
                        f"Completed deterministic run {run}, embargo {embargo}, fold {fold['fold_id']}",
                        flush=True,
                    )
        checks = sum(compare(a, b, True, g, g) for a, b in zip(first, second))
        dump(
            "reproducibility_audit.json",
            dict(
                status="PASS",
                runs=2,
                checks=checks,
                tolerance=1e-12,
                exact_artifact_match=True,
                blas_threads=1,
                seed=404,
            ),
        )
        audit = real_audits(g, first[0])
        dump("isolation_audit.json", audit)
        persist(g, first)
    evaluate()
    inference(progress=lambda msg: print(msg, flush=True))
    verify_frozen()
    dump(
        "prediction_hashes.json", {p.name: digest(p) for p in OUT.glob("*_oos_predictions.parquet")}
    )
    print(
        "Model experiment and persisted-prediction evaluation complete; independent verification pending.",
        flush=True,
    )


if __name__ == "__main__":
    main()
