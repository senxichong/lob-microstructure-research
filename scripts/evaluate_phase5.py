import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_research.phase3.artifacts import read_grid
from lob_research.phase4.core import GRID
from lob_research.phase5.evaluate import bootstrap, metrics
from lob_research.phase5.interpret import diagnose
from lob_research.phase5.storage import digest, dump, frozen

if __name__ == "__main__":
    frozen()
    dump(
        "evaluation_code.json",
        {
            p.as_posix(): digest(p)
            for p in [
                Path(__file__),
                Path("src/lob_research/phase5/evaluate.py"),
                Path("src/lob_research/phase5/interpret.py"),
            ]
        },
    )
    metrics()
    bootstrap(progress=lambda msg: print(msg, flush=True))
    diagnose(read_grid(GRID), progress=lambda msg: print(msg, flush=True))
    frozen()
    print("Persisted OOS evaluation and post-hoc diagnostics complete.", flush=True)
