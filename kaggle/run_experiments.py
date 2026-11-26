"""Kaggle entry point: backbone/resolution sweep, then the full method grid on the winner.

The winner is the sweep entry with the best mean validation macro-AUROC; test scores play
no part in the choice. Each visible GPU runs its own shard of the grid in parallel.
Attach `orvile/dental-opg-xray-dataset` and a GPU to the notebook. Outputs land in
/kaggle/working: per-run metrics.json files and one summary CSV per phase.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from run_grid import summarize  # noqa: E402

# Kaggle has changed its mount layout before, so locate the folder instead of hard-coding it.
DATA_DIR = str(next(Path("/kaggle/input").rglob("Dental OPG (Classification)")))
OUT = Path("/kaggle/working")
CONFIG = REPO / "configs" / "default.yaml"
FOLDS = [0, 1, 2, 3, 4]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dental_ssl")


def run_phase(name: str, blocks: list[dict]) -> list[dict]:
    grid_file = OUT / f"{name}_grid.yaml"
    grid_file.write_text(yaml.safe_dump({"grid": blocks}))
    shards = max(torch.cuda.device_count(), 1)
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                str(REPO / "scripts" / "run_grid.py"),
                "--data-dir",
                DATA_DIR,
                "--config",
                str(CONFIG),
                "--grid",
                str(grid_file),
                "--runs-dir",
                str(OUT / name),
                "--summary",
                str(OUT / f"{name}_summary_shard{i}.csv"),
                "--shard",
                f"{i}/{shards}",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)},
        )
        for i in range(shards)
    ]
    failed = [w.args for w in workers if w.wait() != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} shard(s) failed in phase {name}")
    return summarize(OUT / name, OUT / f"{name}_summary.csv")


sweep = yaml.safe_load((REPO / "configs" / "sweep.yaml").read_text())["grid"]
rows = run_phase("sweep", sweep)
best = max(rows, key=lambda r: r["val_macro_auroc_mean"])
log.info("selected %s (val AUROC %.4f)", best["experiment"], best["val_macro_auroc_mean"])
winner = next(b for b in sweep if b["name"] == best["experiment"])

run_phase(
    "final",
    [
        {
            **winner,
            "methods": ["supervised", "fixmatch", "meanteacher"],
            "fractions": [0.2, 0.3, 0.5],
            "folds": FOLDS,
        },
        # Fully supervised upper bound: every training label used.
        {**winner, "methods": ["supervised"], "fractions": [1.0], "folds": FOLDS},
    ],
)
