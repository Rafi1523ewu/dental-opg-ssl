"""Run every configuration in a grid, then write an aggregated summary.

Each grid block may set a `name` (the output folder, defaults to the model) and
`overrides` (e.g. a different image size) merged into the base config.
Finished runs are skipped, so the script can be restarted after a timeout.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from dental_ssl.train import DEFAULT_CONFIG, load_config, run

log = logging.getLogger("dental_ssl")


def with_overrides(cfg: dict, overrides: dict) -> dict:
    return {
        key: {**value, **overrides.get(key, {})} if isinstance(value, dict) else value
        for key, value in cfg.items()
    }


def grid(cfg: dict, blocks: list[dict]):
    for block in blocks:
        run_cfg = with_overrides(cfg, block.get("overrides", {}))
        name = block.get("name", block["model"])
        for method in block["methods"]:
            for fraction in block["fractions"]:
                for fold in block["folds"]:
                    yield name, block["model"], run_cfg, method, fraction, fold


def run_dir(root: Path, name: str, method: str, fraction: float, fold: int) -> Path:
    return root / name / method / f"frac{round(fraction * 100)}_fold{fold}"


def summarize(runs_root: Path, out_csv: Path) -> list[dict]:
    """Mean and sample std across folds, one row per experiment/method/fraction."""
    groups = defaultdict(list)
    for path in sorted(runs_root.rglob("metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        experiment = path.relative_to(runs_root).parts[0]
        groups[(experiment, m["method"], m["fraction"])].append(m)

    def spread(values: list[float]) -> float:
        return round(statistics.stdev(values), 4) if len(values) > 1 else 0.0

    rows = []
    for (experiment, method, fraction), runs in sorted(groups.items()):
        row = {
            "experiment": experiment,
            "model": runs[0]["model"],
            "image_size": runs[0]["config"]["data"]["image_size"],
            "method": method,
            "labeled_fraction": fraction,
            "folds": len(runs),
        }
        for key in (
            "val_macro_auroc",
            "test_macro_auroc",
            "test_macro_ap",
            "test_macro_f1",
        ):
            values = [r[key] for r in runs]
            row[f"{key}_mean"] = round(statistics.mean(values), 4)
            row[f"{key}_std"] = spread(values)
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["experiment"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_all(
    cfg: dict,
    blocks: list[dict],
    data_dir: str,
    runs_root: Path,
    summary: Path,
    shard: tuple[int, int] = (0, 1),
):
    """Run this shard's share of the grid (every n-th job), then summarise all finished runs."""
    index, count = shard
    jobs = list(grid(cfg, blocks))[index::count]
    for n, (name, model, run_cfg, method, fraction, fold) in enumerate(jobs, 1):
        out = run_dir(runs_root, name, method, fraction, fold)
        if (out / "metrics.json").exists():
            log.info("[%d/%d] skip %s (done)", n, len(jobs), out)
            continue
        log.info("[%d/%d] %s", n, len(jobs), out)
        run(
            run_cfg,
            data_dir=data_dir,
            method=method,
            model_name=model,
            fraction=fraction,
            fold=fold,
            out_dir=out,
        )
    rows = summarize(runs_root, summary)
    log.info("wrote %d summary rows to %s", len(rows), summary)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--grid", help="YAML file whose `grid` replaces the config's grid")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    args = parser.parse_args()
    index, count = map(int, args.shard.split("/"))
    if not 0 <= index < count:
        parser.error(f"invalid --shard {args.shard}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    blocks = cfg["grid"]
    if args.grid:
        with open(args.grid, encoding="utf-8") as f:
            blocks = yaml.safe_load(f)["grid"]
    run_all(cfg, blocks, args.data_dir, Path(args.runs_dir), Path(args.summary), (index, count))


if __name__ == "__main__":
    main()
