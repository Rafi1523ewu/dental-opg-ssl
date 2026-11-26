"""Train and evaluate one model with one method on one labeled fraction and CV fold."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from dental_ssl import ssl
from dental_ssl.data import (
    LabeledImages,
    UnlabeledImages,
    eval_transform,
    make_split,
    scan_multilabel,
    strong_transform,
    weak_transform,
)

log = logging.getLogger("dental_ssl")

METHODS = ("supervised", "fixmatch", "meanteacher")
MODEL_ALIASES = {
    "efficientnet_b2": "efficientnet_b2.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "vit_b16": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
}
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(
        MODEL_ALIASES.get(name, name), pretrained=pretrained, num_classes=num_classes
    )


def forever(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def positive_weights(targets: np.ndarray) -> torch.Tensor:
    """negatives / positives per label, so rare conditions are not drowned out."""
    positives = targets.sum(axis=0).clip(min=1)
    return torch.tensor((len(targets) - positives) / positives, dtype=torch.float32)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, probs = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).float().sigmoid().cpu().numpy())
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(probs)


def per_label(metric, y_true: np.ndarray, probs: np.ndarray) -> list[float]:
    """Metric per label; NaN where a label has only one class in this set (undefined)."""
    return [
        metric(y_true[:, c], probs[:, c]) if 0 < y_true[:, c].sum() < len(y_true) else np.nan
        for c in range(y_true.shape[1])
    ]


def scores(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = (probs >= 0.5).astype(int)
    return {
        "macro_auroc": float(np.nanmean(per_label(roc_auc_score, y_true, probs))),
        "macro_ap": float(np.nanmean(per_label(average_precision_score, y_true, probs))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float(accuracy_score(y_true, y_pred)),
    }


def run(
    cfg: dict,
    *,
    data_dir: str | Path,
    method: str,
    model_name: str,
    fraction: float,
    fold: int,
    out_dir: str | Path,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    data_cfg, train_cfg, ssl_cfg = cfg["data"], cfg["train"], cfg["ssl"]
    started = time.time()
    set_seed(data_cfg["cv_seed"] + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    paths, targets, classes = scan_multilabel(data_dir, data_cfg["exclude_classes"])
    num_classes = len(classes)
    split = make_split(
        targets,
        fold=fold,
        n_folds=data_cfg["n_folds"],
        val_size=data_cfg["val_size"],
        labeled_fraction=fraction,
        seed=data_cfg["cv_seed"],
    )
    if method != "supervised" and not split.unlabeled:
        raise ValueError(f"{method} needs unlabeled data; use a labeled fraction below 1")

    size, workers = data_cfg["image_size"], data_cfg["num_workers"]
    batch = train_cfg["batch_size"]

    def subset(indices: list[int], transform) -> LabeledImages:
        return LabeledImages([paths[i] for i in indices], targets[indices], transform, size)

    # Persistent workers keep each worker's decoded-image cache alive across epochs.
    loader_args = {"num_workers": workers, "persistent_workers": workers > 0}
    labeled_iter = forever(
        DataLoader(subset(split.labeled, weak_transform(size)), batch, shuffle=True, **loader_args)
    )
    val_loader = DataLoader(subset(split.val, eval_transform(size)), batch * 2, **loader_args)
    test_loader = DataLoader(subset(split.test, eval_transform(size)), batch * 2, **loader_args)

    unlabeled_iter = None
    if method != "supervised":
        unlabeled_set = UnlabeledImages(
            [paths[i] for i in split.unlabeled],
            weak_transform(size),
            strong_transform(size, ssl_cfg["randaugment_ops"], ssl_cfg["randaugment_magnitude"]),
            size,
        )
        unlabeled_batch = batch * train_cfg["unlabeled_ratio"]
        unlabeled_iter = forever(
            DataLoader(
                unlabeled_set,
                unlabeled_batch,
                shuffle=True,
                drop_last=len(unlabeled_set) > unlabeled_batch,
                **loader_args,
            )
        )

    model = create_model(model_name, num_classes, train_cfg["pretrained"]).to(device)
    teacher = copy.deepcopy(model).requires_grad_(False) if method == "meanteacher" else None

    steps = train_cfg["steps"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    pos_weight = positive_weights(targets[split.labeled]).to(device)
    rampup_steps = int(ssl_cfg["consistency_rampup"] * steps)

    history: list[dict] = []
    best: dict = {}
    best_state = None
    running_loss, running_mask, window = 0.0, 0.0, 0

    log.info(
        "%s/%s frac=%.2f fold=%d | labeled=%d unlabeled=%d val=%d test=%d | device=%s",
        model_name,
        method,
        fraction,
        fold,
        len(split.labeled),
        len(split.unlabeled),
        len(split.val),
        len(split.test),
        device,
    )

    for step in range(1, steps + 1):
        model.train()
        x, y = (t.to(device) for t in next(labeled_iter))
        mask_rate = 0.0
        with torch.autocast(device_type=device.type, enabled=amp):
            if method == "supervised":
                logits_x = model(x)
                loss_u, weight_u = torch.zeros((), device=device), 0.0
            else:
                x_weak, x_strong = (t.to(device) for t in next(unlabeled_iter))
                if method == "meanteacher":
                    logits_x, logits_s = model(torch.cat([x, x_strong])).split(
                        [len(x), len(x_strong)]
                    )
                    with torch.no_grad():
                        logits_t = teacher(x_weak)
                    loss_u = ssl.consistency_loss(logits_s, logits_t)
                    weight_u = ssl_cfg["consistency_weight"] * ssl.sigmoid_rampup(
                        step, rampup_steps
                    )
                else:
                    logits_x, logits_w, logits_s = model(torch.cat([x, x_weak, x_strong])).split(
                        [len(x), len(x_weak), len(x_strong)]
                    )
                    pseudo, mask = ssl.fixmatch_targets(logits_w, ssl_cfg["threshold"])
                    loss_u = ssl.masked_bce(logits_s, pseudo, mask)
                    weight_u = ssl_cfg["unlabeled_weight"]
                    mask_rate = mask.mean().item()

            loss_x = F.binary_cross_entropy_with_logits(logits_x, y, pos_weight=pos_weight)
            loss = loss_x + weight_u * loss_u

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if teacher is not None:
            ssl.ema_update(teacher, model, ssl_cfg["ema_decay"])

        running_loss += loss.item()
        running_mask += mask_rate
        window += 1

        if step % train_cfg["eval_every"] == 0 or step == steps:
            # Mean Teacher is evaluated through its teacher, as in the original paper.
            eval_model = teacher if teacher is not None else model
            val = scores(*predict(eval_model, val_loader, device))
            entry = {
                "step": step,
                "train_loss": running_loss / window,
                "mask_rate": running_mask / window,
                **{f"val_{k}": v for k, v in val.items()},
            }
            history.append(entry)
            log.info(
                "step %4d | loss %.4f | mask %.2f | val AUROC %.4f | val F1 %.4f",
                step,
                entry["train_loss"],
                entry["mask_rate"],
                val["macro_auroc"],
                val["macro_f1"],
            )
            running_loss, running_mask, window = 0.0, 0.0, 0
            if best_state is None or val["macro_auroc"] > best["macro_auroc"]:
                best = {**val, "step": step}
                best_state = {
                    k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()
                }

    eval_model = teacher if teacher is not None else model
    eval_model.load_state_dict(best_state)
    y_true, probs = predict(eval_model, test_loader, device)
    test = scores(y_true, probs)
    log.info(
        "test AUROC %.4f | F1 %.4f | AP %.4f (best val step %d)",
        test["macro_auroc"],
        test["macro_f1"],
        test["macro_ap"],
        best["step"],
    )

    y_pred = (probs >= 0.5).astype(int)
    result = {
        "model": model_name,
        "method": method,
        "fraction": fraction,
        "fold": fold,
        "classes": classes,
        "split_sizes": {k: len(v) for k, v in vars(split).items()},
        "best_step": best["step"],
        **{f"val_{k}": v for k, v in best.items() if k != "step"},
        **{f"test_{k}": v for k, v in test.items()},
        "test_per_class": {
            name: {"auroc": auroc, "f1": f1, "positives": int(pos)}
            for name, auroc, f1, pos in zip(
                classes,
                per_label(roc_auc_score, y_true, probs),
                f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
                y_true.sum(axis=0),
                strict=True,
            )
        },
        "confusion_matrices": multilabel_confusion_matrix(y_true, y_pred).tolist(),
        "history": history,
        "config": cfg,
        "seconds": round(time.time() - started, 1),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # NaN (an undefined per-class AUROC) is written as null to keep the JSON standard.
    text = json.dumps(result, indent=2, default=float).replace("NaN", "null")
    (out_dir / "metrics.json").write_text(text, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="folder with one sub-folder per class")
    parser.add_argument("--method", choices=METHODS, default="fixmatch")
    parser.add_argument("--model", default="efficientnet_b2", help="alias or any timm model name")
    parser.add_argument("--fraction", type=float, default=0.5, help="share of training labels kept")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default="runs/single")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(
        load_config(args.config),
        data_dir=args.data_dir,
        method=args.method,
        model_name=args.model,
        fraction=args.fraction,
        fold=args.fold,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
