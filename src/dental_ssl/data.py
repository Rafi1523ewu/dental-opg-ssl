"""Dataset discovery, leak-free cross-validation splits and image pipelines.

The Kaggle release stores each panoramic X-ray once per condition it shows: the same file
sits in several class folders. Files are therefore merged by content into one multi-label
sample, and every split is made over unique images, so an X-ray can never appear on both
sides of a split. Content, not file name, is the key: one name in the release is shared by
two different images.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def scan_multilabel(
    root: str | Path, exclude: list[str] | tuple[str, ...] = ()
) -> tuple[list[Path], np.ndarray, list[str]]:
    """Merge one-folder-per-class copies into unique images with a 0/1 label vector each.

    Folders listed in `exclude` are not used as targets, but their images are kept.
    """
    return _scan(Path(root), tuple(exclude))


@lru_cache(maxsize=4)  # a grid process scans the same folder for every run
def _scan(root: Path, exclude: tuple[str, ...]) -> tuple[list[Path], np.ndarray, list[str]]:
    folders = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not folders:
        raise FileNotFoundError(f"No class folders found in {root}")
    classes = [f for f in folders if f not in exclude]

    path_by_hash: dict[str, Path] = {}
    folders_by_hash: dict[str, set[str]] = {}
    for folder in folders:
        for path in sorted((root / folder).rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                path_by_hash.setdefault(digest, path)
                folders_by_hash.setdefault(digest, set()).add(folder)
    if not path_by_hash:
        raise FileNotFoundError(f"No images found under {root}")

    keys = sorted(path_by_hash, key=lambda k: (path_by_hash[k].name, k))
    targets = np.array([[c in folders_by_hash[k] for c in classes] for k in keys], dtype=np.float32)
    return [path_by_hash[k] for k in keys], targets, classes


def stratification_key(targets: np.ndarray) -> np.ndarray:
    """Each image's rarest positive label (-1 if none).

    Stratifying on it spreads the rare conditions evenly across folds without needing
    a dedicated multi-label stratification library.
    """
    frequency = targets.sum(axis=0)
    rarity = np.where(targets == 1, frequency, np.inf)
    return np.where(targets.any(axis=1), rarity.argmin(axis=1), -1)


@dataclass(frozen=True)
class Split:
    """Indices into the scanned images. The four groups never overlap."""

    labeled: list[int]
    unlabeled: list[int]
    val: list[int]
    test: list[int]


def make_split(
    targets: np.ndarray,
    *,
    fold: int,
    n_folds: int,
    val_size: float,
    labeled_fraction: float,
    seed: int,
) -> Split:
    """Fold `fold` of a stratified K-fold is the test set; the rest is split into
    validation and training, and only `labeled_fraction` of training keeps its labels.

    The test fold does not depend on the labeled fraction, so methods and fractions are
    always compared on exactly the same images.
    """
    if not 0 < labeled_fraction <= 1:
        raise ValueError(f"labeled_fraction must be in (0, 1], got {labeled_fraction}")
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")

    key = stratification_key(targets)
    idx = np.arange(len(key))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    trainval, test = list(folds.split(idx, key))[fold]
    train, val = train_test_split(
        trainval, test_size=val_size, stratify=key[trainval], random_state=seed
    )
    if labeled_fraction == 1:
        labeled, unlabeled = train, np.array([], dtype=int)
    else:
        labeled, unlabeled = train_test_split(
            train, train_size=labeled_fraction, stratify=key[train], random_state=seed
        )
    return Split(*(sorted(int(i) for i in part) for part in (labeled, unlabeled, val, test)))


def eval_transform(size: int) -> T.Compose:
    return T.Compose(
        [T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def weak_transform(size: int) -> T.Compose:
    # Mild geometry only: panoramic X-rays have a fixed layout and orientation cues.
    return T.Compose(
        [
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def strong_transform(size: int, num_ops: int, magnitude: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.RandAugment(num_ops=num_ops, magnitude=magnitude),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.5, scale=(0.02, 0.15)),
        ]
    )


@cache
def load_image(path: Path, size: int) -> Image.Image:
    """Decoded and resized once per process; every augmentation starts from this copy.

    Decoding the ~12 MP phone photos was the training bottleneck, and every pipeline
    resizes to (size, size) first anyway, so caching the resized image changes nothing else.
    """
    image = Image.open(path)
    image.draft("RGB", (size * 2, size * 2))  # let the JPEG decoder downscale early
    return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


class LabeledImages(Dataset):
    def __init__(self, paths: list[Path], targets: np.ndarray, transform, size: int):
        self.paths, self.targets, self.transform, self.size = paths, targets, transform, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.transform(load_image(self.paths[i], self.size)), self.targets[i]


class UnlabeledImages(Dataset):
    """Returns a weakly and a strongly augmented view of the same image."""

    def __init__(self, paths: list[Path], weak, strong, size: int):
        self.paths, self.weak, self.strong, self.size = paths, weak, strong, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        image = load_image(self.paths[i], self.size)
        return self.weak(image), self.strong(image)
