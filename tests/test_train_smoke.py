"""End-to-end run of every method on a tiny synthetic multi-label dataset and tiny model."""

import json

import numpy as np
import pytest
from PIL import Image

from dental_ssl.train import METHODS, load_config, run

CLASSES = ["a", "b", "c"]


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """60 images; each is copied into every class folder it belongs to, like the real data."""
    root = tmp_path_factory.mktemp("opg")
    for name in [*CLASSES, "healthy"]:
        (root / name).mkdir()
    rng = np.random.default_rng(0)
    for i in range(60):
        image = Image.fromarray(rng.integers(0, 255, (40, 60, 3), dtype=np.uint8))
        for c, name in enumerate(CLASSES):
            if i % 3 == c or rng.random() < 0.2:
                image.save(root / name / f"{i}.jpg")
        image.save(root / "healthy" / f"{i}.jpg")
    return root


@pytest.fixture
def cfg():
    cfg = load_config()
    cfg["data"].update(image_size=32, num_workers=0, n_folds=3, exclude_classes=["healthy"])
    cfg["train"].update(steps=4, eval_every=2, batch_size=4, pretrained=False)
    return cfg


@pytest.mark.parametrize("method", METHODS)
def test_method_runs_and_writes_metrics(method, tiny_dataset, cfg, tmp_path):
    result = run(
        cfg,
        data_dir=tiny_dataset,
        method=method,
        model_name="resnet10t",
        fraction=0.5,
        fold=1,
        out_dir=tmp_path,
    )

    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["method"] == method
    assert result["classes"] == CLASSES
    assert sum(result["split_sizes"].values()) == 60
    assert 0.0 <= result["test_macro_auroc"] <= 1.0
    assert len(result["history"]) == 2
    assert np.array(result["confusion_matrices"]).shape == (3, 2, 2)


def test_ssl_method_rejects_fully_labeled_data(tiny_dataset, cfg, tmp_path):
    with pytest.raises(ValueError, match="unlabeled"):
        run(
            cfg,
            data_dir=tiny_dataset,
            method="fixmatch",
            model_name="resnet10t",
            fraction=1.0,
            fold=0,
            out_dir=tmp_path,
        )
