import numpy as np
import pytest
from PIL import Image

from dental_ssl.data import make_split, scan_multilabel, stratification_key

# Label frequencies of the real dataset (231 unique X-rays, 5 condition labels).
RNG = np.random.default_rng(0)
TARGETS = (RNG.random((231, 5)) < np.array([52, 119, 13, 87, 23]) / 231).astype(np.float32)


def split(fold=0, fraction=0.2):
    return make_split(
        TARGETS, fold=fold, n_folds=5, val_size=0.2, labeled_fraction=fraction, seed=42
    )


@pytest.mark.parametrize("fraction", [0.2, 0.3, 0.5, 1.0])
def test_groups_are_disjoint_and_cover_everything(fraction):
    s = split(fraction=fraction)
    groups = [set(s.labeled), set(s.unlabeled), set(s.val), set(s.test)]
    assert sum(map(len, groups)) == len(TARGETS)
    assert set().union(*groups) == set(range(len(TARGETS)))


def test_test_folds_partition_the_dataset():
    tests = [set(split(fold=k).test) for k in range(5)]
    assert sum(map(len, tests)) == len(TARGETS)
    assert set().union(*tests) == set(range(len(TARGETS)))


def test_test_and_val_do_not_depend_on_labeled_fraction():
    assert split(fraction=0.2).test == split(fraction=0.5).test
    assert split(fraction=0.2).val == split(fraction=0.5).val


def test_rare_label_reaches_every_test_fold():
    rare = TARGETS[:, 2] == 1
    assert all(rare[split(fold=k).test].any() for k in range(5))


def test_stratification_key_picks_rarest_positive_label():
    targets = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 0, 1]], dtype=np.float32)
    assert stratification_key(targets).tolist() == [1, 0, -1, 2]


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        split(fraction=0.0)
    with pytest.raises(ValueError):
        split(fold=5)


def test_scan_merges_identical_copies_into_multilabel_samples(tmp_path):
    # Image content is keyed by its gray level; "4.jpg" reuses a name for a different image.
    copies = {
        "Caries": [("1.jpg", 10), ("2.jpg", 20), ("4.jpg", 40)],
        "Infection": [("1.jpg", 10), ("4.jpg", 50)],
        "Healthy": [("1.jpg", 10), ("3.jpg", 30)],
    }
    for folder, files in copies.items():
        (tmp_path / folder).mkdir()
        for name, gray in files:
            Image.new("RGB", (8, 8), (gray, gray, gray)).save(tmp_path / folder / name)
    (tmp_path / "Caries" / "notes.txt").write_text("ignored")

    paths, targets, classes = scan_multilabel(tmp_path, exclude=["Healthy"])
    assert classes == ["Caries", "Infection"]
    assert [p.name for p in paths] == ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "4.jpg"]
    assert sorted(map(tuple, targets.tolist())) == sorted([(1, 1), (1, 0), (0, 0), (1, 0), (0, 1)])
