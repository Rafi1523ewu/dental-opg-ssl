# Semi-Supervised Learning for Dental Panoramic X-ray Classification

[![CI](https://github.com/Rafi1523ewu/dental-opg-ssl/actions/workflows/ci.yml/badge.svg)](https://github.com/Rafi1523ewu/dental-opg-ssl/actions/workflows/ci.yml)

Expert annotation of dental radiographs is slow and expensive, so labeled data is the
bottleneck for diagnostic models. This project measures how much semi-supervised learning
(SSL) recovers when only 20–50% of the training X-rays are labeled, for multi-label detection
of five dental conditions on a small public dataset of panoramic (OPG) X-rays.

| Method | Idea |
|---|---|
| Supervised baseline | Labeled images only, binary cross-entropy weighted per condition |
| [FixMatch](https://arxiv.org/abs/2001.07685) | Per-condition pseudo-labels from a weakly augmented view, used only where the model is confident (p ≥ 0.95 or p ≤ 0.05), learned on a strongly augmented view |
| [Mean Teacher](https://arxiv.org/abs/1703.01780) | Consistency between the student and an exponential-moving-average teacher |

Backbones are ImageNet-pretrained models from `timm`; EfficientNet-B2, ConvNeXt-Tiny and
ViT-B/16 at 224 and 384 px are compared in a validation-only sweep before the main grid.

## Dataset, and why this is a multi-label problem

[Dental OPG X-ray Dataset](https://www.kaggle.com/datasets/orvile/dental-opg-xray-dataset)
(CC BY 4.0): smartphone photographs of panoramic X-rays from three clinics in Bangladesh.
Its classification folder looks like a six-class dataset of 517 images, but it is not:

- 178 file names appear in more than one class folder; for 177 of them the copies are
  **byte-identical**. Merging files by content leaves **232 unique X-rays**, which matches
  the 232 originals the dataset describes: each X-ray is copied into every condition it
  shows. (The remaining shared name belongs to two different images, which is why files are
  merged by content hash rather than by name.)
- Treating it as single-label gives the same image contradictory labels, and a random split
  puts identical files into both training and test sets, which inflates test scores.

This project therefore merges identical files into one sample per X-ray with a label vector,
and splits over unique images only. The check is reproducible: it is the first cell of the Kaggle
notebook, and `tests/test_data.py` covers the merge.

| Condition (target) | X-rays |
|---|---|
| Caries | 119 |
| Impacted Teeth | 87 |
| Broken Down Crown/Root (BDC-BDR) | 52 |
| Infection | 23 |
| Fractured Teeth | 13 |

"Healthy Teeth" is attached to 223 of the 232 X-rays (most panoramic images show some healthy
teeth), so it carries almost no information and is not used as a target.

## Evaluation protocol

- **5-fold cross-validation over unique X-rays.** Every X-ray is in exactly one test fold.
  Folds are stratified on each image's rarest condition, so the 13 fractured-teeth cases are
  spread across folds.
- Within each fold, 20% of the remaining images form a **validation set** used only to pick
  the checkpoint. Of the rest, 20%, 30% or 50% keep their labels; the others become the
  **unlabeled pool**. The test fold is identical for every method and labeled fraction.
- Imbalance is handled with per-condition positive weights in the loss, never by duplicating
  images.
- **Metrics:** macro-AUROC (primary, threshold-free), macro average precision and macro-F1 at
  0.5, reported as mean ± std over the five folds, plus per-condition AUROC and F1.
- The backbone and resolution are chosen by **validation** AUROC only; test scores play no
  part in any choice.

## Results

Run on a Kaggle T4 ×2 via [`kaggle/run_experiments.py`](kaggle/run_experiments.py).

**Backbone sweep** (supervised, 50% labels, folds 0–1). ConvNeXt-Tiny at 384 px had the
best validation AUROC (0.631, level with 224 px) and was used for everything below.
Full table: [`results/sweep_summary.csv`](results/sweep_summary.csv).

**Main grid**: test macro-AUROC, mean ± std over 5 folds.

| Labeled share | Supervised | FixMatch | Mean Teacher |
|---|---|---|---|
| 20% | 0.591 ± 0.054 | 0.588 ± 0.056 | 0.567 ± 0.052 |
| 30% | 0.597 ± 0.048 | 0.591 ± 0.038 | 0.579 ± 0.079 |
| 50% | 0.650 ± 0.051 | 0.651 ± 0.040 | 0.569 ± 0.099 |
| 100% (upper bound) | 0.722 ± 0.072 | – | – |

Macro-AP and macro-F1 tell the same story; all metrics are in
[`results/final_summary.csv`](results/final_summary.csv).

**What this shows**

- **More labels help; unlabeled images, here, do not.** Going from 20% to 100% labels raises
  AUROC from 0.59 to 0.72. FixMatch matches the supervised baseline at every fraction and
  Mean Teacher falls slightly below it, with every difference within about one fold
  standard deviation.
- The likely reason is scale: the unlabeled pool is only 74–119 images per fold, far smaller
  than the settings these methods were designed for, and a 0.95 confidence threshold admits
  few pseudo-labels for the rare conditions.
- Absolute scores are modest (0.72 AUROC with every label): condition-level findings in phone
  photographs of whole panoramic X-rays are a hard target for about 150 training images.

This is a negative result, reported as measured: on this dataset, the cheapest way to a
better model is more labeled X-rays, not semi-supervised training.

## Usage

```bash
git clone https://github.com/Rafi1523ewu/dental-opg-ssl.git
cd dental-opg-ssl
pip install -e ".[dev]"

# Download the data (needs a Kaggle API token in ~/.kaggle/kaggle.json)
kaggle datasets download -d orvile/dental-opg-xray-dataset -p data --unzip
DATA="data/Dental OPG XRAY Dataset/Dental OPG (Classification)"

# One run
python -m dental_ssl.train --data-dir "$DATA" --method fixmatch --model efficientnet_b2 \
    --fraction 0.5 --fold 0 --out-dir runs/example

# A grid (resumable; finished runs are skipped) and its summary table
python scripts/run_grid.py --data-dir "$DATA" --grid configs/sweep.yaml --runs-dir runs/sweep
```

Each run writes `metrics.json` with the split sizes, the validation curve, the share of
pseudo-labels that passed the confidence threshold, test metrics, per-condition results and
confusion matrices. All hyperparameters live in [`configs/default.yaml`](configs/default.yaml).

## Project structure

```
configs/default.yaml        hyperparameters and the default grid
configs/sweep.yaml          backbone / resolution search
src/dental_ssl/data.py      merging duplicated files by content, CV splits, cached augmentations
src/dental_ssl/ssl.py       FixMatch and Mean Teacher objectives for multi-label outputs
src/dental_ssl/train.py     training loop, evaluation, CLI
scripts/run_grid.py         runs a grid (optionally sharded across GPUs), writes a summary CSV
kaggle/run_experiments.py   sweep, then the full grid on the winning backbone
results/                    summary tables from the Kaggle run
tests/                      merge and split invariants, SSL components, end-to-end smoke test
```

## Testing

```bash
pytest -q
```

The smoke test trains every method for a few steps on a synthetic multi-label dataset with a
tiny untrained network, so the whole pipeline is checked on CPU in seconds. CI runs lint and
tests on every push.

## Limitations

- 232 X-rays is small. Fractured Teeth has only 13 positives, about 2–3 per test fold, so its
  per-condition scores are noisy; the fold standard deviations are reported for that reason.
- The dataset publishes no patient IDs. Each X-ray is treated as independent; if a patient
  contributed more than one X-ray, scores may be slightly optimistic.
- Images are phone photographs of radiographs, not native DICOM exports.

## License

Code: MIT. Dataset: CC BY 4.0, by its original authors (see the Kaggle page).
