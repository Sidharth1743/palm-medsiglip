# Palm Results (Subject‑Level Only)

## Subject‑Level Split
- Split: **70/15/15** by subject
- Train images: `3397`
- Val images: `395`
- Test images: `468`
- Labels: `Anemia`, `Non‑Anemia`

## Extra Features Available (Pipeline Options)
- Subject‑level splitting and GroupKFold CV (`--cv-group-subjects`)
- Letterbox resize (no stretching)
- Optional ROI extraction (`--roi-mode`)
- Optional white‑reference normalization (`--white-ref-norm`)
- Optional color constancy (`--color-constancy`)
- TTA with configurable views (`--tta --tta-views 2/4/8`)
- TTA batch override (`--tta-batch-size`)
- Recall‑target thresholding
- Optional CV‑based thresholding (`--cv-threshold`)
- Optional Platt calibration (`--platt-calibration`)
- Console progress logs (`--log-every`)
- W&B logging with plots and tables (`--wandb`)

---

# Subject‑Level Zero‑Shot Baseline (Palms)

Metrics (test set):
- num_examples: `468`
- accuracy: `0.427350`
- precision: `0.666667`
- recall: `0.099291`
- auc: `0.572295`
- f1: `0.172840`
- confusion_matrix: `[[172, 14], [254, 28]]`

Artifacts:
- `results/baseline_palms_subjec/baseline_metrics.json`
- `results/baseline_palms_subjec/baseline_confusion_matrix.png`
- `results/baseline_palms_subjec/baseline_roc_curve.png`
- `results/baseline_palms_subjec/baseline_score_histogram.png`

---

# Subject‑Level Baselines

## 1) Pure Linear Probe (No Color Constancy, No TTA, No Calibration)
- Best C (grid): `0.3`
- CV AUC (mean ± std): `0.721731 ± 0.030607`
- Val AUC: `0.707756`
- Test AUC: `0.774575`
- Best‑F1 threshold: `0.03`
- Metrics at best‑F1 (test):
  - Accuracy `0.690171`
  - Precision `0.679790`
  - Recall `0.918440`
  - F1 `0.781297`
  - Confusion Matrix `[[64, 122], [23, 259]]`

## 2) Pure Linear Probe + Platt Calibration (No Color Constancy, No TTA)
- Best C (grid): `0.3`
- CV AUC (mean ± std): `0.721777 ± 0.030588`
- Val AUC: `0.707625`
- Test AUC: `0.774613`
- Best‑F1 threshold: `0.30`
- Metrics at best‑F1 (test):
  - Accuracy `0.664530`
  - Precision `0.655860`
  - Recall `0.932624`
  - F1 `0.770132`
  - Confusion Matrix `[[48, 138], [19, 263]]`

## Decision
- **Deploy uncalibrated**: higher F1 and better balance.
- **Threshold for deployment**: `0.03`.
