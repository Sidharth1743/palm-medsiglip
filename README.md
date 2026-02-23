# Palm Anemia Detection (MedSigLIP + Linear Probe)

**What it is:** a palm‑image anemia classifier built on a frozen MedSigLIP vision
encoder with a logistic‑regression linear probe. This folder is a self‑contained
palm package with training/evaluation scripts, CSV splits, and Cloud Run
deployment code.

## What we did
- Subject‑level train/val/test splits for palms.
- Tuned linear probe with CV + threshold calibration.
- Explored preprocessing options (letterbox resize, color constancy, white‑ref normalization).
- Deployed a FastAPI service to Cloud Run.

## Folder structure
```
palm/
  main/
    linear_probe_tuned.py       # training + tuning pipeline
  evaluation/
    baseline-model-running.py   # zero-shot baseline inference
    baseline-model-evaluation.py# metrics from baseline predictions
  src_palms_csv/
    train.csv                   # subject-level train split
    val.csv                     # subject-level val split
    test.csv                    # subject-level test split
    all.csv                     # full list (optional)
    build_palm_splits.py         # subject-level split builder
  deploy/
    Dockerfile                  # Cloud Run container
    app.py                      # FastAPI app
    predictor.py                # inference logic
    requirements.txt            # runtime deps
    optimized/
      color_constancy.py        # preprocessing helpers
  palm_results.md               # experiment summary
```

## Data
CSV files contain **absolute image paths** and labels. Images are **not** included
in this repo.

## Model weights
Weights are downloaded separately (e.g., from Hugging Face) and passed via
`--model-dir`. Artifacts/weights are ignored by `.gitignore`.

## Training (linear probe)
Example (full training with CV + tuning):
```bash
uv run python palm/main/linear_probe_tuned.py \
  --train-csv palm/src_palms_csv/train.csv \
  --val-csv palm/src_palms_csv/val.csv \
  --test-csv palm/src_palms_csv/test.csv \
  --model-dir medsiglip \
  --batch-size 16 \
  --max-iter 5000 \
  --cv-folds 4 \
  --wandb --wandb-entity sidhu1743 --wandb-project Medgemma
```

Key options:
- `--resize-mode letterbox`
- `--color-constancy gray_world`
- `--white-ref-norm`
- `--cv-group-subjects` (subject‑aware CV)
- `--tta` / `--tta-views` for test‑time augmentation

## Zero‑shot baseline (MedSigLIP)
Run baseline inference:
```bash
uv run python palm/evaluation/baseline-model-running.py \
  --input-csv palm/src_palms_csv/test.csv \
  --model-dir medsiglip \
  --prompt-set palms
```

Evaluate baseline predictions:
```bash
uv run python palm/evaluation/baseline-model-evaluation.py \
  --predictions-csv results/baseline_zero_shot/baseline_predictions.csv
```

## Deployment (Cloud Run)
Build:
```bash
gcloud builds submit palm/deploy \
  --tag gcr.io/gemini-credits-487316/palm-anemia:latest \
  --project gemini-credits-487316
```

Deploy:
```bash
gcloud run deploy palm-anemia \
  --image gcr.io/gemini-credits-487316/palm-anemia:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --project gemini-credits-487316
```

Test:
```bash
curl -X POST "https://<SERVICE_URL>/predict" \
  -F "file=@/path/to/palm.jpg"
```

## Notes
- Subject‑level splits avoid leakage across train/val/test.
- Deployment artifacts (weights, scaler, config) are **excluded from git**.
