from __future__ import annotations

import argparse
import json
import random
import re
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import torch
import joblib
from PIL import Image, ImageDraw, ImageOps
from sklearn.linear_model import LogisticRegression
import cv2
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transformers import SiglipVisionModel

from optimized.color_constancy import gray_world, white_patch


@dataclass
class Sample:
    path: Path
    label: int
    patient_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimized MedSigLIP linear probe with patient-safe split, C grid search, and threshold calibration."
    )
    parser.add_argument(
        "--data-dir",
        default="Dataset/CP-AnemiC dataset",
        help="Root folder containing class subfolders.",
    )
    parser.add_argument(
        "--class-dirs",
        nargs="+",
        default=["Anemia", "Non-Anemia"],
        help="Class subfolder names under --data-dir (order maps to labels).",
    )
    parser.add_argument(
        "--model-dir",
        default="medsiglip",
        help="Local MedSigLIP model directory.",
    )
    parser.add_argument(
        "--train-csv",
        default=None,
        help="Optional CSV for train split (image_path,label).",
    )
    parser.add_argument(
        "--val-csv",
        default=None,
        help="Optional CSV for val split (image_path,label).",
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Optional CSV for test split (image_path,label).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Console progress logging frequency (in batches).",
    )
    parser.add_argument(
        "--cv-group-subjects",
        action="store_true",
        help="Use GroupKFold in CV based on subject IDs parsed from filenames.",
    )
    parser.add_argument(
        "--cv-threshold",
        action="store_true",
        help="Use CV-based thresholding instead of validation-based thresholding.",
    )
    parser.add_argument(
        "--subject-pooling",
        choices=["none", "mean"],
        default="none",
        help="Pool image embeddings per subject before training/eval (mean).",
    )
    parser.add_argument(
        "--platt-calibration",
        action="store_true",
        help="Apply Platt scaling using validation predictions.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation (eval splits only).",
    )
    parser.add_argument(
        "--tta-angle",
        type=float,
        default=5.0,
        help="Rotation angle for TTA in degrees.",
    )
    parser.add_argument(
        "--tta-views",
        type=int,
        default=8,
        choices=[2, 4, 8],
        help="Number of TTA views to use (2=flip only, 4=flip+rot, 8=full).",
    )
    parser.add_argument(
        "--tta-batch-size",
        type=int,
        default=0,
        help="Override batch size used during TTA (0 = auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
        help="Grid of C values for LogisticRegression.",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.9,
        help="Minimum recall target for threshold calibration on validation.",
    )
    parser.add_argument(
        "--recall-targets",
        nargs="+",
        type=float,
        default=[0.9, 0.93, 0.95],
        help="Recall targets to sweep for threshold calibration on validation.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum iterations for LogisticRegression (epoch-like).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/optimized_linear_probe",
        help="Directory for metrics JSON.",
    )
    parser.add_argument(
        "--save-deploy-dir",
        default=None,
        help="Optional directory to write deployment artifacts.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-entity",
        default="sidhu1743",
        help="Weights & Biases entity/team.",
    )
    parser.add_argument(
        "--wandb-project",
        default="Medgemma",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional wandb tags.",
    )
    parser.add_argument(
        "--no-local-save",
        action="store_true",
        help="Do not write local artifacts/metrics (W&B only).",
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        help="Force local artifacts even when W&B is enabled.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="letterbox",
        help="Resize mode: stretch to 448x448 or letterbox padding.",
    )
    parser.add_argument(
        "--color-constancy",
        choices=["none", "gray_world", "white_patch"],
        default="none",
        help="Optional color constancy method applied before resize.",
    )
    parser.add_argument(
        "--white-patch-percentile",
        type=float,
        default=95.0,
        help="Percentile for white patch color constancy.",
    )
    parser.add_argument(
        "--color-constancy-prob",
        type=float,
        default=1.0,
        help="Probability of applying color constancy to train images (0-1).",
    )
    parser.add_argument(
        "--color-constancy-all",
        action="store_true",
        help="Apply color constancy to train/val/test (not just train).",
    )
    parser.add_argument(
        "--roi-mode",
        choices=["none", "fingernails"],
        default="none",
        help="Optional ROI extraction before preprocessing.",
    )
    parser.add_argument(
        "--roi-min-area-ratio",
        type=float,
        default=0.02,
        help="Minimum ROI area ratio to accept (relative to image area).",
    )
    parser.add_argument(
        "--roi-padding",
        type=float,
        default=0.05,
        help="Padding ratio applied around the ROI crop.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If >0, run K-fold CV (stratified) on train split and report mean/std AUC.",
    )
    parser.add_argument(
        "--white-ref-norm",
        action="store_true",
        help="Normalize RGB using white reference region (best for fingernail board images).",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def iter_images(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def infer_patient_id(path: Path, class_dir: Path) -> str:
    rel = path.relative_to(class_dir)
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    return path.stem


def load_samples(data_dir: Path, class_dirs: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for label, class_name in enumerate(class_dirs):
        class_path = data_dir / class_name
        if not class_path.exists():
            raise FileNotFoundError(f"Missing class folder: {class_path}")
        for img_path in iter_images(class_path):
            pid = infer_patient_id(img_path, class_path)
            samples.append(Sample(img_path, label, pid))
    if not samples:
        raise ValueError(f"No images found under {data_dir}")
    return samples


def load_csv_samples(path: Path) -> list[Sample]:
    import csv
    rows: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not {"image_path", "label"}.issubset(set(r.fieldnames or [])):
            raise ValueError(f"{path} must contain image_path,label columns")
        for row in r:
            img_path = Path(row["image_path"]).expanduser()
            if not img_path.is_absolute():
                img_path = PROJECT_ROOT / img_path
            rows.append(Sample(img_path.resolve(), int(row["label"]), img_path.stem))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _init_wandb(args: argparse.Namespace) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "wandb is enabled but not installed. Install it with `uv sync`."
        ) from exc
    config = {
        "data_dir": str(args.data_dir),
        "class_dirs": args.class_dirs,
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "c_grid": args.c_grid,
        "recall_target": float(args.recall_target),
        "recall_targets": args.recall_targets,
        "color_constancy": args.color_constancy,
        "white_patch_percentile": float(args.white_patch_percentile),
        "color_constancy_prob": float(args.color_constancy_prob),
        "roi_mode": args.roi_mode,
        "roi_min_area_ratio": float(args.roi_min_area_ratio),
        "roi_padding": float(args.roi_padding),
        "white_ref_norm": bool(args.white_ref_norm),
        "max_iter": int(args.max_iter),
        "model_dir": str(args.model_dir),
    }
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=config,
        tags=args.wandb_tags,
    )


def patient_safe_split(
    samples: list[Sample], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    patient_ids = sorted({s.patient_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train : n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val :])
    train = [s for s in samples if s.patient_id in train_ids]
    val = [s for s in samples if s.patient_id in val_ids]
    test = [s for s in samples if s.patient_id in test_ids]
    return train, val, test


def _letterbox_tf(img: tf.Tensor, size: int = 448) -> tf.Tensor:
    with tf.device("/CPU:0"):
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        scale = tf.minimum(
            tf.cast(size, tf.float32) / tf.cast(h, tf.float32),
            tf.cast(size, tf.float32) / tf.cast(w, tf.float32),
        )
        new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
        resized = tf.image.resize(img, [new_h, new_w], method="bilinear", antialias=False)
        pad_h = size - new_h
        pad_w = size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = tf.pad(
            resized, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0
        )
        return padded


def _extract_fingernail_roi(
    np_img: np.ndarray, min_area_ratio: float, padding: float
) -> np.ndarray:
    h, w = np_img.shape[:2]
    if h == 0 or w == 0:
        return np_img

    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    # Low saturation + high value tends to capture nail bed regions
    mask = cv2.inRange(hsv, (0, 0, 60), (180, 80, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if not contours:
        return np_img

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < (min_area_ratio * h * w):
        return np_img

    x, y, cw, ch = cv2.boundingRect(largest)
    pad = int(min(h, w) * padding)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + cw + pad)
    y1 = min(h, y + ch + pad)
    return np_img[y0:y1, x0:x1]

def _white_reference_normalize(np_img: np.ndarray) -> np.ndarray:
    # Estimate white reference from low-saturation, high-value pixels.
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 200)
    if mask.sum() < 100:
        ref = np.median(np_img.reshape(-1, 3), axis=0)
    else:
        ref = np.median(np_img[mask], axis=0)
    ref = np.clip(ref, 1.0, 255.0)
    ref_mean = float(ref.mean())
    img = np_img.astype(np.float32) * (ref_mean / ref)
    return np.clip(img, 0, 255).astype(np.uint8)


def preprocess_images(
    imgs: list[Image.Image],
    device: str,
    dtype: torch.dtype,
    resize_mode: str,
    color_constancy: str,
    white_patch_percentile: float,
    roi_mode: str,
    roi_min_area_ratio: float,
    roi_padding: float,
    white_ref_norm: bool,
    cc_apply: list[bool] | None = None,
) -> torch.Tensor:
    processed: list[torch.Tensor] = []
    for idx, img in enumerate(imgs):
        img = img.convert("RGB")
        np_img = np.asarray(img, dtype=np.uint8)
        if roi_mode == "fingernails":
            np_img = _extract_fingernail_roi(np_img, roi_min_area_ratio, roi_padding)
        if white_ref_norm:
            np_img = _white_reference_normalize(np_img)
        apply_cc = True
        if cc_apply is not None:
            apply_cc = cc_apply[idx]
        if apply_cc:
            if color_constancy == "gray_world":
                np_img = gray_world(np_img)
            elif color_constancy == "white_patch":
                np_img = white_patch(np_img, percentile=white_patch_percentile)
        with tf.device("/CPU:0"):
            tf_img = tf.convert_to_tensor(np_img)
            if resize_mode == "letterbox":
                tf_img = _letterbox_tf(tf_img, size=448)
            else:
                tf_img = tf.image.resize(
                    tf_img, [448, 448], method="bilinear", antialias=False
                )
            tf_img = tf.cast(tf_img, tf.float32) / 255.0
            tf_img = tf_img * 2.0 - 1.0
            np_out = tf_img.numpy()
        tensor = torch.from_numpy(np_out).permute(2, 0, 1)
        processed.append(tensor)
    batch = torch.stack(processed, dim=0).to(device=device, dtype=dtype)
    return batch


def _load_image_for_viz(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        cv_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if cv_img is None:
            return None
        if cv_img.dtype == np.uint16:
            cv_img = ((cv_img.astype(np.float32) / 65535.0) * 255.0).clip(
                0, 255
            ).astype(np.uint8)
        elif cv_img.dtype != np.uint8:
            cv_img = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
        if cv_img.ndim == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_img)


def _make_tf_viz_images(img: Image.Image, resize_mode: str) -> dict[str, Image.Image]:
    rgb = img.convert("RGB")
    np_img = np.asarray(rgb, dtype=np.uint8)
    tf_img = tf.convert_to_tensor(np_img)
    if resize_mode == "letterbox":
        tf_resized = _letterbox_tf(tf_img, size=448)
    else:
        tf_resized = tf.image.resize(
            tf_img,
            [448, 448],
            method="bilinear",
            antialias=False,
        )
    tf_float = tf.cast(tf_resized, tf.float32) / 255.0
    tf_norm = tf_float * 2.0 - 1.0
    resized = Image.fromarray(tf_resized.numpy().astype(np.uint8))
    norm_vis = ((tf_norm + 1.0) / 2.0 * 255.0).numpy().clip(0, 255).astype(np.uint8)
    normalized = Image.fromarray(norm_vis)
    return {
        "original": img,
        "rgb": rgb,
        "resized_448": resized,
        "normalized": normalized,
    }


def extract_embeddings(
    samples: list[Sample],
    model: torch.nn.Module,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    split_name: str,
    log_every: int,
    resize_mode: str,
    color_constancy: str,
    white_patch_percentile: float,
    roi_mode: str,
    roi_min_area_ratio: float,
    roi_padding: float,
    color_constancy_prob: float,
    rng: random.Random,
    white_ref_norm: bool,
    tta: bool,
    tta_angle: float,
    tta_batch_size: int,
    tta_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_embeds: list[np.ndarray] = []
    all_labels: list[int] = []
    total = len(samples)
    processed = 0
    effective_batch = batch_size
    if tta:
        if tta_batch_size > 0:
            effective_batch = tta_batch_size
        else:
            effective_batch = max(1, batch_size // 8)
        if split_name in {"val", "test"} and effective_batch != batch_size:
            print(f"[{split_name}] TTA enabled, batch size adjusted to {effective_batch}")

    for i in range(0, len(samples), effective_batch):
        batch = samples[i : i + effective_batch]
        imgs: list[Image.Image] = []
        kept_labels: list[int] = []
        for s in batch:
            img = _load_image_for_viz(s.path)
            if img is None:
                continue
            if tta:
                views = _tta_generate_views(img, tta_angle)
                if tta_views == 2:
                    views = views[:2]
                elif tta_views == 4:
                    views = views[:4]
                imgs.extend(views)
                kept_labels.append(s.label)
            else:
                imgs.append(img)
                kept_labels.append(s.label)
        if not imgs:
            continue
        tta_factor = tta_views if tta else 1
        cc_apply = None
        if color_constancy != "none" and color_constancy_prob > 0:
            cc_apply = [rng.random() < color_constancy_prob for _ in imgs]
        with torch.no_grad():
            pixel_values = preprocess_images(
                imgs,
                device,
                dtype,
                resize_mode,
                color_constancy,
                white_patch_percentile,
                roi_mode,
                roi_min_area_ratio,
                roi_padding,
                white_ref_norm,
                cc_apply,
            )
            outputs = model(pixel_values=pixel_values)
            embeddings = outputs.pooler_output
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        if tta:
            emb = embeddings.float().cpu().numpy()
            emb = emb.reshape(-1, tta_factor, emb.shape[-1]).mean(axis=1)
            all_embeds.append(emb)
        else:
            all_embeds.append(embeddings.float().cpu().numpy())
        all_labels.extend(kept_labels)
        processed += len(kept_labels)
        if log_every > 0 and ((i // effective_batch + 1) % log_every == 0 or processed == total):
            print(f"[{split_name}] Processed {processed}/{total}")
    x = np.concatenate(all_embeds, axis=0)
    y = np.asarray(all_labels, dtype=np.int64)
    return x, y


def subject_id_from_name(name: str) -> str:
    stem = Path(name).stem
    pattern = re.compile(
        r"^(anemic|non[- ]?anemic)[-_]?(?P<sid>[^ (]+)", re.IGNORECASE
    )
    match = pattern.match(stem)
    if match:
        prefix = match.group(1).lower().replace(" ", "-")
        return f"{prefix}:{match.group('sid')}"
    for sep in (" (", "(", " "):
        if sep in stem:
            return stem.split(sep, 1)[0].lower()
    return stem.lower()


def build_subject_groups(samples: list[Sample]) -> list[str]:
    return [subject_id_from_name(s.path.name) for s in samples]


def pool_by_subject(
    x: np.ndarray, y: np.ndarray, groups: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    by_subject: dict[str, list[int]] = {}
    for idx, sid in enumerate(groups):
        by_subject.setdefault(sid, []).append(idx)

    pooled_x: list[np.ndarray] = []
    pooled_y: list[int] = []
    pooled_groups: list[str] = []
    for sid, idxs in by_subject.items():
        subj_x = x[idxs]
        pooled_x.append(subj_x.mean(axis=0))
        pooled_y.append(int(y[idxs[0]]))
        pooled_groups.append(sid)
    return np.vstack(pooled_x), np.asarray(pooled_y, dtype=np.int64), pooled_groups


def _tta_scale(img: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return img
    w, h = img.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    if scale > 1.0:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    pad_w = (w - new_w) // 2
    pad_h = (h - new_h) // 2
    return ImageOps.expand(
        resized,
        border=(pad_w, pad_h, w - new_w - pad_w, h - new_h - pad_h),
        fill=(0, 0, 0),
    )


def _tta_translate(img: Image.Image, dx: int, dy: int) -> Image.Image:
    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, dx, 0, 1, dy),
        resample=Image.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def _tta_generate_views(img: Image.Image, angle: float) -> list[Image.Image]:
    w, h = img.size
    dx = int(round(w * 0.05))
    views = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.rotate(angle, resample=Image.BILINEAR, expand=False),
        img.rotate(-angle, resample=Image.BILINEAR, expand=False),
        _tta_scale(img, 1.05),
        _tta_scale(img, 0.95),
        _tta_translate(img, dx, 0),
        _tta_translate(img, -dx, 0),
    ]
    return views


def fit_logreg(
    x_train: np.ndarray, y_train: np.ndarray, c_val: float, max_iter: int
) -> tuple[dict[str, np.ndarray], LogisticRegression]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train_std = (x_train - mean) / std
    model = LogisticRegression(
        solver="saga",
        C=c_val,
        max_iter=max_iter,
        class_weight="balanced",
    )
    model.fit(x_train_std, y_train)
    scaler = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return scaler, model


def fit_platt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int,
) -> LogisticRegression:
    platt = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=max_iter,
        class_weight="balanced",
    )
    platt.fit(x_val, y_val)
    return platt


def predict_prob(
    x: np.ndarray, scaler: dict[str, np.ndarray], model: LogisticRegression
) -> np.ndarray:
    x_std = (x - scaler["mean"]) / scaler["std"]
    probs = model.predict_proba(x_std)[:, 1]
    return probs.astype(np.float32)


def apply_platt(
    probs: np.ndarray, platt: LogisticRegression
) -> np.ndarray:
    return platt.predict_proba(probs.reshape(-1, 1))[:, 1].astype(np.float32)


def run_cv(
    x: np.ndarray,
    y: np.ndarray,
    c_grid: list[float],
    max_iter: int,
    cv_folds: int,
    groups: list[str] | None = None,
) -> dict[str, Any]:
    if groups is None:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        splits = splitter.split(x, y)
    else:
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=cv_folds)
        splits = splitter.split(x, y, groups)
    results: dict[str, dict[str, list[float]]] = {
        str(c): {"auc": [], "accuracy": [], "precision": [], "recall": [], "f1": []} for c in c_grid
    }
    for train_idx, val_idx in splits:
        x_tr, x_va = x[train_idx], x[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        for c in c_grid:
            scaler, model = fit_logreg(x_tr, y_tr, c, max_iter)
            val_probs = predict_prob(x_va, scaler, model)
            auc = roc_auc(y_va, val_probs)
            metrics = compute_metrics(y_va, val_probs, threshold=0.5)
            results[str(c)]["auc"].append(float(auc))
            results[str(c)]["accuracy"].append(float(metrics["accuracy"]))
            results[str(c)]["precision"].append(float(metrics["precision"]))
            results[str(c)]["recall"].append(float(metrics["recall"]))
            results[str(c)]["f1"].append(float(metrics["f1"]))

    summary = {}
    for c, metrics in results.items():
        summary[c] = {
            "mean_auc": float(np.mean(metrics["auc"])),
            "std_auc": float(np.std(metrics["auc"])),
            "mean_accuracy": float(np.mean(metrics["accuracy"])),
            "std_accuracy": float(np.std(metrics["accuracy"])),
            "mean_precision": float(np.mean(metrics["precision"])),
            "std_precision": float(np.std(metrics["precision"])),
            "mean_recall": float(np.mean(metrics["recall"])),
            "std_recall": float(np.std(metrics["recall"])),
            "mean_f1": float(np.mean(metrics["f1"])),
            "std_f1": float(np.std(metrics["f1"])),
            "folds": metrics,
        }
    return summary


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    return float(np.trapezoid(tpr, fpr))


def roc_threshold_stats(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {
            "best_threshold": 0.5,
            "best_tpr": float("nan"),
            "best_fpr": float("nan"),
            "best_j": float("nan"),
        }
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    scores_sorted = y_score[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_thresh = float(scores_sorted[best_idx])
    return {
        "best_threshold": best_thresh,
        "best_tpr": float(tpr[best_idx]),
        "best_fpr": float(fpr[best_idx]),
        "best_j": float(j[best_idx]),
    }


def save_deploy_artifacts(
    deploy_dir: Path,
    vision_model: SiglipVisionModel,
    probe: LogisticRegression,
    scaler: dict[str, np.ndarray],
    threshold: float,
    best_c: float,
    args: argparse.Namespace,
) -> None:
    deploy_dir.mkdir(parents=True, exist_ok=True)
    vision_model.save_pretrained(deploy_dir / "vision_model")

    coef = probe.coef_.reshape(-1).astype(np.float32)
    intercept = float(probe.intercept_.reshape(-1)[0])
    linear_head = torch.nn.Linear(vision_model.config.hidden_size, 1, bias=True)
    with torch.no_grad():
        linear_head.weight.copy_(torch.from_numpy(coef).unsqueeze(0))
        linear_head.bias.copy_(torch.tensor([intercept], dtype=torch.float32))
    torch.save(linear_head.state_dict(), deploy_dir / "linear_head.pt")

    joblib.dump(scaler, deploy_dir / "scaler.joblib")
    config = {
        "threshold": float(threshold),
        "best_c": float(best_c),
        "recall_target": float(args.recall_target),
        "class_names": ["Non-Anemia", "Anemia"],
        "resize_mode": args.resize_mode,
        "color_constancy": args.color_constancy,
        "white_patch_percentile": float(args.white_patch_percentile),
        "roi_mode": args.roi_mode,
        "roi_min_area_ratio": float(args.roi_min_area_ratio),
        "roi_padding": float(args.roi_padding),
        "white_ref_norm": bool(args.white_ref_norm),
    }
    (deploy_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    src_opt = PROJECT_ROOT / "src" / "optimized"
    dst_opt = deploy_dir / "optimized"
    if src_opt.exists():
        shutil.copytree(src_opt, dst_opt, dirs_exist_ok=True)


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, steps: int = 101) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(steps):
        t = i / (steps - 1)
        rows.append(compute_metrics(y_true, y_prob, threshold=float(t)) | {"threshold": t})
    return rows


def best_threshold_for_recall(
    y_true: np.ndarray, y_prob: np.ndarray, recall_target: float
) -> float:
    thresholds = np.linspace(0, 1, 101)
    best_t = 0.5
    best_precision = -1.0
    for t in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=float(t))
        if metrics["recall"] >= recall_target:
            if metrics["precision"] > best_precision:
                best_precision = metrics["precision"]
                best_t = float(t)
    return best_t


def cv_best_threshold(
    x: np.ndarray,
    y: np.ndarray,
    c_val: float,
    max_iter: int,
    recall_target: float,
    cv_folds: int,
    groups: list[str] | None = None,
) -> float:
    thresholds: list[float] = []
    if groups is None:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        splits = splitter.split(x, y)
    else:
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=cv_folds)
        splits = splitter.split(x, y, groups)

    for train_idx, val_idx in splits:
        x_tr, x_va = x[train_idx], x[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        scaler, model = fit_logreg(x_tr, y_tr, c_val, max_iter)
        val_probs = predict_prob(x_va, scaler, model)
        t = best_threshold_for_recall(y_va, val_probs, recall_target)
        thresholds.append(float(t))

    if not thresholds:
        return 0.5
    return float(np.median(thresholds))


def sweep_recall_targets(
    y_true: np.ndarray, y_prob: np.ndarray, recall_targets: list[float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in recall_targets:
        threshold = best_threshold_for_recall(y_true, y_prob, target)
        metrics = compute_metrics(y_true, y_prob, threshold=threshold)
        rows.append(
            {
                "recall_target": float(target),
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.wandb and not args.save_local:
        args.no_local_save = True
    data_dir = resolve_path(args.data_dir)
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)
    if not args.no_local_save:
        output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(args)

    if args.train_csv and args.val_csv and args.test_csv:
        train = load_csv_samples(resolve_path(args.train_csv))
        val = load_csv_samples(resolve_path(args.val_csv))
        test = load_csv_samples(resolve_path(args.test_csv))
    else:
        samples = load_samples(data_dir, args.class_dirs)
        train, val, test = patient_safe_split(
            samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    vision_model = SiglipVisionModel.from_pretrained(model_dir, dtype=dtype).to(device)
    vision_model.eval()

    rng = random.Random(args.seed)
    train_cc_prob = args.color_constancy_prob if args.color_constancy != "none" else 0.0
    eval_cc_prob = train_cc_prob if args.color_constancy_all else 0.0
    x_train, y_train = extract_embeddings(
        train,
        vision_model,
        device,
        dtype,
        args.batch_size,
        "train",
        args.log_every,
        args.resize_mode,
        args.color_constancy,
        args.white_patch_percentile,
        args.roi_mode,
        args.roi_min_area_ratio,
        args.roi_padding,
        train_cc_prob,
        rng,
        args.white_ref_norm,
        False,
        args.tta_angle,
        args.tta_batch_size,
        args.tta_views,
    )
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    x_val, y_val = extract_embeddings(
        val,
        vision_model,
        device,
        dtype,
        args.batch_size,
        "val",
        args.log_every,
        args.resize_mode,
        args.color_constancy,
        args.white_patch_percentile,
        args.roi_mode,
        args.roi_min_area_ratio,
        args.roi_padding,
        eval_cc_prob,
        rng,
        args.white_ref_norm,
        args.tta,
        args.tta_angle,
        args.tta_batch_size,
        args.tta_views,
    )
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    x_test, y_test = extract_embeddings(
        test,
        vision_model,
        device,
        dtype,
        args.batch_size,
        "test",
        args.log_every,
        args.resize_mode,
        args.color_constancy,
        args.white_patch_percentile,
        args.roi_mode,
        args.roi_min_area_ratio,
        args.roi_padding,
        eval_cc_prob,
        rng,
        args.white_ref_norm,
        args.tta,
        args.tta_angle,
        args.tta_batch_size,
        args.tta_views,
    )

    train_groups = build_subject_groups(train)
    val_groups = build_subject_groups(val)
    test_groups = build_subject_groups(test)

    if args.subject_pooling == "mean":
        x_train, y_train, train_groups = pool_by_subject(x_train, y_train, train_groups)
        x_val, y_val, val_groups = pool_by_subject(x_val, y_val, val_groups)
        x_test, y_test, test_groups = pool_by_subject(x_test, y_test, test_groups)

    best_c = None
    best_auc = -1.0
    best_scaler = None
    best_model = None
    c_rows: list[dict[str, Any]] = []
    cv_summary = None
    cv_threshold = None
    if args.cv_folds and args.cv_folds > 1:
        cv_groups = train_groups if args.cv_group_subjects else None
        cv_summary = run_cv(
            x_train,
            y_train,
            args.c_grid,
            args.max_iter,
            args.cv_folds,
            groups=cv_groups,
        )
        if not args.no_local_save:
            (output_dir / "cv_summary.json").write_text(
                json.dumps(cv_summary, indent=2), encoding="utf-8"
            )
    for c in args.c_grid:
        scaler, model = fit_logreg(x_train, y_train, c, args.max_iter)
        val_probs = predict_prob(x_val, scaler, model)
        auc = roc_auc(y_val, val_probs)
        c_rows.append({"c": float(c), "val_auc": float(auc)})
        if auc > best_auc:
            best_auc = auc
            best_c = c
            best_scaler = scaler
            best_model = model

    assert best_model is not None and best_scaler is not None and best_c is not None

    val_probs = predict_prob(x_val, best_scaler, best_model)
    test_probs = predict_prob(x_test, best_scaler, best_model)

    platt_model = None
    if args.platt_calibration:
        platt_model = fit_platt(
            x_train,
            y_train,
            val_probs.reshape(-1, 1),
            y_val,
            args.max_iter,
        )
        val_probs = apply_platt(val_probs, platt_model)
        test_probs = apply_platt(test_probs, platt_model)

    val_auc = roc_auc(y_val, val_probs)
    test_auc = roc_auc(y_test, test_probs)

    if args.cv_threshold and args.cv_folds and args.cv_folds > 1:
        cv_groups = train_groups if args.cv_group_subjects else None
        cv_threshold = cv_best_threshold(
            x_train,
            y_train,
            best_c,
            args.max_iter,
            args.recall_target,
            args.cv_folds,
            groups=cv_groups,
        )
        threshold = cv_threshold
    else:
        threshold = best_threshold_for_recall(y_val, val_probs, args.recall_target)
    test_metrics = compute_metrics(y_test, test_probs, threshold=threshold)
    roc_stats = roc_threshold_stats(y_val, val_probs)
    sweep_rows = threshold_sweep(y_val, val_probs, steps=101)
    best_thresholds: dict[str, dict[str, float]] = {}
    for metric_name in ("accuracy", "precision", "recall", "f1"):
        best_row = max(sweep_rows, key=lambda r: r[metric_name])
        best_thresholds[metric_name] = {
            "threshold": float(best_row["threshold"]),
            metric_name: float(best_row[metric_name]),
        }
    best_f1_threshold = best_thresholds["f1"]["threshold"]
    test_metrics_best_f1 = compute_metrics(
        y_test, test_probs, threshold=best_f1_threshold
    )
    recall_sweep = sweep_recall_targets(y_val, val_probs, args.recall_targets)
    best_recall_row = max(recall_sweep, key=lambda r: r["f1"])

    summary = {
        "data": {
            "train_images": len(train),
            "val_images": len(val),
            "test_images": len(test),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "class_dirs": args.class_dirs,
            "roi_mode": args.roi_mode,
            "roi_min_area_ratio": args.roi_min_area_ratio,
            "roi_padding": args.roi_padding,
        },
        "cv_folds": int(args.cv_folds),
        "cv_summary": cv_summary,
        "cv_threshold": cv_threshold,
        "platt_calibration": bool(args.platt_calibration),
        "best_c": best_c,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "threshold": threshold,
        "metrics_at_threshold": test_metrics,
        "metrics_at_best_f1_threshold_test": test_metrics_best_f1,
        "best_f1_threshold": best_f1_threshold,
        "recall_target": args.recall_target,
        "recall_target_sweep": recall_sweep,
        "recall_target_best_by_f1": best_recall_row,
        "roc_threshold": roc_stats,
        "best_thresholds": best_thresholds,
    }

    if not args.no_local_save:
        out_path = output_dir / "summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log(
            {
                "metrics/val_auc": val_auc,
                "metrics/test_auc": test_auc,
                "metrics/threshold": threshold,
                "metrics/threshold_accuracy": test_metrics["accuracy"],
                "metrics/threshold_precision": test_metrics["precision"],
                "metrics/threshold_recall": test_metrics["recall"],
                "metrics/threshold_f1": test_metrics["f1"],
                "metrics/best_f1_threshold": best_f1_threshold,
                "metrics/best_f1_threshold_accuracy": test_metrics_best_f1["accuracy"],
                "metrics/best_f1_threshold_precision": test_metrics_best_f1["precision"],
                "metrics/best_f1_threshold_recall": test_metrics_best_f1["recall"],
                "metrics/best_f1_threshold_f1": test_metrics_best_f1["f1"],
                "metrics/confusion_matrix": test_metrics["confusion_matrix"],
                "metrics/roc_best_threshold": roc_stats["best_threshold"],
                "metrics/roc_best_j": roc_stats["best_j"],
                "metrics/roc_best_tpr": roc_stats["best_tpr"],
                "metrics/roc_best_fpr": roc_stats["best_fpr"],
                "metrics/best_threshold_accuracy": best_thresholds["accuracy"]["threshold"],
                "metrics/best_threshold_precision": best_thresholds["precision"]["threshold"],
                "metrics/best_threshold_recall": best_thresholds["recall"]["threshold"],
                "metrics/best_threshold_f1": best_thresholds["f1"]["threshold"],
                "metrics/recall_sweep_best_target": best_recall_row["recall_target"],
                "metrics/recall_sweep_best_threshold": best_recall_row["threshold"],
                "metrics/recall_sweep_best_accuracy": best_recall_row["accuracy"],
                "metrics/recall_sweep_best_precision": best_recall_row["precision"],
                "metrics/recall_sweep_best_recall": best_recall_row["recall"],
                "metrics/recall_sweep_best_f1": best_recall_row["f1"],
            }
        )
        try:
            import wandb  # type: ignore

            cm = test_metrics["confusion_matrix"]
            cm_img = Image.new("RGB", (760, 640), color=(255, 255, 255))
            draw = ImageDraw.Draw(cm_img)
            draw.text((24, 20), "Optimized Confusion Matrix", fill=(0, 0, 0))
            grid_left, grid_top = 150, 110
            cell = 170
            labels = ["Non-Anemia", "Anemia"]
            arr = np.asarray(cm, dtype=np.float32)
            min_v = float(arr.min())
            max_v = float(arr.max())
            norm = (arr - min_v) / (max_v - min_v + 1e-6)
            for r in range(2):
                for c in range(2):
                    val = int(arr[r, c])
                    shade = int(255 - 170 * float(norm[r, c]))
                    color = (shade, shade, 255)
                    x0 = grid_left + c * cell
                    y0 = grid_top + r * cell
                    x1 = x0 + cell
                    y1 = y0 + cell
                    draw.rectangle(
                        [(x0, y0), (x1, y1)],
                        fill=color,
                        outline=(80, 80, 80),
                        width=2,
                    )
                    draw.text((x0 + 70, y0 + 78), str(val), fill=(0, 0, 0))
            draw.text((grid_left + 120, grid_top + 365), "Predicted", fill=(0, 0, 0))
            draw.text((35, grid_top + 140), "True", fill=(0, 0, 0))
            for i, lbl in enumerate(labels):
                draw.text((grid_left + i * cell + 42, grid_top - 28), lbl, fill=(0, 0, 0))
                draw.text((grid_left - 110, grid_top + i * cell + 78), lbl, fill=(0, 0, 0))
            wandb_run.log({"plots/confusion_matrix": wandb.Image(cm_img)})
            c_table = wandb.Table(columns=["C", "val_auc"], data=[[r["c"], r["val_auc"]] for r in c_rows])
            wandb_run.log({"tables/c_grid": c_table})
            try:
                wandb_run.log(
                    {"plots/val_auc_vs_c": wandb.plot.line(c_table, "C", "val_auc", title="Val AUC vs C")}
                )
            except Exception:
                pass
            sweep_table = wandb.Table(
                columns=["threshold", "accuracy", "precision", "recall", "f1"],
                data=[
                    [r["threshold"], r["accuracy"], r["precision"], r["recall"], r["f1"]]
                    for r in sweep_rows
                ],
            )
            wandb_run.log({"tables/threshold_sweep": sweep_table})
            recall_table = wandb.Table(
                columns=["recall_target", "threshold", "accuracy", "precision", "recall", "f1"],
                data=[
                    [
                        r["recall_target"],
                        r["threshold"],
                        r["accuracy"],
                        r["precision"],
                        r["recall"],
                        r["f1"],
                    ]
                    for r in recall_sweep
                ],
            )
            wandb_run.log({"tables/recall_target_sweep": recall_table})
        except Exception:
            pass
        try:
            import wandb  # type: ignore

            train_samples = train[:2]
            test_samples = test[:2]
            for s, split in (
                [(x, "train") for x in train_samples] + [(x, "test") for x in test_samples]
            ):
                img = _load_image_for_viz(s.path)
                if img is None:
                    continue
                viz = _make_tf_viz_images(img, args.resize_mode)
                wandb_run.log(
                    {
                        f"images/{split}_{s.path.name}_original": wandb.Image(viz["original"]),
                        f"images/{split}_{s.path.name}_rgb": wandb.Image(viz["rgb"]),
                        f"images/{split}_{s.path.name}_resized": wandb.Image(viz["resized_448"]),
                        f"images/{split}_{s.path.name}_normalized": wandb.Image(viz["normalized"]),
                    }
                )
        except Exception:
            pass
        wandb_run.finish()

    if args.save_deploy_dir:
        deploy_dir = resolve_path(args.save_deploy_dir)
        save_deploy_artifacts(
            deploy_dir,
            vision_model,
            best_model,
            best_scaler,
            threshold,
            best_c,
            args,
        )

    print(json.dumps(summary, indent=2))
    if not args.no_local_save:
        print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
