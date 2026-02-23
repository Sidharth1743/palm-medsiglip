from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONJUNCTIVA_NORMAL_PROMPTS = [
    "normal healthy conjunctiva",
    "no conjunctival pallor",
    "no evidence of anemia",
    "normal conjunctival appearance",
    "conjunctiva without pallor",
]

CONJUNCTIVA_ANEMIA_PROMPTS = [
    "conjunctival pallor indicative of anemia",
    "pale conjunctiva suggestive of anemia",
    "there is conjunctival pallor",
    "findings consistent with anemia",
    "anemic conjunctival appearance",
]

FINGERNAILS_NORMAL_PROMPTS = [
    "a photo of a healthy pink fingernail bed",
    "normal healthy fingernail tissue",
    "healthy pink nail bed",
    "normal fingernail bed color",
]

FINGERNAILS_ANEMIA_PROMPTS = [
    "a photo of a pale fingernail bed indicative of anemia",
    "pale fingernail tissue suggestive of anemia",
    "pale nail bed indicative of anemia",
    "fingernail bed pallor due to anemia",
]

PALMS_NORMAL_PROMPTS = [
    "a photo of a healthy pink palm",
    "normal healthy palmar skin color",
    "healthy palm with normal coloration",
    "normal palmar appearance without pallor",
]

PALMS_ANEMIA_PROMPTS = [
    "a photo of a pale palm indicative of anemia",
    "palmar pallor suggestive of anemia",
    "pale palm with reduced coloration",
    "palm pallor consistent with anemia",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedSigLIP zero-shot baseline for anemia vs non-anemia."
    )
    parser.add_argument(
        "--input-csv",
        default="Dataset/dataset anemia/test.csv",
        help="CSV with columns image_path,label",
    )
    parser.add_argument("--model-dir", default="medsiglip", help="Local model directory.")
    parser.add_argument(
        "--output-dir",
        default="results/baseline_zero_shot",
        help="Directory for predictions and status reports.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Force device selection. 'auto' picks cuda if available.",
    )
    parser.add_argument(
        "--prompt-set",
        choices=["conjunctiva", "fingernails", "palms"],
        default="conjunctiva",
        help="Prompt set for zero-shot baseline.",
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
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_rows(csv_path: Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{csv_path} must contain columns image_path,label")
        for row in reader:
            rows.append((resolve_path(row["image_path"]), int(row["label"])))
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")
    return rows


def load_image_with_fallback(path: Path) -> tuple[Image.Image | None, str]:
    if not path.exists():
        return None, "missing"
    try:
        return Image.open(path).convert("RGB"), "used_pillow"
    except Exception:
        cv_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if cv_img is None:
            return None, "unreadable"
        if cv_img.dtype == np.uint16:
            cv_img = ((cv_img.astype(np.float32) / 65535.0) * 255.0).clip(0, 255).astype(
                np.uint8
            )
        elif cv_img.dtype != np.uint8:
            cv_img = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if cv_img.ndim == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_img), "used_opencv_fallback"


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True)


def extract_fingernail_roi(
    np_img: np.ndarray, min_area_ratio: float, padding: float
) -> np.ndarray:
    h, w = np_img.shape[:2]
    if h == 0 or w == 0:
        return np_img
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
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


def main() -> None:
    args = parse_args()
    input_csv = resolve_path(args.input_csv)
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda:0"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Using device={device}, dtype={dtype}, model_dir={model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=dtype).to(device)
    model.eval()

    # Build text prototypes by averaging normalized text embeddings for each class.
    if args.prompt_set == "fingernails":
        class_prompts = {
            "non_anemia": FINGERNAILS_NORMAL_PROMPTS,
            "anemia": FINGERNAILS_ANEMIA_PROMPTS,
        }
    elif args.prompt_set == "palms":
        class_prompts = {
            "non_anemia": PALMS_NORMAL_PROMPTS,
            "anemia": PALMS_ANEMIA_PROMPTS,
        }
    else:
        class_prompts = {
            "non_anemia": CONJUNCTIVA_NORMAL_PROMPTS,
            "anemia": CONJUNCTIVA_ANEMIA_PROMPTS,
        }
    class_proto: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for cls_name, prompts in class_prompts.items():
            text_inputs = processor(text=prompts, padding="max_length", return_tensors="pt").to(
                device
            )
            text_kwargs = {"input_ids": text_inputs["input_ids"]}
            if "attention_mask" in text_inputs:
                text_kwargs["attention_mask"] = text_inputs["attention_mask"]
            text_outputs = model.get_text_features(**text_kwargs)
            text_outputs = l2_normalize(text_outputs)
            proto = text_outputs.mean(dim=0, keepdim=True)
            class_proto[cls_name] = l2_normalize(proto)

    non_anemia_proto = class_proto["non_anemia"]
    anemia_proto = class_proto["anemia"]

    status_rows: list[tuple[str, int, str]] = []
    prediction_rows: list[tuple[str, int, int, float, float, str]] = []
    rescued = 0
    skipped = 0

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        imgs: list[Image.Image] = []
        kept_meta: list[tuple[Path, int, str]] = []
        for path, label in batch:
            img, status = load_image_with_fallback(path)
            status_rows.append((str(path), label, status))
            if img is None:
                skipped += 1
                continue
            if status == "used_opencv_fallback":
                rescued += 1
            if args.roi_mode == "fingernails":
                np_img = np.asarray(img, dtype=np.uint8)
                np_img = extract_fingernail_roi(
                    np_img, args.roi_min_area_ratio, args.roi_padding
                )
                img = Image.fromarray(np_img)
            imgs.append(img)
            kept_meta.append((path, label, status))

        if not imgs:
            if (i // args.batch_size) % 10 == 0:
                print(f"Processed {min(i + args.batch_size, len(rows))}/{len(rows)}")
            continue

        with torch.no_grad():
            img_inputs = processor(images=imgs, return_tensors="pt").to(device)
            image_emb = model.get_image_features(pixel_values=img_inputs["pixel_values"])
            image_emb = l2_normalize(image_emb)

            sim_non_anemia = (image_emb * non_anemia_proto).sum(dim=1)
            sim_anemia = (image_emb * anemia_proto).sum(dim=1)
            logits = torch.stack([sim_non_anemia, sim_anemia], dim=1)
            probs = torch.softmax(logits, dim=1).float().cpu().numpy()

        for (path, label, status), p in zip(kept_meta, probs):
            prob_non_anemia = float(p[0])
            prob_anemia = float(p[1])
            pred = 1 if prob_anemia >= prob_non_anemia else 0
            prediction_rows.append(
                (str(path), int(label), pred, prob_anemia, prob_non_anemia, status)
            )
        if (i // args.batch_size) % 10 == 0:
            print(f"Processed {min(i + args.batch_size, len(rows))}/{len(rows)}")

    pred_csv = output_dir / "baseline_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_path", "label", "pred_label", "prob_anemia", "prob_non_anemia", "status"]
        )
        writer.writerows(prediction_rows)

    status_csv = output_dir / "baseline_image_status.csv"
    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "status"])
        writer.writerows(status_rows)

    run_summary = {
        "total_rows": len(rows),
        "predicted_rows": len(prediction_rows),
        "skipped_rows": skipped,
        "opencv_rescued_rows": rescued,
        "prompts": class_prompts,
    }
    summary_json = output_dir / "baseline_run_summary.json"
    summary_json.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(f"Saved predictions: {pred_csv}")
    print(f"Saved status report: {status_csv}")
    print(f"Saved summary: {summary_json}")
    print(
        f"Counts => total={len(rows)}, predicted={len(prediction_rows)}, skipped={skipped}, opencv_rescued={rescued}"
    )


if __name__ == "__main__":
    main()
