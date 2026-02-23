from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
import torch
import cv2
from PIL import Image
from transformers import SiglipVisionModel

from palm_deploy.optimized.color_constancy import gray_world, white_patch


class PalmAnemiaPredictor:
    def __init__(self, artifact_dir: str | None = None) -> None:
        base = artifact_dir or os.environ.get("ARTIFACT_DIR")
        if not base:
            base = "/app/artifacts"
        self.artifact_dir = Path(base)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.vision_model = SiglipVisionModel.from_pretrained(
            self.artifact_dir / "vision_model"
        ).to(self.device)
        self.vision_model.eval()

        self.linear_head = torch.nn.Linear(self.vision_model.config.hidden_size, 1, bias=True)
        state = torch.load(self.artifact_dir / "linear_head.pt", map_location=self.device)
        self.linear_head.load_state_dict(state)
        self.linear_head.to(self.device)
        self.linear_head.eval()

        self.scaler: dict[str, np.ndarray] = joblib.load(self.artifact_dir / "scaler.joblib")
        with (self.artifact_dir / "config.json").open() as f:
            self.config = json.load(f)

        self.threshold = float(self.config.get("threshold", 0.5))
        self.resize_mode = self.config.get("resize_mode", "letterbox")
        self.color_constancy = self.config.get("color_constancy", "none")
        self.white_patch_percentile = float(self.config.get("white_patch_percentile", 95.0))
        self.white_ref_norm = bool(self.config.get("white_ref_norm", False))

    def _letterbox_tf(self, img: tf.Tensor, size: int = 448) -> tf.Tensor:
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

    @staticmethod
    def _white_reference_normalize(np_img: np.ndarray) -> np.ndarray:
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

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        np_img = np.asarray(img, dtype=np.uint8)

        if self.white_ref_norm:
            np_img = self._white_reference_normalize(np_img)

        if self.color_constancy == "gray_world":
            np_img = gray_world(np_img)
        elif self.color_constancy == "white_patch":
            np_img = white_patch(np_img, percentile=self.white_patch_percentile)

        with tf.device("/CPU:0"):
            tf_img = tf.convert_to_tensor(np_img)
            if self.resize_mode == "letterbox":
                tf_img = self._letterbox_tf(tf_img, size=448)
            else:
                tf_img = tf.image.resize(tf_img, [448, 448], method="bilinear", antialias=False)
            tf_img = tf.cast(tf_img, tf.float32) / 255.0
            tf_img = tf_img * 2.0 - 1.0
            np_out = tf_img.numpy()
        tensor = torch.from_numpy(np_out).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        pixel_values = self.preprocess(image)
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=pixel_values)
            embeds = outputs.pooler_output
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            mean = torch.tensor(self.scaler["mean"], device=self.device)
            std = torch.tensor(self.scaler["std"], device=self.device)
            embeds_std = (embeds - mean) / std
            logit = self.linear_head(embeds_std)
            score = torch.sigmoid(logit).item()

        pred = "Anemia" if score >= self.threshold else "Non-Anemia"
        return {
            "triage_score": float(score),
            "prediction": pred,
            "threshold": float(self.threshold),
            "class_names": ["Non-Anemia", "Anemia"],
        }
