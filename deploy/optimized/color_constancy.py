from __future__ import annotations

import numpy as np


def gray_world(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    img_f = img.astype(np.float32)
    mean = img_f.mean(axis=(0, 1)) + 1e-6
    gray = float(mean.mean())
    scale = gray / mean
    out = img_f * scale
    return np.clip(out, 0, 255).astype(np.uint8)


def white_patch(img: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    img_f = img.astype(np.float32)
    flat = img_f.reshape(-1, 3)
    white = np.percentile(flat, percentile, axis=0) + 1e-6
    scale = 255.0 / white
    out = img_f * scale
    return np.clip(out, 0, 255).astype(np.uint8)
