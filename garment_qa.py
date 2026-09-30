"""Garment colour-accuracy check for generated product/model images.

Why: diffusion models drift on colour. Before an image goes on a product page we compare the
dominant colour of the garment region against the brand's official hex (CIELAB distance, dE76).

    python scripts/garment_qa.py image.png --target "#1F3A5F"            # heuristic torso crop
    python scripts/garment_qa.py image.png --target "#1F3A5F" --mask garment_mask.png
Rule of thumb: dE < 10 close, 10-20 noticeable, > 20 wrong colour. Thresholds are tunable.
"""
from __future__ import annotations
import argparse, sys
import cv2, numpy as np


def hex_to_lab(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    rgb = np.array([[[int(h[i:i + 2], 16) for i in (0, 2, 4)]]], dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0].astype(float)


def dominant_lab(img_bgr: np.ndarray, mask: np.ndarray | None, k: int = 4) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    px = lab[mask > 0] if mask is not None else lab.reshape(-1, 3)
    px = np.float32(px)
    if len(px) < k:
        raise ValueError("Region too small for analysis")
    _, labels, centers = cv2.kmeans(px, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    biggest = np.bincount(labels.flatten()).argmax()
    return centers[biggest].astype(float)


def torso_mask(shape) -> np.ndarray:
    """Heuristic garment region for a full/3-4 body portrait: central chest/torso band."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    m[int(h * .28):int(h * .55), int(w * .32):int(w * .68)] = 255
    return m


def check(image_path: str, target_hex: str, mask_path: str | None = None, threshold: float = 15.0) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    if mask_path:
        mask = cv2.resize(cv2.imread(mask_path, 0), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = torso_mask(img.shape)
    found = dominant_lab(img, mask)
    de = float(np.linalg.norm(found - hex_to_lab(target_hex)))
    return {"image": image_path, "target": target_hex, "delta_e": round(de, 1),
            "status": "PASS" if de <= threshold else "REVIEW", "region": "mask" if mask_path else "torso-heuristic"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--target", required=True)
    ap.add_argument("--mask"); ap.add_argument("--threshold", type=float, default=15.0)
    a = ap.parse_args()
    res = check(a.image, a.target, a.mask, a.threshold)
    print(res); sys.exit(0 if res["status"] == "PASS" else 1)
