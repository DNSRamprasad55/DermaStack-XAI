"""Diagnose identical predictions for different images in DermaStack-XAI.

Run from the project root in the same Python environment as the app:
  py audit_model_outputs.py path\to\image1.jpg path\to\image2.jpg

This script is diagnostic only. It does not modify or retrain model artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import HAM10000_CLASSES
from src.inference.pipeline import (
    validate_and_load_image,
    resize_and_preprocess,
    load_all_fold_models,
    build_stacking_feature_vector,
    load_stacking_meta_model,
)

MODELS = ("custom_cnn", "efficientnet_b0", "resnet50")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def probs_for_model(image_rgb: np.ndarray, model_name: str, fold_models: list):
    x = resize_and_preprocess(image_rgb, model_name)
    tensor_bytes = np.ascontiguousarray(x).tobytes()
    per_fold = []
    for fold_idx, model in enumerate(fold_models):
        p = np.asarray(model.predict(x, verbose=0), dtype=np.float32).reshape(-1)
        if p.shape != (len(HAM10000_CLASSES),):
            raise ValueError(f"{model_name} fold {fold_idx}: unexpected output shape {p.shape}")
        if not np.all(np.isfinite(p)):
            raise ValueError(f"{model_name} fold {fold_idx}: non-finite probability output")
        p = np.clip(p, 0.0, 1.0)
        p /= max(float(p.sum()), 1e-12)
        per_fold.append(p)
    mean_p = np.mean(per_fold, axis=0).astype(np.float32)
    mean_p /= max(float(mean_p.sum()), 1e-12)
    return {
        "tensor_sha256": sha256_bytes(tensor_bytes),
        "tensor_shape": list(x.shape),
        "tensor_min": float(x.min()),
        "tensor_max": float(x.max()),
        "tensor_mean": float(x.mean()),
        "tensor_std": float(x.std()),
        "fold_probabilities": [p.tolist() for p in per_fold],
        "ensemble_probabilities": mean_p,
    }


def top(p):
    idx = int(np.argmax(p))
    return {"class": HAM10000_CLASSES[idx], "probability": float(p[idx])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs=2, help="Two different image paths to compare")
    args = parser.parse_args()
    paths = [Path(p).expanduser().resolve() for p in args.images]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Image not found: {p}")

    images = []
    image_info = []
    for path in paths:
        raw = path.read_bytes()
        rgb, original_size = validate_and_load_image(raw)
        images.append(rgb)
        image_info.append({
            "path": str(path),
            "file_sha256": sha256_bytes(raw),
            "rgb_pixel_sha256": sha256_bytes(np.ascontiguousarray(rgb).tobytes()),
            "original_size_wh": list(original_size),
            "rgb_shape_hwc": list(rgb.shape),
            "pixel_min": int(rgb.min()),
            "pixel_max": int(rgb.max()),
            "pixel_mean": float(rgb.mean()),
            "pixel_std": float(rgb.std()),
        })

    print("\n=== INPUTS ===")
    print(json.dumps(image_info, indent=2))
    if image_info[0]["rgb_pixel_sha256"] == image_info[1]["rgb_pixel_sha256"]:
        print("WARNING: decoded RGB pixels are identical; choose two genuinely different images.")

    fold_models = {}
    for name in MODELS:
        print(f"Loading {name} folds...", flush=True)
        fold_models[name] = load_all_fold_models(name)

    results = []
    for img in images:
        r = {name: probs_for_model(img, name, fold_models[name]) for name in MODELS}
        x_stack = build_stacking_feature_vector(
            r["custom_cnn"]["ensemble_probabilities"],
            r["efficientnet_b0"]["ensemble_probabilities"],
            r["resnet50"]["ensemble_probabilities"],
        )
        std = load_stacking_meta_model(False).predict_proba(x_stack).reshape(-1)
        bal = load_stacking_meta_model(True).predict_proba(x_stack).reshape(-1)
        r["standard_stacking"] = {"probabilities": std.tolist(), "top": top(std)}
        r["balanced_stacking"] = {"probabilities": bal.tolist(), "top": top(bal)}
        for name in MODELS:
            r[name]["top"] = top(r[name]["ensemble_probabilities"])
            r[name]["ensemble_probabilities"] = r[name]["ensemble_probabilities"].tolist()
        results.append(r)

    print("\n=== PER-IMAGE MODEL OUTPUTS ===")
    print(json.dumps(results, indent=2))
    print("\n=== BETWEEN-IMAGE DIFFERENCES ===")
    diffs = {}
    for name in MODELS:
        a, b = results[0][name], results[1][name]
        fold_diffs = [
            float(np.max(np.abs(np.asarray(pa) - np.asarray(pb))))
            for pa, pb in zip(a["fold_probabilities"], b["fold_probabilities"])
        ]
        diffs[name] = {
            "preprocessed_tensor_hashes_differ": a["tensor_sha256"] != b["tensor_sha256"],
            "max_abs_diff_per_fold": fold_diffs,
            "max_abs_diff_ensemble": float(np.max(np.abs(
                np.asarray(a["ensemble_probabilities"]) - np.asarray(b["ensemble_probabilities"])
            ))),
        }
    for name in ("standard_stacking", "balanced_stacking"):
        pa = np.asarray(results[0][name]["probabilities"])
        pb = np.asarray(results[1][name]["probabilities"])
        diffs[name] = {"max_abs_diff": float(np.max(np.abs(pa - pb)))}
    print(json.dumps(diffs, indent=2))
    print("\nInterpretation: identical tensor hashes indicate preprocessing/input path issue; different tensors but identical fold outputs indicate model/checkpoint/output issue; changing base outputs but unchanged stacker output points to meta-model behavior. This is a technical diagnostic, not a medical validation.")

if __name__ == "__main__":
    main()
