#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_ml_pipeline.py

Run the full ML-based OMR pipeline:

input images
-> marker-based warping
-> checkbox ROI extraction
-> SVM classification
-> CSV export
-> overlay generation
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd

try:
    from skimage.feature import hog
except ImportError as exc:
    raise SystemExit("Please install scikit-image: pip install scikit-image") from exc


# ------------------------------------------------------------
# Feature parameters
# ------------------------------------------------------------
DARK_WEIGHT = 4.0
INNER_PAD_RATIO = 0.12


# ------------------------------------------------------------
# Subprocess helpers
# ------------------------------------------------------------
def run_warp(
    warp_script: Path,
    marker_model: Path,
    input_dir: Path,
    warped_out_dir: Path,
    *,
    imgsz: int,
    conf: float,
    out_w: int,
    out_h: int,
) -> None:
    warped_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(warp_script),
        "--model",
        str(marker_model),
        "--input_dir",
        str(input_dir),
        "--out_dir",
        str(warped_out_dir),
        "--imgsz",
        str(imgsz),
        "--conf",
        str(conf),
        "--out_w",
        str(out_w),
        "--out_h",
        str(out_h),
    ]

    print("\n[WARP] " + " ".join(cmd))
    subprocess.check_call(cmd)


def collect_warped_images(warped_out_dir: Path) -> List[Path]:
    return sorted(warped_out_dir.glob("*_warped.jpg"))


def run_roi_extract(
    roi_script: Path,
    warped_dir: Path,
    checkbox_model: Path,
    rois_dir: Path,
    meta_csv: Path,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    min_box: int,
    max_box: int,
    topk_per_sheet: int,
    rows: int,
    cols: int,
    dedup_mode: str,
    min_center_dist_px: float,
    grid_dedup_margin: float,
) -> None:
    rois_dir.mkdir(parents=True, exist_ok=True)
    meta_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(roi_script),
        "--warped_dir",
        str(warped_dir),
        "--model",
        str(checkbox_model),
        "--rois_dir",
        str(rois_dir),
        "--meta_csv",
        str(meta_csv),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--imgsz",
        str(imgsz),
        "--dedup_mode",
        str(dedup_mode),
        "--min_center_dist_px",
        str(min_center_dist_px),
        "--grid_dedup_margin",
        str(grid_dedup_margin),
        "--min_box",
        str(min_box),
        "--max_box",
        str(max_box),
        "--topk_per_sheet",
        str(topk_per_sheet),
        "--rows",
        str(rows),
        "--cols",
        str(cols),
    ]

    print("\n[ROI]  " + " ".join(cmd))
    subprocess.check_call(cmd)


def run_overlay(
    overlay_script: Path,
    meta_csv: Path,
    results_csv: Path,
    warped_dir: Path,
    out_dir: Path,
    *,
    show_labels: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(overlay_script),
        "--meta_csv",
        str(meta_csv),
        "--results_csv",
        str(results_csv),
        "--warped_dir",
        str(warped_dir),
        "--out_dir",
        str(out_dir),
    ]

    if show_labels:
        cmd.append("--show_labels")

    print("\n[OVERLAY] " + " ".join(cmd))
    subprocess.check_call(cmd)


# ------------------------------------------------------------
# CSV / join helpers
# ------------------------------------------------------------
def normalize_path(path_str: str) -> str:
    return str(Path(path_str)).replace("\\", "/").lower()


def write_results_csv(out_csv: Path, results: List[Dict]) -> None:
    if not results:
        raise ValueError("No results available for writing.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sheet_id",
        "row",
        "col",
        "pred",
        "margin",
        "dark_ratio_inner",
        "roi_path",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    for key in results[0].keys():
        if key not in fieldnames:
            fieldnames.append(key)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[OK] Results CSV saved to: {out_csv}")


def enrich_results_with_meta(meta_csv: Path, preds: List[Dict]) -> List[Dict]:
    meta_df = pd.read_csv(meta_csv)
    pred_df = pd.DataFrame(preds)

    if pred_df.empty:
        return preds

    meta_df["roi_key"] = meta_df["roi_path"].astype(str).map(normalize_path)
    pred_df["roi_key"] = pred_df["roi_path"].astype(str).map(normalize_path)

    merged = meta_df.merge(
        pred_df.drop(columns=["roi_path"], errors="ignore"),
        on="roi_key",
        how="left",
    ).drop(columns=["roi_key"], errors="ignore")

    sort_columns = [col for col in ["sheet_id", "row", "col"] if col in merged.columns]
    if sort_columns:
        merged = merged.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    column_order = [
        "sheet_id",
        "row",
        "col",
        "pred",
        "margin",
        "dark_ratio_inner",
        "roi_path",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    column_order += [col for col in merged.columns if col not in column_order]
    merged = merged[column_order]

    if "pred" in merged.columns:
        missing = int(pd.isna(merged["pred"]).sum())
        if missing:
            print(f"[WARN] {missing} rows have no prediction after ROI-path merge.")

    return merged.to_dict(orient="records")


# ------------------------------------------------------------
# SVM feature extraction
# ------------------------------------------------------------
def compute_dark_ratio_inner(img_gray_64: np.ndarray) -> float:
    height, width = img_gray_64.shape
    pad = int(INNER_PAD_RATIO * min(height, width))
    pad = max(1, min(pad, min(height, width) // 4))

    inner = img_gray_64[pad : height - pad, pad : width - pad]
    blurred = cv2.GaussianBlur(inner, (3, 3), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return float(np.mean(thresholded == 0))


def roi_to_feature(
    roi_path: Path,
    *,
    img_size: int,
    hog_params: dict,
    add_dark_ratio: bool,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    image = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None

    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    hog_feature = hog(image, **hog_params).astype(np.float32)

    if not add_dark_ratio:
        return hog_feature, None

    dark_ratio_inner = compute_dark_ratio_inner(image)
    dark_weighted = float(dark_ratio_inner * DARK_WEIGHT)
    feature = np.hstack(
        [hog_feature, np.array([dark_weighted], dtype=np.float32)]
    ).astype(np.float32)

    return feature, dark_ratio_inner


def predict_rois_local(
    roi_paths: List[Path],
    *,
    model_file: Path,
    return_margin: bool = True,
) -> List[Dict]:
    if not model_file.exists():
        raise FileNotFoundError(f"SVM model not found: {model_file}")

    pack = joblib.load(model_file)
    model = pack["model"]
    scaler = pack["scaler"]
    img_size = int(pack.get("img_size", 64))
    hog_params = pack.get(
        "hog_params",
        {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "block_norm": "L2-Hys",
        },
    )

    expected_features = getattr(scaler, "n_features_in_", None)
    add_dark_ratio = expected_features == 1765

    print(f"\n[SVM] {model_file}")
    print(
        f"      img_size={img_size} | expected_features={expected_features} "
        f"| add_dark_ratio={add_dark_ratio}"
    )
    if add_dark_ratio:
        print(f"      DARK_WEIGHT={DARK_WEIGHT} | INNER_PAD_RATIO={INNER_PAD_RATIO}")

    features: List[np.ndarray] = []
    valid_paths: List[Path] = []
    dark_ratio_values: List[Optional[float]] = []
    failed = 0

    for roi_path in roi_paths:
        feature, dark_ratio_inner = roi_to_feature(
            roi_path,
            img_size=img_size,
            hog_params=hog_params,
            add_dark_ratio=add_dark_ratio,
        )
        if feature is None:
            failed += 1
            continue

        features.append(feature)
        valid_paths.append(roi_path)
        dark_ratio_values.append(dark_ratio_inner)

    if not features:
        raise RuntimeError("No valid ROIs available for SVM prediction.")

    X = np.vstack(features)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled).astype(int)

    margins = None
    if return_margin:
        try:
            margins = model.decision_function(X_scaled)
        except Exception:
            margins = None

    output: List[Dict] = []
    for i, roi_path in enumerate(valid_paths):
        row = {
            "roi_path": str(roi_path).replace("\\", "/"),
            "pred": int(y_pred[i]),
            "dark_ratio_inner": dark_ratio_values[i],
        }

        if return_margin and margins is not None:
            margin = margins[i]
            if isinstance(margin, (np.ndarray, list)) and np.array(margin).ndim > 0:
                margin = float(np.array(margin).ravel()[0])
            row["margin"] = float(margin)

        output.append(row)

    if failed:
        print(f"[WARN] {failed} ROIs could not be loaded and were skipped.")

    return output


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full ML-based OMR pipeline: warp, ROI extraction, "
            "SVM classification, CSV export, and overlay generation."
        )
    )

    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--marker_model", required=True, type=str)
    parser.add_argument("--checkbox_model", required=True, type=str)
    parser.add_argument("--svm_model", required=True, type=str)

    parser.add_argument("--warp_imgsz", type=int, required=True)
    parser.add_argument("--warp_conf", type=float, required=True)
    parser.add_argument("--out_w", type=int, required=True)
    parser.add_argument("--out_h", type=int, required=True)

    parser.add_argument("--roi_imgsz", type=int, required=True)
    parser.add_argument("--roi_conf", type=float, required=True)
    parser.add_argument("--roi_iou", type=float, required=True)

    parser.add_argument(
        "--dedup_mode",
        type=str,
        required=True,
        choices=["none", "center", "grid", "center+grid"],
    )
    parser.add_argument("--min_center_dist_px", type=float, required=True)
    parser.add_argument("--grid_dedup_margin", type=float, required=True)

    parser.add_argument("--min_box", type=int, required=True)
    parser.add_argument("--max_box", type=int, required=True)
    parser.add_argument("--topk_per_sheet", type=int, required=True)

    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    input_dir = Path(args.input_dir)
    output_base = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_base.mkdir(parents=True, exist_ok=True)

    marker_model = Path(args.marker_model)
    checkbox_model = Path(args.checkbox_model)
    svm_model = Path(args.svm_model)

    for required_path in [marker_model, checkbox_model, svm_model]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    warp_script = base_dir / "warp.py"
    roi_script = base_dir / "extract_rois.py"
    overlay_script = base_dir / "overlay.py"

    for required_script in [warp_script, roi_script, overlay_script]:
        if not required_script.exists():
            raise FileNotFoundError(f"Required script not found: {required_script}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_base / f"run_{timestamp}"

    warped_out = run_dir / "warped"
    rois_dir = run_dir / "rois"
    meta_csv = run_dir / "rois_meta.csv"
    results_csv = run_dir / "results.csv"
    overlay_dir = run_dir / "overlays"

    run_warp(
        warp_script=warp_script,
        marker_model=marker_model,
        input_dir=input_dir,
        warped_out_dir=warped_out,
        imgsz=int(args.warp_imgsz),
        conf=float(args.warp_conf),
        out_w=int(args.out_w),
        out_h=int(args.out_h),
    )

    warped_images = collect_warped_images(warped_out)
    if not warped_images:
        print("[ERROR] No warped images were generated. Aborting.")
        return 2

    run_roi_extract(
        roi_script=roi_script,
        warped_dir=warped_out,
        checkbox_model=checkbox_model,
        rois_dir=rois_dir,
        meta_csv=meta_csv,
        conf=float(args.roi_conf),
        iou=float(args.roi_iou),
        imgsz=int(args.roi_imgsz),
        dedup_mode=str(args.dedup_mode),
        min_center_dist_px=float(args.min_center_dist_px),
        grid_dedup_margin=float(args.grid_dedup_margin),
        min_box=int(args.min_box),
        max_box=int(args.max_box),
        topk_per_sheet=int(args.topk_per_sheet),
        rows=int(args.rows),
        cols=int(args.cols),
    )

    roi_paths = sorted(rois_dir.glob("*.png"))
    if not roi_paths:
        print("[ERROR] No ROIs were generated. Aborting.")
        return 3

    preds = predict_rois_local(roi_paths, model_file=svm_model, return_margin=True)
    preds_with_meta = enrich_results_with_meta(meta_csv, preds)
    write_results_csv(results_csv, preds_with_meta)

    run_overlay(
        overlay_script=overlay_script,
        meta_csv=meta_csv,
        results_csv=results_csv,
        warped_dir=warped_out,
        out_dir=overlay_dir,
        show_labels=True,
    )

    print("\n[OK] Finished.")
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())