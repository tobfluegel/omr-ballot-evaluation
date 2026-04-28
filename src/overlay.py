#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
overlay.py

Create prediction overlays on warped ballot images.

The script:
- loads ROI metadata and classification results
- matches predictions to ROI bounding boxes via roi_path
- finds the corresponding warped image for each sheet
- draws colored rectangles for predictions
- optionally displays row/column labels

Color convention:
- green: marked
- red: empty
- yellow: prediction missing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def normalize_path(path_str: str) -> str:
    return str(Path(path_str)).replace("\\", "/").lower()


def find_warped_image(warped_dir: Path, sheet_id: str) -> Path | None:
    candidates = list(warped_dir.glob(f"*{sheet_id}*_warped.*"))
    if candidates:
        return candidates[0]

    candidates = list(warped_dir.glob(f"*{sheet_id}*.*"))
    return candidates[0] if candidates else None


def sheet_id_from_roi_path(roi_path: str) -> str:
    """
    Expected ROI naming examples:
    - .../20260131_213720_r01_c01.png
    - .../20260131_213720__003.png
    """
    name = Path(roi_path).stem

    if "_r" in name and "_c" in name:
        return name.split("_r")[0]
    if "__" in name:
        return name.split("__")[0]

    return name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw prediction overlays on warped ballot images."
    )
    parser.add_argument("--meta_csv", required=True, type=str)
    parser.add_argument("--results_csv", required=True, type=str)
    parser.add_argument(
        "--warped_dir",
        required=True,
        type=str,
        help="Directory containing warped images, e.g. *_warped.jpg",
    )
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument(
        "--show_labels",
        action="store_true",
        help="Display row/column labels if available",
    )
    args = parser.parse_args()

    meta_csv = Path(args.meta_csv)
    results_csv = Path(args.results_csv)
    warped_dir = Path(args.warped_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_csv}")
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    if not warped_dir.exists():
        raise FileNotFoundError(f"Warped-image directory not found: {warped_dir}")

    meta = pd.read_csv(meta_csv)
    results = pd.read_csv(results_csv)

    if "roi_path" not in meta.columns:
        raise KeyError("The metadata CSV is missing the column 'roi_path'.")
    if not all(column in meta.columns for column in ["x1", "y1", "x2", "y2"]):
        raise KeyError("The metadata CSV must contain the columns x1, y1, x2, and y2.")

    if "sheet_id" not in meta.columns:
        meta["sheet_id"] = meta["roi_path"].astype(str).map(sheet_id_from_roi_path)

    meta["roi_key"] = meta["roi_path"].astype(str).map(normalize_path)
    results["roi_key"] = results["roi_path"].astype(str).map(normalize_path)

    pred_col = None
    for candidate in ["final_pred", "pred", "prediction", "y_pred"]:
        if candidate in results.columns:
            pred_col = candidate
            break

    if pred_col is None:
        raise KeyError(
            "The results CSV does not contain a prediction column. "
            f"Expected one of: final_pred, pred, prediction, y_pred. "
            f"Available columns: {list(results.columns)}"
        )

    merged = meta.merge(results[["roi_key", pred_col]], on="roi_key", how="left")

    for sheet_id, group in merged.groupby("sheet_id"):
        warped_path = find_warped_image(warped_dir, str(sheet_id))
        if warped_path is None or not warped_path.exists():
            print(f"[ERROR] No warped image found for sheet_id={sheet_id} in {warped_dir}")
            continue

        image = cv2.imread(str(warped_path))
        if image is None:
            print(f"[ERROR] Could not load warped image: {warped_path}")
            continue

        for row in group.itertuples(index=False):
            pred_value = getattr(row, pred_col, None)

            if pred_value is None or (isinstance(pred_value, float) and pd.isna(pred_value)):
                color = (0, 255, 255)  # yellow: missing prediction
            else:
                pred_int = int(pred_value)
                color = (0, 255, 0) if pred_int == 1 else (0, 0, 255)

            x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, args.thickness)

            if args.show_labels and ("row" in merged.columns) and ("col" in merged.columns):
                row_idx = getattr(row, "row", None)
                col_idx = getattr(row, "col", None)

                if row_idx is not None and col_idx is not None:
                    if not (pd.isna(row_idx) or pd.isna(col_idx)):
                        label = f"{int(row_idx)},{int(col_idx)}"
                        cv2.putText(
                            image,
                            label,
                            (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

        out_name = out_dir / f"{Path(warped_path).stem}__pred_overlay.jpg"
        cv2.imwrite(str(out_name), image)
        print(f"[OK] Overlay saved: {out_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())