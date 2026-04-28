#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_rois.py

Extract checkbox ROIs from warped ballot images using a YOLO detector.

Features:
- configurable confidence and NMS IoU thresholds
- size filtering for bounding boxes
- optional post-NMS deduplication
- optional grid-based row/column assignment
- fallback naming when a full grid assignment is not possible

Outputs:
- cropped ROI images
- metadata CSV file
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def list_warped_images(folder: Path) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images: List[Path] = []
    for path in folder.iterdir():
        if path.suffix.lower() not in extensions:
            continue
        if "_warped" not in path.name.lower():
            continue
        images.append(path)
    return sorted(images)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def assign_grid_by_sort(
    centers_xy: List[Tuple[float, float]],
    rows: int,
    cols: int,
) -> Optional[List[Tuple[int, int]]]:
    """
    Assign row/column indices by sorting ROI centers first by y and then by x.

    This assumes a regular grid layout and requires exactly rows * cols detections.
    Returns None if the number of detections does not match the expected grid size.
    """
    n = len(centers_xy)
    expected = rows * cols
    if n != expected:
        return None

    indices = list(range(n))
    indices.sort(key=lambda i: (centers_xy[i][1], centers_xy[i][0]))

    assignment: List[Optional[Tuple[int, int]]] = [None] * n

    for row in range(rows):
        block = indices[row * cols : (row + 1) * cols]
        block.sort(key=lambda i: centers_xy[i][0])
        for col, i in enumerate(block):
            assignment[i] = (row, col)

    return assignment  # type: ignore[return-value]


def dedup_by_center_distance(filtered: List[Dict], min_dist_px: float) -> List[Dict]:
    """
    Deduplicate by center distance:
    - sort by confidence descending
    - keep one box
    - discard other boxes whose center is closer than min_dist_px
    """
    if min_dist_px <= 0:
        return filtered

    filtered_sorted = sorted(filtered, key=lambda d: d["conf"], reverse=True)
    kept: List[Dict] = []

    min_dist_sq = float(min_dist_px) ** 2
    for candidate in filtered_sorted:
        keep = True
        for existing in kept:
            dx = candidate["cx"] - existing["cx"]
            dy = candidate["cy"] - existing["cy"]
            if (dx * dx + dy * dy) < min_dist_sq:
                keep = False
                break
        if keep:
            kept.append(candidate)

    return kept


def dedup_by_grid_cells(
    filtered: List[Dict],
    *,
    img_w: int,
    img_h: int,
    rows: int,
    cols: int,
    margin: float = 0.0,
) -> List[Dict]:
    """
    Deduplicate by allowing at most one box per grid cell.

    The grid cell is determined by the bounding-box center (cx, cy).
    Optional margin (0..0.49) shrinks the cell interior used for assignment.

    For each cell, only the highest-confidence box is kept.
    Boxes outside the assignable cell interior are preserved separately.
    """
    if rows <= 0 or cols <= 0 or not filtered:
        return filtered

    margin = float(clamp(margin, 0.0, 0.49))
    cell_w = img_w / float(cols)
    cell_h = img_h / float(rows)

    best_in_cell: Dict[Tuple[int, int], Dict] = {}
    others: List[Dict] = []

    for box in filtered:
        cx = float(box["cx"])
        cy = float(box["cy"])

        col = int(cx // cell_w)
        row = int(cy // cell_h)

        if not (0 <= row < rows and 0 <= col < cols):
            others.append(box)
            continue

        x0 = col * cell_w
        x1 = (col + 1) * cell_w
        y0 = row * cell_h
        y1 = (row + 1) * cell_h

        shrink_x0 = x0 + margin * cell_w
        shrink_x1 = x1 - margin * cell_w
        shrink_y0 = y0 + margin * cell_h
        shrink_y1 = y1 - margin * cell_h

        if not (shrink_x0 <= cx <= shrink_x1 and shrink_y0 <= cy <= shrink_y1):
            others.append(box)
            continue

        key = (row, col)
        previous = best_in_cell.get(key)
        if previous is None or box["conf"] > previous["conf"]:
            best_in_cell[key] = box

    return list(best_in_cell.values()) + others


# ------------------------------------------------------------
# Core
# ------------------------------------------------------------
def extract_rois_from_warped(
    warped_dir: Path,
    model_path: Path,
    rois_dir: Path,
    meta_csv: Path,
    *,
    conf: float = 0.25,
    iou: float = 0.40,
    imgsz: int = 1280,
    min_box: int = 25,
    max_box: int = 200,
    topk_per_sheet: int = 200,
    rows: int = 26,
    cols: int = 5,
    dedup_mode: str = "none",            # none|center|grid|center+grid
    min_center_dist_px: float = 0.0,     # for center
    grid_dedup_margin: float = 0.0,      # for grid
) -> Path:
    rois_dir.mkdir(parents=True, exist_ok=True)
    meta_csv.parent.mkdir(parents=True, exist_ok=True)

    images = list_warped_images(warped_dir)
    if not images:
        raise RuntimeError(f"No *_warped images found in {warped_dir}.")

    model = YOLO(str(model_path))
    output_rows: List[Dict] = []

    for image_path in images:
        sheet_id = image_path.stem.replace("_warped", "")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Could not load image: {image_path}")
            continue

        height, width = image.shape[:2]

        result = model.predict(
            source=str(image_path),
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            print(f"[WARN] No detections for: {image_path.name}")
            continue

        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        order = confidences.argsort()[::-1][: int(topk_per_sheet)]

        filtered: List[Dict] = []
        for index in order:
            x1, y1, x2, y2 = xyxy[index]
            score = float(confidences[index])

            x1i = int(clamp(round(float(x1)), 0, width - 1))
            y1i = int(clamp(round(float(y1)), 0, height - 1))
            x2i = int(clamp(round(float(x2)), 0, width - 1))
            y2i = int(clamp(round(float(y2)), 0, height - 1))

            box_width = x2i - x1i
            box_height = y2i - y1i

            if box_width < int(min_box) or box_height < int(min_box):
                continue
            if box_width > int(max_box) or box_height > int(max_box):
                continue
            if box_width <= 1 or box_height <= 1:
                continue

            cx = (x1i + x2i) / 2.0
            cy = (y1i + y2i) / 2.0

            filtered.append(
                {
                    "idx": int(index),
                    "conf": score,
                    "x1": x1i,
                    "y1": y1i,
                    "x2": x2i,
                    "y2": y2i,
                    "bw": int(box_width),
                    "bh": int(box_height),
                    "cx": float(cx),
                    "cy": float(cy),
                }
            )

        if not filtered:
            print(f"[WARN] No boxes left after filtering: {image_path.name}")
            continue

        mode = (dedup_mode or "none").strip().lower()

        if mode in {"center", "center+grid"}:
            before = len(filtered)
            filtered = dedup_by_center_distance(filtered, float(min_center_dist_px))
            after = len(filtered)
            if after != before:
                print(f"[INFO] {image_path.name}: center-dedup {before} -> {after}")

        if mode in {"grid", "center+grid"}:
            before = len(filtered)
            filtered = dedup_by_grid_cells(
                filtered,
                img_w=width,
                img_h=height,
                rows=int(rows),
                cols=int(cols),
                margin=float(grid_dedup_margin),
            )
            after = len(filtered)
            if after != before:
                print(f"[INFO] {image_path.name}: grid-dedup {before} -> {after}")

        centers = [(entry["cx"], entry["cy"]) for entry in filtered]
        assignment = assign_grid_by_sort(centers, rows=int(rows), cols=int(cols))

        if assignment is None:
            expected = int(rows) * int(cols)
            print(
                f"[WARN] {image_path.name}: found {len(filtered)} boxes, expected {expected}. "
                f"Saving fallback naming without row/column assignment."
            )

            indices = list(range(len(filtered)))
            indices.sort(key=lambda i: (filtered[i]["cy"], filtered[i]["cx"]))

            for fallback_index, i in enumerate(indices):
                entry = filtered[i]
                roi = image[entry["y1"] : entry["y2"], entry["x1"] : entry["x2"]]
                if roi.size == 0:
                    continue

                roi_path = rois_dir / f"{sheet_id}__{fallback_index:03d}.png"
                cv2.imwrite(str(roi_path), roi)

                output_rows.append(
                    {
                        "sheet_id": sheet_id,
                        "warped_image": str(image_path).replace("\\", "/"),
                        "roi_path": str(roi_path).replace("\\", "/"),
                        "conf": float(entry["conf"]),
                        "x1": int(entry["x1"]),
                        "y1": int(entry["y1"]),
                        "x2": int(entry["x2"]),
                        "y2": int(entry["y2"]),
                        "bw": int(entry["bw"]),
                        "bh": int(entry["bh"]),
                        "cx": float(entry["cx"]),
                        "cy": float(entry["cy"]),
                        "row": None,
                        "col": None,
                    }
                )
        else:
            saved = 0
            for entry, (row, col) in zip(filtered, assignment):
                roi = image[entry["y1"] : entry["y2"], entry["x1"] : entry["x2"]]
                if roi.size == 0:
                    continue

                roi_path = rois_dir / f"{sheet_id}_r{row + 1:02d}_c{col + 1:02d}.png"
                cv2.imwrite(str(roi_path), roi)
                saved += 1

                output_rows.append(
                    {
                        "sheet_id": sheet_id,
                        "warped_image": str(image_path).replace("\\", "/"),
                        "roi_path": str(roi_path).replace("\\", "/"),
                        "conf": float(entry["conf"]),
                        "x1": int(entry["x1"]),
                        "y1": int(entry["y1"]),
                        "x2": int(entry["x2"]),
                        "y2": int(entry["y2"]),
                        "bw": int(entry["bw"]),
                        "bh": int(entry["bh"]),
                        "cx": float(entry["cx"]),
                        "cy": float(entry["cy"]),
                        "row": int(row + 1),
                        "col": int(col + 1),
                    }
                )

            print(f"[OK] {image_path.name}: saved {saved} ROIs with row/column naming")

    df = pd.DataFrame(output_rows)
    df.to_csv(meta_csv, index=False, encoding="utf-8")

    print(f"\n[INFO] Metadata saved to: {meta_csv}")
    print(f"[INFO] ROI directory: {rois_dir}")
    return meta_csv


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract checkbox ROIs from warped images using YOLO and optional deduplication."
    )
    parser.add_argument("--warped_dir", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--rois_dir", required=True, type=str)
    parser.add_argument("--meta_csv", required=True, type=str)

    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.40, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")

    parser.add_argument("--min_box", type=int, default=25, help="Minimum bounding-box size in pixels")
    parser.add_argument("--max_box", type=int, default=200, help="Maximum bounding-box size in pixels")
    parser.add_argument("--topk_per_sheet", type=int, default=200, help="Keep only the top-K detections per sheet")

    parser.add_argument("--rows", type=int, default=26, help="Expected number of grid rows")
    parser.add_argument("--cols", type=int, default=5, help="Expected number of grid columns")

    parser.add_argument(
        "--dedup_mode",
        type=str,
        default="center+grid",
        choices=["none", "center", "grid", "center+grid"],
        help="Additional deduplication after YOLO NMS",
    )
    parser.add_argument(
        "--min_center_dist_px",
        type=float,
        default=18.0,
        help="Minimum center distance for center-based deduplication",
    )
    parser.add_argument(
        "--grid_dedup_margin",
        type=float,
        default=0.10,
        help="Margin for grid-based deduplication (0.0 to 0.49)",
    )

    args = parser.parse_args()

    extract_rois_from_warped(
        warped_dir=Path(args.warped_dir),
        model_path=Path(args.model),
        rois_dir=Path(args.rois_dir),
        meta_csv=Path(args.meta_csv),
        conf=float(args.conf),
        iou=float(args.iou),
        imgsz=int(args.imgsz),
        min_box=int(args.min_box),
        max_box=int(args.max_box),
        topk_per_sheet=int(args.topk_per_sheet),
        rows=int(args.rows),
        cols=int(args.cols),
        dedup_mode=str(args.dedup_mode),
        min_center_dist_px=float(args.min_center_dist_px),
        grid_dedup_margin=float(args.grid_dedup_margin),
    )


if __name__ == "__main__":
    main()