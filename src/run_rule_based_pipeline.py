#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_rule_based_pipeline.py

Run the full rule-based OMR pipeline for a batch of input images.

Pipeline steps:
- detect the largest suitable quadrilateral iteratively
- warp the image step by step
- perform a final inner warp to a normalized grid area
- extract all checkbox ROIs
- classify them with a rule-based classifier
- save overlays, ROIs, and CSV results
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from classify_rule_based import classify_roi_rulebased


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------
def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # top-left
    rect[2] = points[np.argmax(s)]  # bottom-right

    d = np.diff(points, axis=1)
    rect[1] = points[np.argmin(d)]  # top-right
    rect[3] = points[np.argmax(d)]  # bottom-left

    return rect


def warp_to_size(image_bgr: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    rect = order_points(quad)
    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image_bgr, matrix, (out_w, out_h))


def precompute_edges(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, 40, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges


def find_largest_quad_with_height_band(
    image_bgr: np.ndarray,
    orig_h: int,
    min_h_frac: float,
    max_h_frac: float,
    min_area_frac: float = 0.02,
    max_area_frac: float = 0.98,
    min_aspect: float = 1.02,
    approx_eps_frac: float = 0.02,
) -> np.ndarray | None:
    edges = precompute_edges(image_bgr)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = edges.shape
    image_area = float(height * width)
    min_area = image_area * float(min_area_frac)
    max_area = image_area * float(max_area_frac)

    min_h_px = int(orig_h * float(min_h_frac))
    max_h_px = int(orig_h * float(max_h_frac))

    best_quad = None
    best_area = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_eps_frac * perimeter, True)

        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        _, _, w_box, h_box = cv2.boundingRect(approx)
        if w_box <= 0 or h_box <= 0:
            continue

        if h_box < min_h_px or h_box > max_h_px:
            continue

        aspect = h_box / float(w_box)
        if aspect < min_aspect:
            continue

        if area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2).astype(np.float32)

    return best_quad


def save_debug_polygon(
    image_bgr: np.ndarray,
    quad: np.ndarray,
    out_path: Path,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 3,
) -> None:
    debug_image = image_bgr.copy()
    ordered = order_points(quad).astype(int)
    cv2.polylines(debug_image, [ordered], True, color, thickness)
    cv2.imwrite(str(out_path), debug_image)


# ------------------------------------------------------------
# Grid extraction and classification
# ------------------------------------------------------------
def overlay_extract_and_export_csv(
    inner_warp: np.ndarray,
    out_rois: Path,
    overlay_path: Path,
    csv_path: Path,
    cfg: dict,
) -> None:
    ensure_dir(out_rois)
    ensure_dir(csv_path.parent)

    grid_cfg = cfg["grid"]
    rows = int(grid_cfg["rows"])
    cols = int(grid_cfg["cols"])
    x0 = int(grid_cfg["x0"])
    y0 = int(grid_cfg["y0"])
    dx = int(grid_cfg["dx"])
    dy = int(grid_cfg["dy"])
    box_width = int(grid_cfg["box_width"])
    box_height = int(grid_cfg["box_height"])
    pad_px = int(grid_cfg["pad_px"])

    debug_cfg = cfg.get("debug", {})
    show_score = bool(debug_cfg.get("show_score_on_overlay", False))
    overlay_thickness = int(debug_cfg.get("overlay_thickness", 2))

    rule_cfg = cfg.get("classification_rulebased", {})
    save_debug_masks = bool(rule_cfg.get("save_debug_masks", False))

    visual = inner_warp.copy()
    image_h, image_w = inner_warp.shape[:2]

    green = (0, 255, 0)
    red = (0, 0, 255)

    rows_out = []

    for row in range(rows):
        for col in range(cols):
            x = int(x0 + col * dx)
            y = int(y0 + row * dy)

            x1 = max(0, x - pad_px)
            y1 = max(0, y - pad_px)
            x2 = min(image_w, x + box_width + pad_px)
            y2 = min(image_h, y + box_height + pad_px)

            roi = inner_warp[y1:y2, x1:x2]

            roi_filename = f"checkbox_r{row:02d}_c{col:02d}.png"
            roi_path = out_rois / roi_filename
            cv2.imwrite(str(roi_path), roi)

            debug_mask_path = None
            if save_debug_masks:
                debug_mask_path = str(out_rois / roi_filename.replace(".png", "_darkmask.png"))

            pred, score = classify_roi_rulebased(roi, cfg, debug_out_path=debug_mask_path)

            color = green if pred == 1 else red
            cv2.rectangle(visual, (x1, y1), (x2, y2), color, overlay_thickness)

            if show_score:
                cv2.putText(
                    visual,
                    f"{score:.2f}",
                    (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            rows_out.append(
                [row, col, str(roi_path).replace("\\", "/"), f"{score:.6f}", pred]
            )

    cv2.imwrite(str(overlay_path), visual)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "roi_path", "score_dark_ratio", "pred_marked"])
        writer.writerows(rows_out)


def process_one_image(
    input_path: Path,
    out_root: Path,
    overlay_collect_dir: Path,
    csv_dir: Path,
    cfg: dict,
) -> tuple[Path, Path]:
    ensure_dir(out_root)

    debug_dir = out_root / "debug"
    rois_dir = out_root / "rois"
    ensure_dir(debug_dir)
    ensure_dir(rois_dir)

    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")

    orig_h, orig_w = image.shape[:2]

    quad_cfg = cfg["quad_detection"]
    min_h_frac = float(quad_cfg["min_height_frac"])
    max_h_frac = float(quad_cfg["max_height_frac"])
    iterations = int(quad_cfg.get("iterations", 5))

    debug_cfg = cfg.get("debug", {})
    save_steps = bool(debug_cfg.get("save_steps", True))

    current = image.copy()

    for iteration in range(1, iterations + 1):
        quad = find_largest_quad_with_height_band(
            current,
            orig_h=orig_h,
            min_h_frac=min_h_frac,
            max_h_frac=max_h_frac,
        )

        if quad is None:
            print(f"[WARN] No quadrilateral found in iteration {iteration}; stopping iteration loop.")
            break

        if save_steps:
            save_debug_polygon(current, quad, debug_dir / f"step{iteration:02d}_quad.jpg")

        current = warp_to_size(current, quad, orig_w, orig_h)

        if save_steps:
            cv2.imwrite(str(debug_dir / f"step{iteration:02d}_warp.jpg"), current)

    grid_cfg = cfg["grid"]
    inner_width = int(grid_cfg["inner_width"])
    inner_height = int(grid_cfg["inner_height"])

    final_quad = find_largest_quad_with_height_band(
        current,
        orig_h=orig_h,
        min_h_frac=min_h_frac,
        max_h_frac=max_h_frac,
    )

    if final_quad is not None:
        if save_steps:
            save_debug_polygon(current, final_quad, debug_dir / "step_final_quad.jpg")
        inner_warp = warp_to_size(current, final_quad, inner_width, inner_height)
    else:
        print("[WARN] Final quadrilateral not found; resizing image directly to inner target size.")
        inner_warp = cv2.resize(current, (inner_width, inner_height), interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(debug_dir / "step_inner_warp.jpg"), inner_warp)

    overlay_path = debug_dir / "step_grid_overlay.jpg"

    ensure_dir(csv_dir)
    csv_path = csv_dir / f"results_{out_root.name}.csv"

    overlay_extract_and_export_csv(
        inner_warp=inner_warp,
        out_rois=rois_dir,
        overlay_path=overlay_path,
        csv_path=csv_path,
        cfg=cfg,
    )

    ensure_dir(overlay_collect_dir)
    overlay_dst = overlay_collect_dir / f"{out_root.name}_step_grid_overlay.jpg"

    try:
        shutil.copy2(str(overlay_path), str(overlay_dst))
    except Exception as exc:
        print(f"[WARN] Could not copy overlay to collection directory: {exc}")

    return overlay_path, csv_path


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rule-based OMR pipeline on a batch of images."
    )

    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--out_batch_dir", required=True, type=str)
    parser.add_argument("--overlay_collect_dir", required=True, type=str)
    parser.add_argument("--csv_dir", required=True, type=str)

    parser.add_argument("--min_height_frac", required=True, type=float)
    parser.add_argument("--max_height_frac", required=True, type=float)
    parser.add_argument("--iterations", required=True, type=int)

    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--cols", required=True, type=int)
    parser.add_argument("--inner_width", required=True, type=int)
    parser.add_argument("--inner_height", required=True, type=int)
    parser.add_argument("--x0", required=True, type=int)
    parser.add_argument("--y0", required=True, type=int)
    parser.add_argument("--dx", required=True, type=int)
    parser.add_argument("--dy", required=True, type=int)
    parser.add_argument("--box_width", required=True, type=int)
    parser.add_argument("--box_height", required=True, type=int)
    parser.add_argument("--pad_px", required=True, type=int)

    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--ignore_border_ratio", required=True, type=float)
    parser.add_argument("--save_debug_masks", action="store_true")

    parser.add_argument("--save_steps", action="store_true")
    parser.add_argument("--show_score_on_overlay", action="store_true")
    parser.add_argument("--overlay_thickness", type=int, default=2)

    args = parser.parse_args()

    cfg = {
        "quad_detection": {
            "min_height_frac": args.min_height_frac,
            "max_height_frac": args.max_height_frac,
            "iterations": args.iterations,
        },
        "grid": {
            "rows": args.rows,
            "cols": args.cols,
            "inner_width": args.inner_width,
            "inner_height": args.inner_height,
            "x0": args.x0,
            "y0": args.y0,
            "dx": args.dx,
            "dy": args.dy,
            "box_width": args.box_width,
            "box_height": args.box_height,
            "pad_px": args.pad_px,
        },
        "classification_rulebased": {
            "threshold": args.threshold,
            "ignore_border_ratio": args.ignore_border_ratio,
            "save_debug_masks": bool(args.save_debug_masks),
        },
        "debug": {
            "save_steps": bool(args.save_steps),
            "show_score_on_overlay": bool(args.show_score_on_overlay),
            "overlay_thickness": int(args.overlay_thickness),
        },
    }

    input_dir = Path(args.input_dir)
    out_batch_dir = Path(args.out_batch_dir)
    overlay_collect_dir = Path(args.overlay_collect_dir)
    csv_dir = Path(args.csv_dir)

    ensure_dir(out_batch_dir)
    ensure_dir(overlay_collect_dir)
    ensure_dir(csv_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = sorted(
        [path for path in input_dir.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    print(f"Found images: {len(images)}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {out_batch_dir}")

    for index, image_path in enumerate(images, start=1):
        stem = image_path.stem
        out_root = out_batch_dir / stem

        print(f"\n[{index}/{len(images)}] Processing: {image_path.name}")

        try:
            overlay_path, csv_path = process_one_image(
                input_path=image_path,
                out_root=out_root,
                overlay_collect_dir=overlay_collect_dir,
                csv_dir=csv_dir,
                cfg=cfg,
            )
            print("  -> OK")
            print(f"     Overlay: {overlay_path}")
            print(f"     CSV:     {csv_path}")
        except Exception as exc:
            print(f"  -> ERROR: {exc}")


if __name__ == "__main__":
    main()