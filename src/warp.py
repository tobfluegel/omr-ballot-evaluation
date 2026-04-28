#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
warp.py

Detect corner markers on ballot images using YOLO and generate normalized
top-down warped images.

This implementation corresponds to the marker-based warping stage
used in the thesis ML pipeline.

Key characteristics:
- marker positions are derived from detection centers
- if exactly one marker is missing, it is estimated from the other three
- perspective transformation is applied using four corner points
- warped images, overlays, and metadata are written to disk
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------
@dataclass
class Det:
    cls_name: str
    conf: float
    xyxy: np.ndarray
    center: tuple[float, float]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def pick_best_per_class(dets: list[Det], wanted: list[str]) -> dict[str, Det]:
    """Select the highest-confidence detection for each requested class."""
    best: dict[str, Det] = {}
    for det in dets:
        if det.cls_name not in wanted:
            continue
        if det.cls_name not in best or det.conf > best[det.cls_name].conf:
            best[det.cls_name] = det
    return best


def order_points_from_markers(best: dict[str, Det]) -> np.ndarray:
    """Return points in the order [TL, TR, BL, BR] using q, w, a, s marker centers."""
    tl = best["q"].center
    tr = best["w"].center
    bl = best["a"].center
    br = best["s"].center
    return np.array([tl, tr, bl, br], dtype=np.float32)


def clamp_point(point: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp a point to valid image coordinates."""
    x = float(np.clip(point[0], 0, width - 1))
    y = float(np.clip(point[1], 0, height - 1))
    return np.array([x, y], dtype=np.float32)


def estimate_missing_marker(
    best: dict[str, Det],
    wanted: list[str],
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray | None, str | None]:
    """
    If exactly one of q, w, a, s is missing, estimate its center from the other three
    using the parallelogram rule: TL + BR = TR + BL

    Returns:
        (src_pts [TL, TR, BL, BR], missing_key) or (None, None)
    """
    missing = [key for key in wanted if key not in best]
    present = [key for key in wanted if key in best]

    if len(missing) != 1 or len(present) != 3:
        return None, None

    miss = missing[0]

    tl = np.array(best["q"].center, dtype=np.float32) if "q" in best else None
    tr = np.array(best["w"].center, dtype=np.float32) if "w" in best else None
    bl = np.array(best["a"].center, dtype=np.float32) if "a" in best else None
    br = np.array(best["s"].center, dtype=np.float32) if "s" in best else None

    if miss == "q":
        if tr is None or bl is None or br is None:
            return None, None
        tl = tr + bl - br
    elif miss == "w":
        if tl is None or br is None or bl is None:
            return None, None
        tr = tl + br - bl
    elif miss == "a":
        if tl is None or br is None or tr is None:
            return None, None
        bl = tl + br - tr
    elif miss == "s":
        if tr is None or bl is None or tl is None:
            return None, None
        br = tr + bl - tl
    else:
        return None, None

    if miss == "q":
        tl = clamp_point(tl, img_w, img_h)
    elif miss == "w":
        tr = clamp_point(tr, img_w, img_h)
    elif miss == "a":
        bl = clamp_point(bl, img_w, img_h)
    elif miss == "s":
        br = clamp_point(br, img_w, img_h)

    src_pts = np.array([tl, tr, bl, br], dtype=np.float32)
    return src_pts, miss


def compute_target_size(src_pts: np.ndarray, out_w: int | None, out_h: int | None) -> tuple[int, int]:
    """
    Estimate target size from marker geometry if width and/or height are not provided.
    src_pts: [TL, TR, BL, BR]
    """
    tl, tr, bl, br = src_pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    est_w = int(round(max(width_top, width_bottom)))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    est_h = int(round(max(height_left, height_right)))

    if out_w is None and out_h is None:
        return max(est_w, 200), max(est_h, 200)

    if out_w is None:
        ratio = est_w / max(est_h, 1)
        out_w = int(round(out_h * ratio))

    if out_h is None:
        ratio = est_h / max(est_w, 1)
        out_h = int(round(out_w * ratio))

    return int(out_w), int(out_h)


def warp_perspective(img_bgr: np.ndarray, src_pts: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Warp the image using source points [TL, TR, BL, BR]."""
    dst_pts = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [0, out_h - 1],
            [out_w - 1, out_h - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_bgr, homography, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return warped


def draw_debug_overlay(
    img_bgr: np.ndarray,
    dets: list[Det],
    chosen: dict[str, Det],
    estimated_key: str | None,
    estimated_pt: tuple[float, float] | None,
) -> np.ndarray:
    """
    Draw all detections (thin), selected markers (thick), center points,
    and the estimated point if one marker was reconstructed.
    """
    out = img_bgr.copy()

    for det in dets:
        x1, y1, x2, y2 = det.xyxy.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(
            out,
            f"{det.cls_name} {det.conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    color_map = {
        "q": (255, 0, 0),
        "w": (255, 255, 0),
        "a": (0, 0, 255),
        "s": (0, 255, 0),
    }

    for key, det in chosen.items():
        x1, y1, x2, y2 = det.xyxy.astype(int)
        color = color_map.get(key, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        cx, cy = int(round(det.center[0])), int(round(det.center[1]))
        cv2.circle(out, (cx, cy), 5, color, -1)

        cv2.putText(
            out,
            f"{key} {det.conf:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    if estimated_key is not None and estimated_pt is not None:
        ex, ey = int(round(estimated_pt[0])), int(round(estimated_pt[1]))
        color = color_map.get(estimated_key, (0, 255, 255))
        cv2.circle(out, (ex, ey), 10, color, 3)
        cv2.putText(
            out,
            f"EST {estimated_key}",
            (ex + 8, ey - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def run_yolo(model: YOLO, img_bgr: np.ndarray, imgsz: int, conf: float) -> tuple[list[Det], dict[int, str]]:
    """Run YOLO inference and return detections plus the model name mapping."""
    result = model.predict(source=img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
    names = result.names

    dets: list[Det] = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets, names

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)

    for bb, cf, ci in zip(xyxy, confs, clss):
        cls_name = names.get(ci, str(ci))
        x1, y1, x2, y2 = bb.astype(float)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dets.append(Det(cls_name=cls_name, conf=float(cf), xyxy=bb, center=(cx, cy)))

    return dets, names


def iter_images(input_dir: Path, exts: list[str], recursive: bool) -> Iterable[Path]:
    patterns = []
    for ext in exts:
        ext = ext.lower().lstrip(".")
        patterns.append(f"*.{ext}")

    if recursive:
        for pattern in patterns:
            yield from input_dir.rglob(pattern)
    else:
        for pattern in patterns:
            yield from input_dir.glob(pattern)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect ballot markers with YOLO and warp images into a normalized top-down view."
    )
    parser.add_argument("--model", required=True, type=str, help="Path to YOLO weights")
    parser.add_argument("--input_dir", required=True, type=str, help="Folder with raw input images")
    parser.add_argument("--out_dir", default="out_warp", type=str, help="Output directory")
    parser.add_argument("--imgsz", default=1280, type=int, help="YOLO inference size")
    parser.add_argument("--conf", default=0.25, type=float, help="YOLO confidence threshold")
    parser.add_argument("--out_w", default=1000, type=int, help="Output warped width")
    parser.add_argument("--out_h", default=1400, type=int, help="Output warped height")
    parser.add_argument("--ext", default="jpg,jpeg,png", type=str, help="Comma-separated file extensions")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively")
    parser.add_argument("--limit", default=0, type=int, help="Process only the first N images (0 = all)")

    args = parser.parse_args()

    model_path = Path(args.model)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        return 2
    if not input_dir.exists():
        print(f"ERROR: input_dir not found: {input_dir}")
        return 2

    exts = [part.strip() for part in args.ext.split(",") if part.strip()]
    images = sorted(iter_images(input_dir, exts=exts, recursive=args.recursive))
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    print(f"Model:     {model_path}")
    print(f"Input dir: {input_dir}")
    print(f"Out dir:   {out_dir}")
    print(f"Images:    {len(images)}")
    print(f"imgsz={args.imgsz} conf={args.conf} out={args.out_w}x{args.out_h}\n")

    model = YOLO(str(model_path))
    wanted = ["q", "w", "a", "s"]

    ok_count = 0
    miss_count = 0
    est_count = 0
    err_count = 0

    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {image_path.name}")

        try:
            img = load_image_bgr(image_path)
            h, w = img.shape[:2]

            dets, _ = run_yolo(model, img, imgsz=args.imgsz, conf=args.conf)
            best = pick_best_per_class(dets, wanted=wanted)

            estimated_key = None
            estimated_pt = None
            missing = [key for key in wanted if key not in best]

            if missing:
                src_pts, estimated_key = estimate_missing_marker(best, wanted=wanted, img_w=w, img_h=h)
                if src_pts is None:
                    overlay = draw_debug_overlay(img, dets, best, None, None)
                    overlay_path = out_dir / f"{image_path.stem}_overlay.jpg"
                    cv2.imwrite(str(overlay_path), overlay)

                    miss_count += 1
                    print(f"  [WARN] Missing markers: {missing} -> overlay saved (no estimate)")
                    continue
                else:
                    if estimated_key == "q":
                        estimated_pt = tuple(src_pts[0].tolist())
                    elif estimated_key == "w":
                        estimated_pt = tuple(src_pts[1].tolist())
                    elif estimated_key == "a":
                        estimated_pt = tuple(src_pts[2].tolist())
                    elif estimated_key == "s":
                        estimated_pt = tuple(src_pts[3].tolist())
                    est_count += 1
                    print(f"  [INFO] Missing marker {estimated_key} -> estimated from 3 markers")
            else:
                src_pts = order_points_from_markers(best)

            overlay = draw_debug_overlay(img, dets, best, estimated_key, estimated_pt)
            overlay_path = out_dir / f"{image_path.stem}_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay)

            out_w, out_h = compute_target_size(src_pts, args.out_w, args.out_h)
            warped = warp_perspective(img, src_pts, out_w=out_w, out_h=out_h)

            warped_path = out_dir / f"{image_path.stem}_warped.jpg"
            meta_path = out_dir / f"{image_path.stem}_meta.txt"

            cv2.imwrite(str(warped_path), warped)

            with meta_path.open("w", encoding="utf-8") as f:
                f.write(f"image: {image_path}\n")
                f.write(f"model: {model_path}\n")
                f.write(f"imgsz: {args.imgsz}\n")
                f.write(f"conf_thr: {args.conf}\n")
                f.write(f"out_size: {out_w}x{out_h}\n")
                f.write(f"estimated_missing_marker: {estimated_key if estimated_key else 'none'}\n")
                f.write("chosen_markers (best per class):\n")
                for key in ["q", "w", "a", "s"]:
                    if key in best:
                        det = best[key]
                        f.write(
                            f"  {key}: conf={det.conf:.4f} center=({det.center[0]:.2f},{det.center[1]:.2f}) "
                            f"xyxy={det.xyxy.tolist()}\n"
                        )
                    else:
                        f.write(f"  {key}: MISSING\n")

                f.write("src_pts [TL,TR,BL,BR]:\n")
                for i, point in enumerate(src_pts.tolist()):
                    f.write(f"  {i}: ({point[0]:.2f},{point[1]:.2f})\n")

            ok_count += 1
            if estimated_key:
                print(f"  [OK] Warped image and metadata saved (estimated {estimated_key})")
            else:
                print("  [OK] Warped image and metadata saved")

        except Exception as exc:
            err_count += 1
            print(f"  [ERROR] {exc}")

    print("\nDONE.")
    print(f"OK: {ok_count} | Estimated: {est_count} | Missing markers: {miss_count} | Errors: {err_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())