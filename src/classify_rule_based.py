#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
classify_rule_based.py

Rule-based checkbox classification using a dark-pixel ratio computed after
Otsu thresholding, with optional border exclusion.

Interface:
- classify_roi_rulebased(roi_bgr, cfg) -> (pred: int, score: float)

Output:
- pred = 1 for marked
- pred = 0 for empty
- score = dark-pixel ratio in the ROI
"""

from __future__ import annotations

import cv2
import numpy as np


def dark_ratio(
    image_bgr_or_gray: np.ndarray,
    ignore_border_ratio: float = 0.0,
    debug_out_path: str | None = None,
) -> float:
    """
    Compute the fraction of dark pixels (0..1) in an ROI.

    The image is converted to grayscale if needed, lightly blurred, and then
    binarized using Otsu thresholding. Dark pixels correspond to zeros in the
    binary image.

    Optionally, a border region can be excluded to reduce the influence of
    printed checkbox borders.
    """
    if image_bgr_or_gray is None:
        return 0.0

    if len(image_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray.copy()

    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresholded = cv2.threshold(
        gray_blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    dark_mask = (thresholded == 0).astype(np.uint8)

    height, width = dark_mask.shape[:2]

    if ignore_border_ratio > 0.0:
        border_x = int(width * ignore_border_ratio)
        border_y = int(height * ignore_border_ratio)

        x1, y1 = border_x, border_y
        x2, y2 = width - border_x, height - border_y

        if x2 <= x1 + 1 or y2 <= y1 + 1:
            cropped = dark_mask
        else:
            cropped = dark_mask[y1:y2, x1:x2]
    else:
        cropped = dark_mask

    total_pixels = cropped.size
    if total_pixels == 0:
        return 0.0

    ratio = float(np.sum(cropped)) / float(total_pixels)

    if debug_out_path is not None:
        debug_image = (cropped * 255).astype(np.uint8)
        cv2.imwrite(debug_out_path, debug_image)

    return ratio


def classify_roi_rulebased(
    roi_bgr: np.ndarray,
    cfg: dict,
    debug_out_path: str | None = None,
) -> tuple[int, float]:
    """
    Classify one ROI using the rule-based dark-ratio method.

    Expected configuration keys:
      cfg["classification_rulebased"]["threshold"]
      cfg["classification_rulebased"]["ignore_border_ratio"]
    """
    config = cfg.get("classification_rulebased", {})
    threshold = float(config.get("threshold", 0.20))
    ignore_border_ratio = float(config.get("ignore_border_ratio", 0.15))

    score = dark_ratio(
        roi_bgr,
        ignore_border_ratio=ignore_border_ratio,
        debug_out_path=debug_out_path,
    )
    pred = 1 if score >= threshold else 0
    return pred, score