#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
review_labels.py

Small manual review tool for ROI labels.

The tool loads ROIs listed in an input CSV file and allows a user to assign
binary labels interactively:

- 0 = empty
- 1 = marked
- s = skip
- b = undo last label in the current session
- q = quit

The current ROI is shown in large view, while the next unlabeled ROI is shown
as a preview. Labels are continuously written to an output CSV file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


WINDOW_HEIGHT = 900
LEFT_PANEL_WIDTH = 900
RIGHT_PANEL_WIDTH = 360
PANEL_GAP = 12
ROI_DISPLAY_MAX_EDGE = 700
PREVIEW_DISPLAY_MAX_EDGE = 260
BACKGROUND_COLOR = (0, 0, 0)
WINDOW_NAME = "Label Review Tool"


def scale_to_max(image_bgr: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = max_edge / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def place_center(
    canvas: np.ndarray,
    image_bgr: np.ndarray,
    *,
    x0: int,
    y0: int,
    area_width: int,
    area_height: int,
) -> None:
    image_height, image_width = image_bgr.shape[:2]
    x = x0 + (area_width - image_width) // 2
    y = y0 + (area_height - image_height) // 2
    canvas[y : y + image_height, x : x + image_width] = image_bgr


def build_view(
    current_roi_bgr: np.ndarray,
    next_roi_bgr: np.ndarray | None,
    title_text: str,
) -> np.ndarray:
    canvas_width = LEFT_PANEL_WIDTH + PANEL_GAP + RIGHT_PANEL_WIDTH
    canvas = np.full((WINDOW_HEIGHT, canvas_width, 3), BACKGROUND_COLOR, dtype=np.uint8)

    current_large = scale_to_max(current_roi_bgr, ROI_DISPLAY_MAX_EDGE)
    place_center(
        canvas,
        current_large,
        x0=0,
        y0=0,
        area_width=LEFT_PANEL_WIDTH,
        area_height=WINDOW_HEIGHT,
    )

    divider_x = LEFT_PANEL_WIDTH + PANEL_GAP // 2
    cv2.line(canvas, (divider_x, 0), (divider_x, WINDOW_HEIGHT), (60, 60, 60), 1)

    right_x0 = LEFT_PANEL_WIDTH + PANEL_GAP

    if next_roi_bgr is not None:
        preview = scale_to_max(next_roi_bgr, PREVIEW_DISPLAY_MAX_EDGE)
        place_center(
            canvas,
            preview,
            x0=right_x0,
            y0=80,
            area_width=RIGHT_PANEL_WIDTH,
            area_height=RIGHT_PANEL_WIDTH,
        )
        cv2.putText(
            canvas,
            "Next",
            (right_x0 + 20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            canvas,
            "Next: -",
            (right_x0 + 20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        canvas,
        title_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
    )

    help_lines = [
        "0 = empty",
        "1 = marked",
        "s = skip",
        "b = undo",
        "q = quit",
    ]
    y = WINDOW_HEIGHT - 160
    for line in help_lines:
        cv2.putText(
            canvas,
            line,
            (right_x0 + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )
        y += 30

    return canvas


def find_next_unlabeled(
    rows: list[dict],
    start_idx: int,
    done_set: set[str],
) -> tuple[str | None, np.ndarray | None]:
    index = start_idx + 1
    while index < len(rows):
        roi_path = str(rows[index]["roi_path"])
        if roi_path not in done_set:
            image = cv2.imread(roi_path)
            if image is not None:
                return roi_path, image
        index += 1
    return None, None


def load_existing_labels(output_csv: Path) -> tuple[pd.DataFrame, set[str]]:
    if output_csv.exists():
        labeled_df = pd.read_csv(output_csv)
        done = set(labeled_df["roi_path"].astype(str).tolist())
        return labeled_df, done

    labeled_df = pd.DataFrame(columns=["roi_path", "sheet_id", "label"])
    return labeled_df, set()


def infer_sheet_id(row: dict, roi_path: str) -> str:
    if "sheet_id" in row and pd.notna(row["sheet_id"]):
        return str(row["sheet_id"])
    return Path(roi_path).stem.split("__")[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive tool for manually reviewing ROI labels."
    )
    parser.add_argument("--input_csv", required=True, type=str, help="CSV containing ROI paths")
    parser.add_argument("--output_csv", required=True, type=str, help="CSV file to store labels")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(input_csv)
    if dataframe.empty:
        raise RuntimeError("Input CSV is empty.")

    if "roi_path" not in dataframe.columns:
        raise KeyError("Input CSV must contain the column 'roi_path'.")

    labeled_df, done = load_existing_labels(output_csv)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, LEFT_PANEL_WIDTH + PANEL_GAP + RIGHT_PANEL_WIDTH, WINDOW_HEIGHT)

    rows = dataframe.to_dict("records")
    session_stack: list[tuple[str, str]] = []

    index = 0
    while index < len(rows):
        row = rows[index]
        roi_path = str(row["roi_path"])
        sheet_id = infer_sheet_id(row, roi_path)

        if roi_path in done:
            index += 1
            continue

        current_image = cv2.imread(roi_path)
        if current_image is None:
            print(f"[WARN] ROI could not be loaded: {roi_path}")
            index += 1
            continue

        _, next_image = find_next_unlabeled(rows, index, done)

        title = f"{Path(roi_path).name} | 0=empty 1=marked s=skip b=undo q=quit"
        view = build_view(current_image, next_image, title)

        cv2.imshow(WINDOW_NAME, view)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            index += 1
            continue

        if key == ord("b"):
            if session_stack:
                last_roi, _last_sheet = session_stack.pop()
                labeled_df = labeled_df[labeled_df["roi_path"] != last_roi].copy()
                done.discard(last_roi)
                labeled_df.to_csv(output_csv, index=False, encoding="utf-8")
                print(f"[INFO] Undo: {Path(last_roi).name}")
                index = max(0, index - 1)
            else:
                print("[INFO] Nothing to undo.")
            continue

        if key == ord("0"):
            label = 0
        elif key == ord("1"):
            label = 1
        else:
            continue

        labeled_df.loc[len(labeled_df)] = [roi_path, sheet_id, label]
        done.add(roi_path)
        session_stack.append((roi_path, sheet_id))

        labeled_df.to_csv(output_csv, index=False, encoding="utf-8")
        index += 1

    cv2.destroyAllWindows()
    print(f"[OK] Labels saved: {output_csv} ({len(labeled_df)} entries)")


if __name__ == "__main__":
    main()