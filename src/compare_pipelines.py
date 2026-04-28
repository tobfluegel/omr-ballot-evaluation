#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_pipelines.py

Run and compare the ML-based and rule-based OMR pipelines based on a single
configuration file.

Behavior:
- loads the main YAML configuration
- resolves input/output paths
- starts the enabled pipelines
- stores logs for both runs
- supports sequential or parallel execution
- skips the ML pipeline gracefully if required model files are missing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in configuration file: {path}")

    return data


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and compare the ML-based and rule-based OMR pipelines."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        type=str,
        help="Path to the main configuration YAML file.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        type=str,
        help="Optional run name. If omitted, a timestamp is used.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Run pipelines sequentially instead of in parallel.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    cfg = read_yaml(config_path)

    config_root = config_path.parent.parent

    project_root_cfg = Path(cfg.get("project", {}).get("root_dir", "."))
    if project_root_cfg.is_absolute():
        repo_root = project_root_cfg
    else:
        repo_root = (config_root / project_root_cfg).resolve()

    input_dir = resolve_path(repo_root, cfg["io"]["input_dir"])
    output_dir = resolve_path(repo_root, cfg["io"]["output_dir"])

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name.strip() or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = output_dir / f"comparison_{run_name}"
    ml_output = run_root / "ml"
    rule_output = run_root / "rule_based"
    log_dir = run_root / "logs"

    ml_output.mkdir(parents=True, exist_ok=True)
    rule_output.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration: {config_path}")
    print(f"Repository root: {repo_root}")
    print(f"Input dir:       {input_dir}")
    print(f"Run dir:         {run_root}")

    commands: list[tuple[str, list[str], Path]] = []
    log_paths: dict[str, Path] = {}

    # ------------------------------------------------------------
    # ML pipeline
    # ------------------------------------------------------------
    if cfg.get("ml", {}).get("enabled", True):
        ml_cfg = cfg["ml"]
        ml_paths = ml_cfg["paths"]
        ml_params = ml_cfg["params"]

        ml_entry = (base_dir / "run_ml_pipeline.py").resolve()
        if not ml_entry.exists():
            raise FileNotFoundError(f"ML pipeline script not found: {ml_entry}")

        marker_model = resolve_path(repo_root, ml_paths["marker_model"])
        checkbox_model = resolve_path(repo_root, ml_paths["checkbox_model"])
        svm_model = resolve_path(repo_root, ml_paths["svm_model"])

        missing_models: list[tuple[str, Path]] = []
        for name, path in [
            ("marker_model", marker_model),
            ("checkbox_model", checkbox_model),
            ("svm_model", svm_model),
        ]:
            if not path.exists():
                missing_models.append((name, path))

        if missing_models:
            print("\n[WARN] ML pipeline skipped due to missing model files:")
            for name, path in missing_models:
                print(f"  - {name}: {path}")
            print("\nTo enable the ML pipeline, place the required files in the 'models/' directory.\n")
        else:
            cmd_ml = [
                sys.executable,
                str(ml_entry),
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(ml_output),
                "--marker_model",
                str(marker_model),
                "--checkbox_model",
                str(checkbox_model),
                "--svm_model",
                str(svm_model),
                "--warp_imgsz",
                str(ml_params["warp_imgsz"]),
                "--warp_conf",
                str(ml_params["warp_conf"]),
                "--out_w",
                str(ml_params["out_w"]),
                "--out_h",
                str(ml_params["out_h"]),
                "--roi_imgsz",
                str(ml_params["roi_imgsz"]),
                "--roi_conf",
                str(ml_params["roi_conf"]),
                "--roi_iou",
                str(ml_params.get("roi_iou", 0.35)),
                "--dedup_mode",
                str(ml_params.get("dedup_mode", "center+grid")),
                "--min_center_dist_px",
                str(ml_params.get("min_center_dist_px", 18)),
                "--grid_dedup_margin",
                str(ml_params.get("grid_dedup_margin", 0.10)),
                "--min_box",
                str(ml_params["min_box"]),
                "--max_box",
                str(ml_params["max_box"]),
                "--topk_per_sheet",
                str(ml_params["topk_per_sheet"]),
                "--rows",
                str(ml_params["rows"]),
                "--cols",
                str(ml_params["cols"]),
            ]

            commands.append(("ML", cmd_ml, project_root))
            log_paths["ML"] = log_dir / "ml.log"

    # ------------------------------------------------------------
    # Rule-based pipeline
    # ------------------------------------------------------------
    if cfg.get("rule_based", {}).get("enabled", True):
        rb_cfg = cfg["rule_based"]

        rb_entry = (base_dir / "run_rule_based_pipeline.py").resolve()
        if not rb_entry.exists():
            raise FileNotFoundError(f"Rule-based pipeline script not found: {rb_entry}")

        grid = rb_cfg["grid"]
        quad = rb_cfg["quad_detection"]
        classification = rb_cfg["classification"]
        debug = rb_cfg.get("debug", {})

        out_batch_dir = rule_output / "batch"
        overlay_collect_dir = rule_output / "overlays_collect"
        csv_dir = rule_output / "csv"

        cmd_rb = [
            sys.executable,
            str(rb_entry),
            "--input_dir",
            str(input_dir),
            "--out_batch_dir",
            str(out_batch_dir),
            "--overlay_collect_dir",
            str(overlay_collect_dir),
            "--csv_dir",
            str(csv_dir),
            "--min_height_frac",
            str(quad["min_height_frac"]),
            "--max_height_frac",
            str(quad["max_height_frac"]),
            "--iterations",
            str(quad.get("iterations", 5)),
            "--rows",
            str(grid["rows"]),
            "--cols",
            str(grid["cols"]),
            "--inner_width",
            str(grid["inner_width"]),
            "--inner_height",
            str(grid["inner_height"]),
            "--x0",
            str(grid["x0"]),
            "--y0",
            str(grid["y0"]),
            "--dx",
            str(grid["dx"]),
            "--dy",
            str(grid["dy"]),
            "--box_width",
            str(grid["box_width"]),
            "--box_height",
            str(grid["box_height"]),
            "--pad_px",
            str(grid["pad_px"]),
            "--threshold",
            str(classification["threshold"]),
            "--ignore_border_ratio",
            str(classification["ignore_border_ratio"]),
            "--overlay_thickness",
            str(debug.get("overlay_thickness", 2)),
        ]

        if bool(classification.get("save_debug_masks", False)):
            cmd_rb.append("--save_debug_masks")
        if bool(debug.get("save_steps", True)):
            cmd_rb.append("--save_steps")
        if bool(debug.get("show_score_on_overlay", False)):
            cmd_rb.append("--show_score_on_overlay")

        commands.append(("RULE_BASED", cmd_rb, project_root))
        log_paths["RULE_BASED"] = log_dir / "rule_based.log"

    if not commands:
        print("Nothing to do: no runnable pipelines are enabled or all required ML models are missing.")
        return 0

    for name, cmd, _ in commands:
        print(f"\n{name} command:")
        print("  " + " ".join(cmd))

    print(f"\nLog directory: {log_dir}")

    processes: list[tuple[str, subprocess.Popen]] = []
    open_log_files: dict[str, object] = {}

    try:
        for name, log_path in log_paths.items():
            open_log_files[name] = log_path.open("w", encoding="utf-8")

        if args.no_parallel:
            success = True
            for name, cmd, cwd in commands:
                return_code = subprocess.call(
                    cmd,
                    cwd=str(cwd),
                    stdout=open_log_files[name],
                    stderr=subprocess.STDOUT,
                )
                print(f"{name} finished with exit code {return_code}")
                success = success and (return_code == 0)
            return 0 if success else 1

        for name, cmd, cwd in commands:
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=open_log_files[name],
                stderr=subprocess.STDOUT,
            )
            processes.append((name, process))

        success = True
        for name, process in processes:
            return_code = process.wait()
            print(f"{name} finished with exit code {return_code}")
            success = success and (return_code == 0)

        return 0 if success else 1

    finally:
        for file_handle in open_log_files.values():
            try:
                file_handle.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())