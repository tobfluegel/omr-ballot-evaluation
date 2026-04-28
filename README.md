# OMR Ballot Evaluation: Rule-Based vs Machine Learning

This repository contains the implementation used in a bachelor's thesis to compare a rule-based and a machine learning–based pipeline for the automated evaluation of paper ballots.

## Overview

The system processes scanned or photographed ballots and classifies checkbox fields as marked or unmarked.

Two approaches are implemented:

- Rule-based pipeline 
  Based on image processing and thresholding (dark pixel ratio via Otsu).

- Machine learning pipeline  
  Uses object detection (YOLO) for localization and an SVM classifier with HOG features for classification.

The goal is to evaluate differences in accuracy, robustness, and error behavior.

---

## Pipeline Structure

### Machine Learning Pipeline

1. Marker detection (YOLO)
2. Perspective transformation (warp)
3. Checkbox detection (YOLO)
4. ROI extraction
5. Feature extraction (HOG + optional dark ratio)
6. Classification (SVM)
7. Result export (CSV + overlay)

### Rule-Based Pipeline

1. Quadrilateral detection
2. Iterative perspective normalization
3. Grid-based ROI extraction
4. Dark pixel ratio computation (Otsu thresholding)
5. Threshold-based classification
6. Result export (CSV + overlay)

---


## Installation

Requirements:

- Python 3.10+
- OpenCV
- NumPy
- Pandas
- scikit-image
- joblib
- ultralytics (YOLO)

Install dependencies:

pip install -r requirements.txt

---

## Quick Start

1. Clone the repository:

git clone https://github.com/tobfluegel/omr-ballot-evaluation.git
cd omr-ballot-evaluation

2. Install dependencies:

pip install -r requirements.txt

3. (Optional) Place trained models in:

models/

4. Run the pipeline:

python src/compare_pipelines.py

---

## Usage

Run both pipelines:

python src/compare_pipelines.py

Optional arguments:

--config config/config.yaml
--run-name test_run
--no-parallel

---

## Configuration

All parameters are defined in:

config/config.yaml

This includes:

- input/output paths
- model paths
- YOLO parameters
- SVM settings
- grid configuration (rule-based pipeline)
- threshold parameters

---

## Models

The trained models are NOT included in this repository.

The rule-based pipeline can be executed without any additional files.

The machine learning pipeline requires the following models:

- marker_model.pt (YOLO model for corner marker detection)
- checkbox_model.pt (YOLO model for checkbox detection)
- svm_model.joblib (SVM classifier)

These models must be placed in the `models/` directory and referenced in `config/config.yaml`.

If the models are not provided, only the rule-based pipeline will run.
The trained models can be provided upon request.

---

## Labeling Tool (Optional)

A small interactive tool is included to manually review and correct ROI labels.

Location:
tools/review_labels.py

Usage:

python tools/review_labels.py --input_csv path/to/input.csv --output_csv path/to/output.csv

Controls:

- 0 = empty  
- 1 = marked  
- s = skip  
- b = undo last label  
- q = quit  

This tool was used to manually create and verify ground truth labels for evaluation and, optionally, for training data preparation.

---

## Evaluation Context

The system was evaluated on a dataset of scanned ballots with the following characteristics:

- fixed layout (grid-based answer fields)
- varying lighting conditions
- perspective distortions (camera images)
- different marking styles

Metrics used:

- Precision
- Recall
- F1-score

The machine learning pipeline achieved significantly higher recall, while the rule-based approach showed weaknesses near threshold boundaries.

---

## Known Limitations

Rule-based pipeline:

- sensitive to lighting variations  
- sensitive to weak markings  
- fixed threshold leads to classification errors near the decision boundary  

Machine learning pipeline:

- depends on correct marker detection  
- sensitive to imperfect geometric normalization  
- errors may occur when marker estimation is required  

## License

This project is intended for academic use.
