# Design and Implementation of an MLOps-Enabled Framework for Autism Spectrum Disorder Diagnosis

> **An end-to-end machine learning pipeline for binary classification of Autism Spectrum Disorder (ASD) using Structural MRI (sMRI) data, integrated with MLOps tooling for reproducibility, versioning, and automated CI/CD deployment.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Pipeline Stages](#pipeline-stages)
   - [1. Data Ingestion](#1-data-ingestion)
   - [2. Preprocessing](#2-preprocessing)
   - [3. Feature Extraction](#3-feature-extraction)
   - [4. Model Training](#4-model-training)
   - [5. Model Evaluation](#5-model-evaluation)
5. [Feature Engineering](#feature-engineering)
6. [Model Registry & MLflow Tracking](#model-registry--mlflow-tracking)
7. [Web Application](#web-application)
8. [CI/CD with GitHub Actions](#cicd-with-github-actions)
9. [DVC Pipeline](#dvc-pipeline)
10. [Docker Deployment](#docker-deployment)
11. [Configuration](#configuration)
12. [Setup & Installation](#setup--installation)
13. [Running the Pipeline](#running-the-pipeline)
14. [Running the Web Application](#running-the-web-application)
15. [Environment Variables](#environment-variables)
16. [Dependencies](#dependencies)
17. [MLOps Tools Used](#mlops-tools-used)

---

## Project Overview

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition that is typically diagnosed through behavioral assessment. This project explores an alternative, neuroimaging-based approach — using **Structural MRI (sMRI)** brain scans to train a machine learning model capable of classifying subjects as **Autistic** or **Non-Autistic**.

### Key Goals

- Build a **production-grade MLOps pipeline** for ASD diagnosis using sMRI data.
- Extract interpretable handcrafted features (GLCM, LBP, GFCC) from MRI brain slices.
- Train and evaluate multiple classical ML classifiers with rigorous cross-validation.
- Integrate **MLflow** for experiment tracking and model versioning via **DagsHub**.
- Automate model promotion with overfit detection and drift monitoring.
- Serve predictions through a **Flask web application** with a modern UI.
- Containerize and deploy via **Docker** with a fully automated **GitHub Actions** CI/CD pipeline.

---

## Architecture Overview

```
sMRI Data (NIfTI / MGZ)
        │
        ▼
┌──────────────────┐
│  Data Ingestion  │  ← Copies dataset to artifacts/data_ingestion/
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Preprocessing   │  ← Train/test split + slice extraction (PNG)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  Feature Extraction      │  ← GLCM (7) + LBP (256) + GFCC (6) = 269 features
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Model Training          │  ← LR, RF, SVM, KNN with SMOTE + PCA + GridSearchCV
│  (MLflow tracking)       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Model Evaluation        │  ← Metrics, confusion matrix, ROC, SHAP, drift detection
│  + Model Registry        │  ← Auto-promotes best model → @staging alias
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Flask Web App           │  ← Upload sMRI scan → prediction + confidence scores
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Docker + GitHub Actions │  ← Build, push image, auto-deploy on push to main
└──────────────────────────┘
```

---

## Project Structure

```
ASD_ML_SMRI_MLOPS/
│
├── app.py                          # Flask web application (prediction server)
├── config.yaml                     # Centralized pipeline configuration
├── dvc.yaml                        # DVC pipeline stage definitions
├── dvc.lock                        # DVC reproducibility lock file
├── Dockerfile                      # Container definition
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (not committed)
├── .dvcignore                      # Files excluded from DVC tracking
├── .dockerignore                   # Files excluded from Docker build
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading and copying logic
│   │   ├── preprocessing.py        # Slice extraction and train/test split
│   │   ├── feature_extraction.py   # GLCM, LBP, GFCC feature extraction
│   │   ├── model_training.py       # Multi-model training with MLflow logging
│   │   └── model_evaluation.py     # Evaluation, SHAP, drift detection, auto-promotion
│   │
│   ├── pipeline/
│   │   ├── data_ingestion_pipeline.py
│   │   ├── preprocessing_pipeline.py
│   │   ├── feature_extraction_pipeline.py
│   │   ├── model_training_pipeline.py
│   │   └── model_evaluation_pipeline.py
│   │
│   └── utils/
│       ├── logger.py               # File-based logging utility
│       └── exception.py            # Custom exception with traceback detail
│
├── templates/
│   └── index.html                  # Frontend UI for the prediction web app
│
├── static/
│   └── style.css                   # Additional styling
│
├── artifacts/                      # Auto-generated pipeline outputs
│   ├── data_ingestion/             # Raw copied dataset
│   ├── preprocessing/              # Train/test split PNG slices
│   ├── features/                   # CSV feature matrices
│   ├── models/                     # Best model .pkl file
│   ├── evaluation/                 # Metrics, plots, SHAP outputs
│   └── temp/                       # Intermediate artifacts (CV results, etc.)
│
├── logs/                           # Timestamped log files
│
└── .github/
    └── workflows/
        ├── main_pipeline.yaml      # CI/CD: runs on push to main
        └── retraining.yaml         # Scheduled retraining (Sundays 2 AM)
```

---

## Pipeline Stages

### 1. Data Ingestion

**File:** `src/components/data_ingestion.py`

Responsible for copying the raw ASD sMRI dataset from a local source path (configured in `config.yaml`) into the `artifacts/data_ingestion/ASD_DATASET/` directory.

- Validates that the source path exists before proceeding.
- Removes any previously ingested data to ensure a fresh copy.
- Logs all steps using the centralized logger.
- Raises a `CustomException` with detailed traceback on failure.

**Config:**
```yaml
data_ingestion:
  root_dir: artifacts/data_ingestion
  source_dir: "D:/main_project/Kaggle data/ASD_DATASET"
  local_data_file: artifacts/data_ingestion/ASD_DATASET
```

---

### 2. Preprocessing

**File:** `src/components/preprocessing.py`

Converts raw 3D NIfTI (`.nii`, `.nii.gz`) brain volumes into 2D PNG slices suitable for feature extraction.

#### Steps:

1. **Train/Test Split** — Shuffles subject files and splits them per class into `train/` and `test/` subdirectories (default 80/20).
2. **RAS Reorientation** — Reorients each NIfTI volume to RAS (Right-Anterior-Superior) canonical orientation using `nibabel.as_closest_canonical`.
3. **Train Slice Extraction** — Extracts **11 axial slices** centered around the mid-sagittal plane (mid ± 5 slices) for each training subject. Each slice is rotated 90° and min-max normalized to 0–255 (uint8).
4. **Test Slice Extraction** — Extracts a **single mid-sagittal slice** per test subject.
5. All extracted slices are saved as PNG files in the respective class subdirectory.

**Config:**
```yaml
preprocessing:
  root_dir: artifacts/preprocessing
  input_dir: artifacts/data_ingestion/ASD_DATASET
  train_dir: artifacts/preprocessing/train
  test_dir: artifacts/preprocessing/test
  test_size: 0.2
  random_state: 42
  num_slices: 11
```

---

### 3. Feature Extraction

**File:** `src/components/feature_extraction.py`  
**Pipeline:** `src/pipeline/feature_extraction_pipeline.py`

The `FeatureExtractor` class processes each PNG slice through three handcrafted feature extraction methods, producing a **269-dimensional feature vector** per image.

#### Feature Methods:

| Method | Features | Description |
|--------|----------|-------------|
| **GLCM** (Gray-Level Co-occurrence Matrix) | 7 | Contrast, Correlation, Energy, Homogeneity, Mean, Variance, Entropy — computed over distances [1,2,3] and angles [0°, 45°, 90°, 135°] |
| **LBP** (Local Binary Pattern) | 256 | Normalized histogram of LBP codes (P=8 neighbors, R=1 radius) — captures local texture micro-patterns |
| **GFCC** (Geometric Features of Corpus Callosum) | 6 | Area, Perimeter, Major Axis, Minor Axis, Solidity, Extent — computed via multi-Otsu thresholding + connected component analysis |

**Total features per image: 269**

All features are saved as CSV files:
- `artifacts/features/train_features.csv`
- `artifacts/features/test_features.csv`

The pipeline also logs feature parameters and dataset statistics to **MLflow** under the `feature_extraction_experiment`.

---

### 4. Model Training

**File:** `src/components/model_training.py`  
**Pipeline:** `src/pipeline/model_training_pipeline.py`

The `ModelTrainer` class trains four classical ML classifiers with full hyperparameter tuning, class imbalance handling, and experiment tracking.

#### Training Pipeline per Classifier:

Each model is wrapped in an `imblearn.pipeline.Pipeline` with the following steps:

```
StandardScaler / RobustScaler
       ↓
VarianceThreshold (low-variance feature removal)
       ↓
SMOTE (Synthetic Minority Oversampling — k_neighbors=3)
       ↓
PCA (variance-explained threshold: 0.90 / 0.95 / 0.99)
       ↓
Classifier
```

#### Classifiers & Hyperparameter Grids:

| Classifier | Key Hyperparameters |
|------------|---------------------|
| Logistic Regression | Scaler, variance threshold, PCA variance, C (0.01–5) |
| Random Forest | Variance threshold, PCA variance, n_estimators (200–400), max_depth (5–15) |
| SVM (RBF/Linear/Poly) | Scaler, variance threshold, PCA variance, C (0.1–1), kernel |
| KNN | Scaler, variance threshold, PCA variance, n_neighbors (5–11) |

#### Cross-Validation Strategy:

- **5-fold StratifiedKFold** for hyperparameter search.
- **Primary refit metric:** `balanced_accuracy` (handles class imbalance fairly).
- **All 6 metrics tracked per fold:** balanced accuracy, recall, F1, AUC, accuracy, precision.

#### Holdout Evaluation:

- 20% of training data is stratified out as an **internal holdout set** (~500 samples).
- Final model selection uses **holdout accuracy** (not CV alone).
- An **overfitting gate**: a model is only eligible for "best model" if its CV-to-holdout accuracy gap is ≤ 0.10 (10%).

#### Logged to MLflow (per run):

- Candidate model name and best hyperparameters.
- Data version hash (MD5 of train CSV).
- CV metrics (6 metrics) + Holdout metrics (6 metrics + gap).
- PCA: number of components selected, variance explained, scree plot.
- Model artifact (`.pkl`) and CV results CSV.
- Final `best_model_summary` run with best run ID, model name, and accuracy.

The best model is saved locally at `artifacts/models/best_model.pkl` along with the list of selected features at `artifacts/models/selected_features.pkl`.

---

### 5. Model Evaluation

**File:** `src/components/model_evaluation.py`  
**Pipeline:** `src/pipeline/model_evaluation_pipeline.py`

The `ModelEvaluator` class performs comprehensive evaluation of the best trained model on the held-out test set, with automated model promotion to the MLflow Registry.

#### Evaluation Steps:

1. **Load Data & Model** — Reads test features CSV and downloads the best model artifact from MLflow using the `best_model_summary` run ID.

2. **Drift Detection** — Computes a **PCA-space drift score** by comparing the mean of PCA-transformed train and test features. If drift score > 0.15, it is flagged as "high drift" and logged.

3. **Test Set Metrics:**
   - Accuracy, Balanced Accuracy, AUC, F1, Recall, Precision.
   - Holdout-to-Test accuracy gap (overfitting indicator).
   - Overfit flag triggered if gap > 0.10.

4. **Visualizations (logged as MLflow artifacts):**
   - **Confusion Matrix** — annotated with counts and percentages (YlGnBu heatmap).
   - **ROC Curve** — with AUC score.
   - **Classification Report** — per-class precision, recall, F1.

5. **SHAP Explainability:**
   - For **RandomForest**: uses `TreeExplainer` (fast, exact).
   - For all others: uses `KernelExplainer` with 50-sample background.
   - Generates **SHAP bar plot** and **SHAP dot plot** at the PCA component level.
   - Maps SHAP importance back to original features (LBP/GLCM/GFCC) via PCA components.
   - Logged: top 15 original features CSV + bar chart, grouped feature importance (LBP vs GLCM vs GFCC) CSV + chart.
   - Logs top contributing PCA component and top 3 original features as MLflow parameters.

6. **Automated Model Promotion:**
   - Compares the new model's test accuracy against the current `@staging` model in the MLflow Model Registry.
   - Promotes (registers + sets `@staging` alias) **only if**:
     - The new model is NOT overfit (gap ≤ 0.10), **AND**
     - Its test accuracy is strictly better than the current staging model.

---

## Feature Engineering

The feature engineering follows a deliberate brain-imaging-informed approach:

| Feature Group | Rationale |
|---------------|-----------|
| **GLCM** | Captures grey-matter texture regularity in cortical regions — known to differ in ASD |
| **LBP** | Encodes local structural patterns across the brain surface from the mid-sagittal slice |
| **GFCC** | Targets the Corpus Callosum — a white matter structure whose morphology has shown consistent differences in ASD participants across multiple studies |

The corpus callosum segmentation (GFCC) uses **multi-Otsu thresholding** (3 classes) followed by morphological opening and closing with a disk-shaped structuring element, then isolates the largest connected component as the candidate Corpus Callosum region.

---

## Model Registry & MLflow Tracking

All experiments are tracked on **DagsHub MLflow**:

- **Tracking URI:** `https://dagshub.com/renjini2539thomas/ASD_ML_SMRI_MLOPS.mlflow`
- **Experiments:** `feature_extraction_experiment`, `model_training_experiment`
- **Registered Model:** `ASD_BEST_MODEL`
- **Active Alias:** `@staging` — points to the best-performing, non-overfit model version.

The production `app.py` always loads the model tagged `@staging` via:
```python
model_uri = "models:/ASD_BEST_MODEL@staging"
mlflow.sklearn.load_model(model_uri)
```

The app also checks for model version changes at runtime, lazily reloading only when the version differs (`get_model()` versioning logic).

---

## Web Application

**File:** `app.py`  
**Template:** `templates/index.html`

A **Flask**-based prediction server that accepts brain scan uploads and returns ASD predictions.

### Supported Input Formats

| Format | Description |
|--------|-------------|
| `.nii` | NIfTI 3D MRI volume |
| `.nii.gz` | Compressed NIfTI |
| `.mgz` | FreeSurfer MGZ volume |
| `.png` / `.jpg` / `.jpeg` | Pre-extracted 2D brain slice |

### Prediction Flow (for 3D volumes)

1. File is saved to a temp path.
2. NIfTI/MGZ volume is loaded and reoriented to RAS canonical orientation.
3. Mid-sagittal slice is extracted and normalized to uint8.
4. GLCM + LBP + GFCC features are computed (269-dim vector).
5. The `@staging` model is loaded from DagsHub MLflow Registry (with lazy-reload on version change).
6. Prediction and probability scores are returned as JSON.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the main prediction UI |
| `/predict` | POST | Accepts multipart file upload, returns JSON prediction |
| `/health` | GET | Health check — returns model load status |

### Prediction Response Schema

```json
{
  "prediction": "Autistic",
  "label": 1,
  "confidence": 87.43,
  "probabilities": {
    "Non-Autistic": 12.57,
    "Autistic": 87.43
  }
}
```

### Frontend UI

A modern, dark-mode glassmorphism interface built with vanilla HTML + CSS + JavaScript:

- Drag-and-drop file upload zone.
- Real-time file preview with name and size.
- Animated loading state on the "Run Prediction" button.
- Animated confidence bar meters for Autistic / Non-Autistic probabilities.
- Verdict badge with contextual color coding:
  - 🔬 Purple — Autistic
  - ✅ Green — Non-Autistic
- Responsive design for mobile screens.

---

## CI/CD with GitHub Actions

### Main Pipeline (`.github/workflows/main_pipeline.yaml`)

Triggered on every push to the `main` branch.

**CI Job:**
1. Checkout repository.
2. Set up Python 3.10 and install all dependencies.
3. Export MLflow and DagsHub secrets as environment variables.
4. Validate that `app.py` and `dvc.yaml` exist.
5. Run `dvc dag` to validate the DVC pipeline structure.
6. Start Flask app and verify the `/health` endpoint responds with `curl`.

**CD Job** (runs after CI passes):
1. Log in to DockerHub.
2. Build the Docker image: `<DOCKER_USERNAME>/asd-autism-detector:latest`.
3. Push the image to DockerHub.

### Retraining Pipeline (`.github/workflows/retraining.yaml`)

Triggered:
- **Manually** via `workflow_dispatch`.
- **Automatically every Sunday at 2:00 AM UTC** via cron schedule.

Steps:
1. Pull the latest DVC-tracked data (`dvc pull`).
2. Run feature extraction pipeline.
3. Run model training pipeline.
4. Run model evaluation pipeline (includes auto-promotion to staging).

---

## DVC Pipeline

**File:** `dvc.yaml`

DVC manages the full ML pipeline with dependency tracking and reproducibility.

```yaml
stages:
  data_ingestion   →  src/components/data_ingestion.py
  preprocessing    →  artifacts/preprocessing/  (output)
  feature_extraction → artifacts/features/     (output)
  model_training   →  artifacts/models/best_model.pkl (output)
  model_evaluation →  depends on features + best_model.pkl
```

To run the full pipeline:
```bash
dvc repro
```

To view the pipeline DAG:
```bash
dvc dag
```

---

## Docker Deployment

**File:** `Dockerfile`

```dockerfile
FROM python:3.10-slim

# System libraries (OpenCV dependencies)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Install Python dependencies + gunicorn
RUN pip install --no-cache-dir -r requirements.txt && pip install mlflow dagshub gunicorn

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

### Build & Run Locally

```bash
# Build image
docker build -t asd-autism-detector .

# Run container (pass DagsHub credentials)
docker run -p 5000:5000 \
  -e DAGSHUB_TOKEN=<your_token> \
  asd-autism-detector
```

---

## Configuration

**File:** `config.yaml`

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_dir: "D:/main_project/Kaggle data/ASD_DATASET"
  local_data_file: artifacts/data_ingestion/ASD_DATASET

preprocessing:
  root_dir: artifacts/preprocessing
  input_dir: artifacts/data_ingestion/ASD_DATASET
  train_dir: artifacts/preprocessing/train
  test_dir: artifacts/preprocessing/test
  test_size: 0.2
  random_state: 42
  num_slices: 11
```

> **Note:** Update `source_dir` to point to your local ASD dataset before running the data ingestion stage.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Git
- [DVC](https://dvc.org/)
- A [DagsHub](https://dagshub.com) account (for MLflow tracking)
- Docker (optional, for containerized deployment)

### 1. Clone the repository

```bash
git clone https://github.com/renjini2539thomas/ASD_ML_SMRI_MLOPS.git
cd ASD_ML_SMRI_MLOPS
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install mlflow dagshub
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
DAGSHUB_USERNAME=your_dagshub_username
DAGSHUB_TOKEN=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/ASD_ML_SMRI_MLOPS.mlflow
```

---

## Running the Pipeline

### Run individual stages

```bash
# Stage 1: Data Ingestion
python -m src.pipeline.data_ingestion_pipeline

# Stage 2: Preprocessing
python -m src.pipeline.preprocessing_pipeline

# Stage 3: Feature Extraction
python -m src.pipeline.feature_extraction_pipeline

# Stage 4: Model Training
python -m src.pipeline.model_training_pipeline

# Stage 5: Model Evaluation
python -m src.pipeline.model_evaluation_pipeline
```

### Run the full DVC pipeline

```bash
dvc repro
```

---

## Running the Web Application

```bash
python app.py
```

The app will start at `http://0.0.0.0:5000`.

> The app attempts to pre-load the `@staging` model from DagsHub MLflow on startup. Ensure the `DAGSHUB_TOKEN` environment variable is set.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub personal access token (used as MLflow password) |
| `MLFLOW_TRACKING_URI` | Full MLflow tracking URI on DagsHub |

GitHub Actions Secrets required:
- `DAGSHUB_USERNAME`
- `DAGSHUB_TOKEN`
- `MLFLOW_TRACKING_URI`
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `mlflow` | Experiment tracking and model registry |
| `dvc` | Pipeline reproducibility and data versioning |
| `scikit-learn` | ML models, metrics, preprocessing |
| `imbalanced-learn` | SMOTE oversampling + Pipeline |
| `nibabel` | Loading and reorienting NIfTI / MGZ MRI files |
| `nilearn` | Neuroimaging utilities |
| `opencv-python` | Image I/O and resizing |
| `scikit-image` | GLCM, LBP, morphological operations |
| `numpy` / `pandas` | Numerical computation and data handling |
| `matplotlib` / `seaborn` | Plotting (confusion matrix, ROC curves) |
| `shap` | Model explainability (SHAP values) |
| `flask` | Web application framework |
| `torch` / `torchvision` | Deep learning (available for future DenseNet integration) |
| `tensorflow` / `keras` | Deep learning backend |
| `joblib` | Model serialization |
| `python-dotenv` | Environment variable loading |
| `boto3` / `s3fs` | AWS S3 / DagsHub artifact storage |
| `gunicorn` | Production WSGI server |

---

## MLOps Tools Used

| Tool | Role |
|------|------|
| **DVC** | Pipeline stage orchestration, data and artifact versioning |
| **MLflow** | Experiment tracking, metric/param/artifact logging, model registry |
| **DagsHub** | Remote MLflow tracking server + Git + DVC remote storage |
| **GitHub Actions** | CI/CD automation — code validation, Docker build & push, scheduled retraining |
| **Docker** | Container-based deployment of the Flask prediction server |
| **SHAP** | Post-hoc model explainability at PCA and original feature level |

---

## Utilities

### Logger (`src/utils/logger.py`)

A file-based logger that writes timestamped log files to the `logs/` directory.  
Format: `[ YYYY-MM-DD HH:MM:SS ] LEVEL - message`

### Custom Exception (`src/utils/exception.py`)

A `CustomException` class that enriches error messages with:
- The source script filename.
- The exact line number where the exception occurred.

This enables precise debugging across the pipeline without relying on standard stack traces alone.

---

## Dataset

The dataset consists of structural MRI brain scans (NIfTI format) organized into two classes:

```
ASD_DATASET/
├── autistic/       # sMRI scans of ASD subjects  (label = 1)
└── non-autistic/   # sMRI scans of control subjects (label = 0)
```

> The dataset is sourced from Kaggle and is not included in this repository. Update `config.yaml` with the path to your local copy before running the ingestion stage.

---

*Built with Python 3.10 · MLflow · DVC · Flask · Docker · GitHub Actions*
