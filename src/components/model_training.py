
# import os
# import sys
# import hashlib
# import pandas as pd
# import mlflow
# import mlflow.sklearn

# from pathlib import Path
# from dotenv import load_dotenv

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
# from sklearn.decomposition import PCA

# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.metrics import make_scorer, recall_score, f1_score, balanced_accuracy_score

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier


# class ModelTrainer:

#     def __init__(self):

#         load_dotenv()

#         os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
#         os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("DAGSHUB_USERNAME")
#         os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("DAGSHUB_TOKEN")
#         os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"https://dagshub.com"
        

#         mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
#         mlflow.set_experiment("model_training_experiment")

#         self.feature_dir = Path("artifacts/features")
#         self.temp_dir = Path("artifacts/temp")
#         self.temp_dir.mkdir(parents=True, exist_ok=True)

#     # ---------------- DATA HASH ----------------
#     def get_file_hash(self, path):
#         with open(path, "rb") as f:
#             return hashlib.md5(f.read()).hexdigest()

#     # ---------------- LOAD DATA ----------------
#     def load_data(self):

#         train_df = pd.read_csv(self.feature_dir / "train_features.csv")

#         X_train = train_df.drop("label", axis=1)
#         y_train = train_df["label"]

#         return X_train, y_train

#     # ---------------- MODELS ----------------
#     def get_models(self):

#         return {

#             "logistic_regression": (
#                 Pipeline([
#                     ("scaler", StandardScaler()),
#                     ("pca", PCA()),
#                     ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))
#                 ]),
#                 {
#                     "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler(), PowerTransformer()],
#                     "pca__n_components": [20, 50, 100],
#                     "model__C": [0.01, 0.1, 1, 10]
#                 }
#             ),

#             "random_forest": (
#                 Pipeline([
#                     ("pca", PCA()),
#                     ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
#                 ]),
#                 {
#                     "pca__n_components": [20, 50, 100],
#                     "model__n_estimators": [200, 400],
#                     "model__max_depth": [5, 10, 15]
#                 }
#             ),

#             "svm": (
#                 Pipeline([
#                     ("scaler", StandardScaler()),
#                     ("pca", PCA()),
#                     ("model", SVC(probability=True, class_weight="balanced"))
#                 ]),
#                 {
#                     "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()],
#                     "model__kernel": ["rbf"],
#                     "pca__n_components": [20,50, 100],
#                     "model__C": [0.5, 1, 5],
#                     "model__gamma": ["scale"]
#                 }
#             ),

#             "knn": (
#                 Pipeline([
#                     ("scaler", StandardScaler()),
#                     ("pca", PCA()),
#                     ("model", KNeighborsClassifier())
#                 ]),
#                 {
#                     "scaler": [StandardScaler(), MinMaxScaler()],
#                     "pca__n_components": [20, 50, 100],
#                     "model__n_neighbors": [5, 7, 9, 11]
#                 }
#             )
#         }

#     # ---------------- TRAINING ----------------
#     def run(self):

#         X_train, y_train = self.load_data()

#         data_hash = self.get_file_hash(
#             self.feature_dir / "train_features.csv"
#         )

#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         scoring = {
#             "recall": make_scorer(recall_score),
#             "f1": make_scorer(f1_score),
#             "bal_acc": make_scorer(balanced_accuracy_score),
#             "auc": "roc_auc",
#             "accuracy": "accuracy"
#         }

#         best_global_score = -1
#         best_run_id = None
#         best_model_name = None

#         for name, (pipe, params) in self.get_models().items():

#             with mlflow.start_run(run_name=name) as run:

#                 print(f"🚀 Training {name}")

#                 grid = GridSearchCV(
#                     pipe,
#                     params,
#                     cv=cv,
#                     scoring=scoring,
#                     refit="bal_acc",
#                     n_jobs=-1,
#                     return_train_score=True
#                 )

#                 grid.fit(X_train, y_train)

#                 idx = grid.best_index_

#                 metrics = {
#                     "cv_recall": grid.cv_results_["mean_test_recall"][idx],
#                     "cv_f1": grid.cv_results_["mean_test_f1"][idx],
#                     "cv_balanced_accuracy": grid.cv_results_["mean_test_bal_acc"][idx],
#                     "cv_auc": grid.cv_results_["mean_test_auc"][idx],
#                     "cv_accuracy": grid.cv_results_["mean_test_accuracy"][idx]
#                 }

#                 # ---------------- LOG PARAMS ----------------
#                 mlflow.log_param("candidate_model", name)
#                 mlflow.log_param("data_version", data_hash)
#                 mlflow.log_params(grid.best_params_)

#                 # ---------------- LOG METRICS ----------------
#                 for k, v in metrics.items():
#                     mlflow.log_metric(k, v)

#                 # ---------------- LOG CV RESULTS ----------------
#                 cv_df = pd.DataFrame(grid.cv_results_)
#                 cv_path = self.temp_dir / f"{name}_cv_results.csv"
#                 cv_df.to_csv(cv_path, index=False)
#                 mlflow.log_artifact(str(cv_path))

#                 # ---------------- LOG MODEL ----------------
#                 best_model = grid.best_estimator_
#                 # print("before log model")
#                 # mlflow.sklearn.log_model(best_model, artifact_path="model")
#                 # print("after log model")
#                 import joblib

# # ---------------- SAVE MODEL LOCALLY ----------------
#                 model_path = self.temp_dir / f"{name}_model.pkl"
#                 joblib.dump(best_model, model_path)

#                 print(f"✅ Model saved locally: {model_path}")

#                 # ---------------- LOG AS ARTIFACT ----------------
#                 mlflow.log_artifact(str(model_path), artifact_path="model")

#                 print("✅ Model logged as artifact")


#                 print(f"✅ {name} → {metrics}")

#                 # ---------------- TRACK BEST ----------------
#                 if metrics["cv_balanced_accuracy"] > best_global_score:
#                     best_global_score = metrics["cv_balanced_accuracy"]
#                     best_run_id = run.info.run_id
#                     best_model_name = name

#         # ---------------- GLOBAL BEST RUN ----------------
#         with mlflow.start_run(run_name="best_model_summary"):

#             mlflow.log_param("best_run_id", best_run_id)
#             mlflow.log_param("best_model", best_model_name)
#             mlflow.log_metric("best_balanced_accuracy", best_global_score)

#         print("\n🏆 FINAL BEST MODEL")
#         print(f"Model: {best_model_name}")
#         print(f"Run ID: {best_run_id}")
#         print(f"Score: {best_global_score}")

#         return best_run_id, best_model_name, best_global_score
# import os
# import sys
# import hashlib
# import pandas as pd
# import numpy as np
# import mlflow
# import mlflow.sklearn

# from pathlib import Path
# from dotenv import load_dotenv

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.decomposition import PCA

# from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
# from sklearn.metrics import make_scorer, recall_score, f1_score, balanced_accuracy_score

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier

# import joblib


# class ModelTrainer:

#     def __init__(self):

#         load_dotenv()

#         os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
#         os.environ["AWS_ACCESS_KEY_ID"]         = os.getenv("DAGSHUB_USERNAME")
#         os.environ["AWS_SECRET_ACCESS_KEY"]     = os.getenv("DAGSHUB_TOKEN")
#         os.environ["MLFLOW_S3_ENDPOINT_URL"]    = "https://dagshub.com"

#         mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
#         mlflow.set_experiment("model_training_experiment")

#         self.feature_dir = Path("artifacts/features")
#         self.temp_dir    = Path("artifacts/temp")
#         self.temp_dir.mkdir(parents=True, exist_ok=True)

#     # ---------------- DATA HASH ----------------
#     def get_file_hash(self, path):
#         with open(path, "rb") as f:
#             return hashlib.md5(f.read()).hexdigest()

#     # ---------------- LOAD DATA ----------------
#     def load_data(self):
#         """
#         FIX 1 — 80/20 STRATIFIED SPLIT
#         ─────────────────────────────────────────────────────
#         Previously the entire train_features.csv was used for
#         CV only, and evaluation was done on a tiny 58-sample
#         test set. This caused unreliable eval metrics because:
#           - 58 samples → 1 wrong pred = 1.7% accuracy swing
#           - Class ratio not guaranteed in small fixed test set

#         Now we carve 20% out of train_features.csv as an
#         internal holdout set (≈500 samples). This gives stable
#         eval metrics WITHOUT needing new data.
#         ─────────────────────────────────────────────────────
#         """
#         train_df = pd.read_csv(self.feature_dir / "train_features.csv")

#         X = train_df.drop("label", axis=1).values
#         y = train_df["label"].values

#         X_train, X_holdout, y_train, y_holdout = train_test_split(
#             X, y,
#             test_size=0.20,       # 20% → ~500 samples for eval
#             stratify=y,           # preserves class ratio in both splits
#             random_state=42
#         )

#         print(f"Train size   : {X_train.shape[0]} samples")
#         print(f"Holdout size : {X_holdout.shape[0]} samples")
#         print(f"Train class distribution   : {dict(zip(*np.unique(y_train,   return_counts=True)))}")
#         print(f"Holdout class distribution : {dict(zip(*np.unique(y_holdout, return_counts=True)))}")

#         return X_train, y_train, X_holdout, y_holdout

#     # ---------------- MODELS ----------------
#     def get_models(self):
#         """
#         FIX 2 — HYPERPARAMETER GRID CHANGES
#         ─────────────────────────────────────────────────────
#         SVM:
#           - Removed C=5       → too aggressive, caused overfitting
#           - Removed MinMaxScaler → sensitive to MRI intensity outliers
#           - PCA capped at 50  → 100 dims for 58 test samples was
#                                  3.4 dims/sample (way too high)

#         All models:
#           - PCA max capped at 50 for same reason
#           - StandardScaler and RobustScaler kept (both stable)
#           - MinMaxScaler removed from SVM grid entirely

#         class_weight="balanced" already present — kept as-is.
#         ─────────────────────────────────────────────────────
#         """
#         return {

#             "logistic_regression": (
#                 Pipeline([
#                     ("scaler", StandardScaler()),
#                     ("pca",    PCA()),
#                     ("model",  LogisticRegression(
#                         max_iter=3000,
#                         class_weight="balanced"
#                     ))
#                 ]),
#                 {
#                     "scaler":          [StandardScaler(), RobustScaler()],
#                     "pca__n_components": [20, 50,100,150,200],          # removed 100
#                     "model__C":        [0.01, 0.1, 1, 5]
#                 }
#             ),

#             "random_forest": (
#                 Pipeline([
#                     ("pca",   PCA()),
#                     ("model", RandomForestClassifier(
#                         class_weight="balanced",
#                         random_state=42
#                     ))
#                 ]),
#                 {
#                     "pca__n_components":  [20, 50,100,150,200],           # removed 100
#                     "model__n_estimators": [200, 400],
#                     "model__max_depth":   [5, 10, 15]
#                 }
#             ),

#             "svm": (
#                 Pipeline([
#                     ("scaler", StandardScaler()),
#                     ("pca",    PCA()),
#                     ("model",  SVC(
#                         probability=True,
#                         class_weight="balanced"
#                     ))
#                 ]),
#                 {
#                     # FIX: removed MinMaxScaler — unstable for MRI features
#                     "scaler":            [StandardScaler(), RobustScaler()],
#                     "pca__n_components": [20, 50,100,150,200],           # removed 100
#                     # FIX: removed C=5 — was causing overfitting
#                     "model__C":          [0.1, 0.5, 1.0],
#                     "model__kernel":     ["rbf"],
#                     "model__gamma":      ["scale"]
#                 }
#             ),

#             "knn": (
#                 Pipeline([
#                     ("scaler", StandardScaler()),
#                     ("pca",    PCA()),
#                     ("model",  KNeighborsClassifier())
#                 ]),
#                 {
#                     "scaler":            [StandardScaler(), RobustScaler()],
#                     "pca__n_components": [20, 50,100,150,200],           # removed 100
#                     "model__n_neighbors": [5, 7, 9, 11]
#                 }
#             )
#         }

#     # ---------------- TRAINING ----------------
#     def run(self):

#         X_train, y_train, X_holdout, y_holdout = self.load_data()

#         data_hash = self.get_file_hash(
#             self.feature_dir / "train_features.csv"
#         )

#         # FIX 3 — StratifiedKFold already correct — kept as-is
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         # FIX 4 — RECALL SCORER WITH EXPLICIT pos_label
#         # ─────────────────────────────────────────────────
#         # Previously recall_score was called without pos_label.
#         # For binary classification pos_label=1 (autistic) must
#         # be explicit — otherwise sklearn may score the wrong class
#         # depending on label ordering.
#         # ─────────────────────────────────────────────────
#         scoring = {
#             "recall":   make_scorer(recall_score,            pos_label=1),
#             "f1":       make_scorer(f1_score,                pos_label=1),
#             "bal_acc":  make_scorer(balanced_accuracy_score),
#             "auc":      "roc_auc",
#             "accuracy": "accuracy"
#         }

#         best_global_score = -1
#         best_run_id       = None
#         best_model_name   = None

#         for name, (pipe, params) in self.get_models().items():

#             with mlflow.start_run(run_name=name) as run:

#                 print(f"\nTraining: {name}")

#                 grid = GridSearchCV(
#                     pipe,
#                     params,
#                     cv=cv,
#                     scoring=scoring,
#                     refit="recall",      # optimise for balanced accuracy
#                     n_jobs=-1,
#                     return_train_score=True
#                 )

#                 grid.fit(X_train, y_train)

#                 idx = grid.best_index_

#                 # ---------------- CV METRICS ----------------
#                 cv_metrics = {
#                     "cv_recall":            grid.cv_results_["mean_test_recall"][idx],
#                     "cv_f1":                grid.cv_results_["mean_test_f1"][idx],
#                     "cv_balanced_accuracy": grid.cv_results_["mean_test_bal_acc"][idx],
#                     "cv_auc":               grid.cv_results_["mean_test_auc"][idx],
#                     "cv_accuracy":          grid.cv_results_["mean_test_accuracy"][idx]
#                 }

#                 # FIX 5 — HOLDOUT EVALUATION
#                 # ─────────────────────────────────────────────────
#                 # Instead of relying on the tiny 58-sample external
#                 # test set, we evaluate on our internal 500-sample
#                 # holdout that was stratified from the same dataset.
#                 # This gives stable, trustworthy eval metrics.
#                 # ─────────────────────────────────────────────────
#                 best_model   = grid.best_estimator_
#                 y_pred       = best_model.predict(X_holdout)
#                 y_pred_proba = best_model.predict_proba(X_holdout)[:, 1]

#                 from sklearn.metrics import (
#                     accuracy_score, roc_auc_score,
#                     precision_score
#                 )

#                 holdout_metrics = {
#                     "holdout_accuracy":          accuracy_score(y_holdout, y_pred),
#                     "holdout_balanced_accuracy": balanced_accuracy_score(y_holdout, y_pred),
#                     "holdout_auc":               roc_auc_score(y_holdout, y_pred_proba),
#                     "holdout_f1":                f1_score(y_holdout, y_pred,      pos_label=1),
#                     "holdout_recall":            recall_score(y_holdout, y_pred,   pos_label=1),
#                     "holdout_precision":         precision_score(y_holdout, y_pred, pos_label=1)
#                 }

#                 # Gap metric — key overfitting indicator
#                 # If this is > 0.10 (10%), overfitting is present
#                 gap = cv_metrics["cv_balanced_accuracy"] - holdout_metrics["holdout_balanced_accuracy"]
#                 holdout_metrics["cv_holdout_gap"] = gap

#                 # ---------------- LOG PARAMS ----------------
#                 mlflow.log_param("candidate_model", name)
#                 mlflow.log_param("data_version",    data_hash)
#                 mlflow.log_params(grid.best_params_)

#                 # ---------------- LOG ALL METRICS ----------------
#                 for k, v in cv_metrics.items():
#                     mlflow.log_metric(k, v)

#                 for k, v in holdout_metrics.items():
#                     mlflow.log_metric(k, v)

#                 # ---------------- LOG CV RESULTS ----------------
#                 cv_df   = pd.DataFrame(grid.cv_results_)
#                 cv_path = self.temp_dir / f"{name}_cv_results.csv"
#                 cv_df.to_csv(cv_path, index=False)
#                 mlflow.log_artifact(str(cv_path))

#                 # ---------------- SAVE + LOG MODEL ----------------
#                 model_path = self.temp_dir / f"{name}_model.pkl"
#                 joblib.dump(best_model, model_path)
#                 mlflow.log_artifact(str(model_path), artifact_path="model")

#                 self.model_dir = Path("artifacts/models")
#                 self.model_dir.mkdir(parents=True, exist_ok=True)

#                 best_model_path = self.model_dir / "best_model.pkl"
#                 joblib.dump(best_model, best_model_path)

#                 # ---------------- PRINT SUMMARY ----------------
#                 print(f"CV metrics      : {cv_metrics}")
#                 print(f"Holdout metrics : {holdout_metrics}")
#                 print(f"Overfit gap     : {gap:.4f} {'⚠ OVERFIT' if gap > 0.10 else '✓ OK'}")

#                 # ---------------- TRACK BEST ----------------
#                 # Use holdout balanced accuracy (not CV) to select
#                 # best model — more honest indicator of real performance
#                 if holdout_metrics["holdout_recall"] > best_global_score:
#                     best_global_score = holdout_metrics["holdout_recall"]
#                     best_run_id       = run.info.run_id
#                     best_model_name   = name

#         # ---------------- GLOBAL BEST RUN ----------------
#         with mlflow.start_run(run_name="best_model_summary"):
#             mlflow.log_param("best_run_id",  best_run_id)
#             mlflow.log_param("best_model",   best_model_name)
#             mlflow.log_metric("best_holdout_recall", best_global_score)

#         print("\nFINAL BEST MODEL")
#         print(f"Model : {best_model_name}")
#         print(f"Run ID: {best_run_id}")
#         print(f"Score : {best_global_score:.4f}")

#         return best_run_id, best_model_name, best_global_score
import os
import hashlib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from pathlib import Path
from dotenv import load_dotenv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    make_scorer, recall_score, f1_score,
    balanced_accuracy_score, accuracy_score,
    roc_auc_score, precision_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# COMPOSITE SCORING STRATEGY
# ============================================================
# Single-metric selection is unstable for medical tasks:
#   recall=1.0  is achievable by predicting all-positive
#   accuracy    is biased toward the majority class
#   f1          ignores true negatives entirely
#
# Composite score combines four metrics with clinical weights:
#   recall            0.35  — missing autism is worst outcome
#   balanced_accuracy 0.30  — penalizes class bias
#   auc               0.25  — overall discrimination power
#   f1                0.10  — precision/recall balance
#
# Minimum thresholds act as a gate — a model must pass ALL
# thresholds or it is disqualified (composite = -1.0).
# This prevents degenerate models from being selected.
# ============================================================

COMPOSITE_WEIGHTS = {
    "recall":            0.50,
    "balanced_accuracy": 0.20,
    "auc":               0.20,
    "f1":                0.10,
}

MIN_THRESHOLDS = {
    "recall":            0.72,
    "balanced_accuracy": 0.70,
    "auc":               0.75,
    "f1":                0.60,
}


def compute_composite_score(metrics: dict) -> float:
    """
    Returns weighted composite score from holdout metrics.
    Returns -1.0 if any minimum threshold is not met.
    """
    for key, threshold in MIN_THRESHOLDS.items():
        value = metrics.get(f"holdout_{key}", 0.0)
        if value < threshold:
            print(f"  FAILED threshold: holdout_{key} = {value:.4f} < {threshold}")
            return -1.0

    return sum(
        COMPOSITE_WEIGHTS[key] * metrics[f"holdout_{key}"]
        for key in COMPOSITE_WEIGHTS
    )


class ModelTrainer:

    def __init__(self):

        load_dotenv()

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
        os.environ["AWS_ACCESS_KEY_ID"]         = os.getenv("DAGSHUB_USERNAME")
        os.environ["AWS_SECRET_ACCESS_KEY"]     = os.getenv("DAGSHUB_TOKEN")
        os.environ["MLFLOW_S3_ENDPOINT_URL"]    = "https://dagshub.com"

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("model_training_experiment")

        self.feature_dir = Path("artifacts/features")
        self.model_dir   = Path("artifacts/models")
        self.temp_dir    = Path("artifacts/temp")

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- DATA HASH ----------------
    def get_file_hash(self, path):
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    # ---------------- LOAD DATA ----------------
    def load_data(self):
        """
        80/20 stratified holdout split from train_features.csv.
        random_state=42 is fixed so the split is identical
        across every run — holdout never leaks into training.
        """
        train_df = pd.read_csv(self.feature_dir / "train_features.csv")

        X = train_df.drop("label", axis=1).values
        y = train_df["label"].values

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y,
            test_size=0.20,
            stratify=y,
            random_state=42
        )

        print(f"Train size        : {X_train.shape[0]}")
        print(f"Holdout size      : {X_holdout.shape[0]}")
        print(f"Train distribution: {dict(zip(*np.unique(y_train,   return_counts=True)))}")
        print(f"Holdout distribution: {dict(zip(*np.unique(y_holdout, return_counts=True)))}")

        return X_train, y_train, X_holdout, y_holdout

    # ---------------- MODELS ----------------
    def get_models(self):

        return {

            "logistic_regression": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca",    PCA()),
                    ("model",  LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced"
                    ))
                ]),
                {
                    "scaler":            [StandardScaler(), RobustScaler()],
                    "pca__n_components": [20, 50, 100,150,200],
                    "model__C":          [0.01, 0.1, 1.0, 5.0]
                }
            ),

            "random_forest": (
                Pipeline([
                    ("pca",   PCA()),
                    ("model", RandomForestClassifier(
                        class_weight="balanced",
                        random_state=42
                    ))
                ]),
                {
                    "pca__n_components":   [20, 50, 100,150,200],
                    "model__n_estimators": [200, 400],
                    "model__max_depth":    [5, 10, 15]
                }
            ),

            "svm": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca",    PCA()),
                    ("model",  SVC(
                        probability=True,
                        class_weight="balanced"
                    ))
                ]),
                {
                    "scaler":            [StandardScaler(), RobustScaler()],
                    "pca__n_components": [20, 50, 100,150,200],
                    "model__C":          [0.1, 0.5, 1.0],
                    "model__kernel":     ["rbf"],
                    "model__gamma":      ["scale"]
                }
            ),

            "knn": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca",    PCA()),
                    ("model",  KNeighborsClassifier())
                ]),
                {
                    "scaler":             [StandardScaler(), RobustScaler()],
                    "pca__n_components":  [20, 50, 100,150,200],
                    "model__n_neighbors": [5, 7, 9, 11]
                }
            )
        }

    # ---------------- TRAINING ----------------
    def run(self):

        X_train, y_train, X_holdout, y_holdout = self.load_data()
        data_hash = self.get_file_hash(self.feature_dir / "train_features.csv")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scoring = {
            "recall":   make_scorer(recall_score,            pos_label=1),
            "f1":       make_scorer(f1_score,                pos_label=1),
            "bal_acc":  make_scorer(balanced_accuracy_score),
            "auc":      "roc_auc",
            "accuracy": "accuracy"
        }

        best_global_composite = -1.0
        best_run_id           = None
        best_model_name       = None
        results_summary       = []

        for name, (pipe, params) in self.get_models().items():

            with mlflow.start_run(run_name=name) as run:

                print(f"\n{'='*50}")
                print(f"Training: {name}")
                print(f"{'='*50}")

                grid = GridSearchCV(
                    pipe,
                    params,
                    cv=cv,
                    scoring=scoring,
                    refit="bal_acc",     # stable CV refit — cannot be gamed
                    n_jobs=-1,
                    return_train_score=True
                )

                grid.fit(X_train, y_train)
                idx = grid.best_index_

                # ---------------- CV METRICS ----------------
                cv_metrics = {
                    "cv_recall":            grid.cv_results_["mean_test_recall"][idx],
                    "cv_f1":                grid.cv_results_["mean_test_f1"][idx],
                    "cv_balanced_accuracy": grid.cv_results_["mean_test_bal_acc"][idx],
                    "cv_auc":               grid.cv_results_["mean_test_auc"][idx],
                    "cv_accuracy":          grid.cv_results_["mean_test_accuracy"][idx]
                }

                # ---------------- HOLDOUT METRICS ----------------
                best_model   = grid.best_estimator_
                y_pred       = best_model.predict(X_holdout)
                y_pred_proba = best_model.predict_proba(X_holdout)[:, 1]

                holdout_metrics = {
                    "holdout_accuracy":          accuracy_score(y_holdout, y_pred),
                    "holdout_balanced_accuracy": balanced_accuracy_score(y_holdout, y_pred),
                    "holdout_auc":               roc_auc_score(y_holdout, y_pred_proba),
                    "holdout_f1":                f1_score(y_holdout, y_pred,       pos_label=1),
                    "holdout_recall":            recall_score(y_holdout, y_pred,    pos_label=1),
                    "holdout_precision":         precision_score(y_holdout, y_pred, pos_label=1)
                }

                # ---------------- COMPOSITE SCORE ----------------
                composite = compute_composite_score(holdout_metrics)
                holdout_metrics["holdout_composite_score"] = composite

                # ---------------- OVERFITTING GAP ----------------
                gap = (
                    cv_metrics["cv_balanced_accuracy"]
                    - holdout_metrics["holdout_balanced_accuracy"]
                )
                holdout_metrics["cv_holdout_gap"] = gap

                # ---------------- LOG PARAMS ----------------
                mlflow.log_param("candidate_model", name)
                mlflow.log_param("data_version",    data_hash)
                mlflow.log_params(grid.best_params_)

                # ---------------- LOG METRICS ----------------
                for k, v in cv_metrics.items():
                    mlflow.log_metric(k, v)
                for k, v in holdout_metrics.items():
                    mlflow.log_metric(k, v)

                # ---------------- SAVE MODEL PER CANDIDATE ----------------
                model_path = self.temp_dir / f"{name}_model.pkl"
                joblib.dump(best_model, model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")

                cv_df   = pd.DataFrame(grid.cv_results_)
                cv_path = self.temp_dir / f"{name}_cv_results.csv"
                cv_df.to_csv(cv_path, index=False)
                mlflow.log_artifact(str(cv_path))

                print(f"CV metrics       : {cv_metrics}")
                print(f"Holdout metrics  : {holdout_metrics}")
                print(f"Composite score  : {composite:.4f}")
                print(f"Overfit gap      : {gap:.4f}  "
                      f"{'[OVERFIT]' if gap > 0.10 else '[OK]'}")

                results_summary.append({
                    "name":    name,
                    "run_id":  run.info.run_id,
                    "composite": composite,
                    **holdout_metrics
                })

                # ---------------- TRACK BEST ----------------
                # Save best_model.pkl ONLY when a new best is found.
                # This guarantees best_model.pkl = true best model,
                # regardless of which model finishes training last.
                if composite > best_global_composite:
                    best_global_composite = composite
                    best_run_id           = run.info.run_id
                    best_model_name       = name

                    best_model_path = self.model_dir / "best_model.pkl"
                    joblib.dump(best_model, best_model_path)
                    print(f"New best saved: {best_model_name} "
                          f"(composite={composite:.4f})")

        # ---------------- COMPARISON TABLE ----------------
        results_df = pd.DataFrame(results_summary).sort_values(
            "composite", ascending=False
        )
        print("\nMODEL COMPARISON")
        print(results_df[[
            "name", "composite", "holdout_recall",
            "holdout_balanced_accuracy", "holdout_auc",
            "holdout_f1", "cv_holdout_gap"
        ]].to_string(index=False))

        table_path = self.temp_dir / "model_comparison.csv"
        results_df.to_csv(table_path, index=False)

        # ---------------- BEST MODEL SUMMARY RUN ----------------
        with mlflow.start_run(run_name="best_model_summary"):
            mlflow.log_param("best_run_id",  best_run_id  or "none")
            mlflow.log_param("best_model",   best_model_name or "none")
            mlflow.log_metric("best_composite_score", best_global_composite)
            mlflow.log_artifact(str(table_path))

        if best_run_id is None:
            print("\nWARNING: No model passed minimum thresholds.")
            print("Relax MIN_THRESHOLDS or check feature extraction.")
        else:
            print(f"\nFINAL BEST MODEL : {best_model_name}")
            print(f"Run ID           : {best_run_id}")
            print(f"Composite score  : {best_global_composite:.4f}")

        return best_run_id, best_model_name, best_global_composite