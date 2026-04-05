
import os
import sys
import hashlib
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from dotenv import load_dotenv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, f1_score, balanced_accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class ModelTrainer:

    def __init__(self):

        load_dotenv()

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("DAGSHUB_TOKEN")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"https://dagshub.com"
        

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("model_training_experiment")

        self.feature_dir = Path("artifacts/features")
        self.temp_dir = Path("artifacts/temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- DATA HASH ----------------
    def get_file_hash(self, path):
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    # ---------------- LOAD DATA ----------------
    def load_data(self):

        train_df = pd.read_csv(self.feature_dir / "train_features.csv")

        X_train = train_df.drop("label", axis=1)
        y_train = train_df["label"]

        return X_train, y_train

    # ---------------- MODELS ----------------
    def get_models(self):

        return {

            "logistic_regression": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA()),
                    ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))
                ]),
                {
                    "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler(), PowerTransformer()],
                    "pca__n_components": [50, 100, 150, 200],
                    "model__C": [0.01, 0.1, 1, 10]
                }
            ),

            "random_forest": (
                Pipeline([
                    # ("pca", PCA()),
                    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
                ]),
                {
                    # "pca__n_components": [50, 100, 150, 200],
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [5, 10, 15]
                }
            ),

            "svm": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA()),
                    ("model", SVC(probability=True, class_weight="balanced"))
                ]),
                {
                    "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()],
                    "model__kernel": ["rbf", "linear"],
                    "pca__n_components": [50, 100, 150, 200],
                    "model__C": [0.5, 1, 5]
                }
            ),

            "knn": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA()),
                    ("model", KNeighborsClassifier())
                ]),
                {
                    "scaler": [StandardScaler(), MinMaxScaler()],
                    "pca__n_components": [50, 100, 150, 200],
                    "model__n_neighbors": [5, 7, 9, 11]
                }
            )
        }

    # ---------------- TRAINING ----------------
    def run(self):

        X_train, y_train = self.load_data()

        data_hash = self.get_file_hash(
            self.feature_dir / "train_features.csv"
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scoring = {
            "recall": make_scorer(recall_score),
            "f1": make_scorer(f1_score),
            "bal_acc": make_scorer(balanced_accuracy_score),
            "auc": "roc_auc",
            "accuracy": "accuracy"
        }

        best_global_score = -1
        best_run_id = None
        best_model_name = None

        for name, (pipe, params) in self.get_models().items():

            with mlflow.start_run(run_name=name) as run:

                print(f"🚀 Training {name}")

                grid = GridSearchCV(
                    pipe,
                    params,
                    cv=cv,
                    scoring=scoring,
                    refit="bal_acc",
                    n_jobs=-1,
                    return_train_score=True
                )

                grid.fit(X_train, y_train)

                idx = grid.best_index_

                metrics = {
                    "cv_recall": grid.cv_results_["mean_test_recall"][idx],
                    "cv_f1": grid.cv_results_["mean_test_f1"][idx],
                    "cv_balanced_accuracy": grid.cv_results_["mean_test_bal_acc"][idx],
                    "cv_auc": grid.cv_results_["mean_test_auc"][idx],
                    "cv_accuracy": grid.cv_results_["mean_test_accuracy"][idx]
                }

                # ---------------- LOG PARAMS ----------------
                mlflow.log_param("candidate_model", name)
                mlflow.log_param("data_version", data_hash)
                mlflow.log_params(grid.best_params_)

                # ---------------- LOG METRICS ----------------
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

                # ---------------- LOG CV RESULTS ----------------
                cv_df = pd.DataFrame(grid.cv_results_)
                cv_path = self.temp_dir / f"{name}_cv_results.csv"
                cv_df.to_csv(cv_path, index=False)
                mlflow.log_artifact(str(cv_path))

                # ---------------- LOG MODEL ----------------
                best_model = grid.best_estimator_
                # print("before log model")
                # mlflow.sklearn.log_model(best_model, artifact_path="model")
                # print("after log model")
                import joblib

# ---------------- SAVE MODEL LOCALLY ----------------
                model_path = self.temp_dir / f"{name}_model.pkl"
                joblib.dump(best_model, model_path)

                print(f"✅ Model saved locally: {model_path}")

                # ---------------- LOG AS ARTIFACT ----------------
                mlflow.log_artifact(str(model_path), artifact_path="model")

                print("✅ Model logged as artifact")


                print(f"✅ {name} → {metrics}")

                # ---------------- TRACK BEST ----------------
                if metrics["cv_balanced_accuracy"] > best_global_score:
                    best_global_score = metrics["cv_balanced_accuracy"]
                    best_run_id = run.info.run_id
                    best_model_name = name

        # ---------------- GLOBAL BEST RUN ----------------
        with mlflow.start_run(run_name="best_model_summary"):

            mlflow.log_param("best_run_id", best_run_id)
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_balanced_accuracy", best_global_score)

        print("\n🏆 FINAL BEST MODEL")
        print(f"Model: {best_model_name}")
        print(f"Run ID: {best_run_id}")
        print(f"Score: {best_global_score}")

        return best_run_id, best_model_name, best_global_score
