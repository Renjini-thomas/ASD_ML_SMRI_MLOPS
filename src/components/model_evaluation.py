
# import os
# import pandas as pd
# import mlflow
# import mlflow.sklearn
# import matplotlib.pyplot as plt
# import seaborn as sns

# from pathlib import Path
# from dotenv import load_dotenv
# from mlflow.tracking import MlflowClient

# from sklearn.metrics import (
#     recall_score,
#     f1_score,
#     accuracy_score,
#     roc_auc_score,
#     confusion_matrix,
#     classification_report,
#     RocCurveDisplay,
#     balanced_accuracy_score,
#     precision_score
# )


# class ModelEvaluator:

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
#         self.eval_dir = Path("artifacts/evaluation")
#         self.eval_dir.mkdir(parents=True, exist_ok=True)

#         self.model_name = "ASD_BEST_MODEL"

#     # ---------------- GET BEST RUN ----------------
#     def get_best_run(self):

#         client = MlflowClient()
#         exp = client.get_experiment_by_name("model_training_experiment")

#         runs = client.search_runs(
#             experiment_ids=[exp.experiment_id],
#             filter_string="tags.mlflow.runName = 'best_model_summary'",
#             order_by=["start_time DESC"],
#             max_results=1
#         )

#         if len(runs) == 0:
#             raise Exception("❌ No best_model_summary run found")

#         summary_run = runs[0]

#         best_run_id = summary_run.data.params.get("best_run_id")

#         if best_run_id is None:
#             raise Exception("❌ best_run_id not found in summary run")

#         print("⭐ Best Training Run ID:", best_run_id)

#         return best_run_id

#     # ---------------- LOAD DATA + MODEL ----------------
#     def load_data(self):

#         test_df = pd.read_csv(self.feature_dir / "test_features.csv")

#         X_test = test_df.drop("label", axis=1)
#         y_test = test_df["label"]

#         run_id = self.get_best_run()

#         import joblib
#         import os
#         from mlflow.tracking import MlflowClient

#         client = MlflowClient()

#         # Download model artifact
#         local_dir = client.download_artifacts(run_id, "model")

#         print(f"📦 Model downloaded to: {local_dir}")

#         # Find .pkl file
#         model_file = None
#         for f in os.listdir(local_dir):
#             if f.endswith(".pkl"):
#                 model_file = f
#                 break

#         if model_file is None:
#             raise Exception("❌ No model file found")

#         model_path = os.path.join(local_dir, model_file)

#         # Load model
#         model = joblib.load(model_path)

#         print(f"✅ Model loaded: {model_path}")

#         return X_test, y_test, model, run_id

#     # ---------------- EVALUATION ----------------
#     def evaluate(self):

#         X_test, y_test, model, run_id = self.load_data()
#         client = MlflowClient()

#         with mlflow.start_run(run_name="evaluation_stage"):

#             # ---------------- BASIC INFO ----------------
#             mlflow.log_param("source_run_id", run_id)
#             mlflow.log_param("model_type", model.named_steps["model"].__class__.__name__)

#             # ---------------- PREDICTIONS ----------------
#             y_pred = model.predict(X_test)
#             y_prob = model.predict_proba(X_test)[:, 1]

#             # ---------------- METRICS ----------------
#             metrics = {
#                 "eval_recall": recall_score(y_test, y_pred),
#                 "eval_f1": f1_score(y_test, y_pred),
#                 "eval_accuracy": accuracy_score(y_test, y_pred),
#                 "eval_auc": roc_auc_score(y_test, y_prob),
#                 "eval_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
#                 "eval_precision": precision_score(y_test, y_pred),
#             }

#             for k, v in metrics.items():
#                 mlflow.log_metric(k, v)

#             # ---------------- CONFUSION MATRIX ----------------
#             cm = confusion_matrix(y_test, y_pred)

#             import seaborn as sns
#             import matplotlib.pyplot as plt
#             import numpy as np

#             # Better style
#             sns.set(style="whitegrid", font_scale=1.2)

#             plt.figure(figsize=(7, 6))

#             # Optional: add percentages
#             cm_percent = cm / cm.sum(axis=1, keepdims=True)

#             labels = np.array([
#                 [f"{cm[i,j]}\n({cm_percent[i,j]*100:.1f}%)" for j in range(cm.shape[1])]
#                 for i in range(cm.shape[0])
#             ])

#             sns.heatmap(
#                 cm,
#                 annot=labels,
#                 fmt="",
#                 cmap="YlGnBu",  # 🔥 clean professional color
#                 cbar=True,
#                 xticklabels=["Non-ASD", "ASD"],
#                 yticklabels=["Non-ASD", "ASD"],
#                 linewidths=0.5,
#                 linecolor="gray"
#             )

#             plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
#             plt.xlabel("Predicted Label", fontsize=12)
#             plt.ylabel("Actual Label", fontsize=12)

#             plt.tight_layout()

#             cm_path = self.eval_dir / "confusion_matrix.png"
#             plt.savefig(cm_path, dpi=300)
#             plt.close()

#             mlflow.log_artifact(str(cm_path))

#             # ---------------- ROC CURVE ----------------
#             plt.figure()
#             RocCurveDisplay.from_predictions(y_test, y_prob)

#             roc_path = self.eval_dir / "roc_curve.png"
#             plt.savefig(roc_path)
#             plt.close()

#             mlflow.log_artifact(str(roc_path))

#             # ---------------- CLASSIFICATION REPORT ----------------
#             report = classification_report(y_test, y_pred)

#             report_path = self.eval_dir / "classification_report.txt"
#             with open(report_path, "w") as f:
#                 f.write(report)

#             mlflow.log_artifact(str(report_path))

#             print("✅ Evaluation metrics & artifacts logged")

#             # ---------------- STAGING MODEL CHECK ----------------
#             # ---------------- GET PREVIOUS STAGING ----------------
#             # ---------------- STAGING MODEL CHECK ----------------
#             staging_score = 0.0

#             try:
#                 model_version = client.get_model_version_by_alias(
#                     name=self.model_name,
#                     alias="staging"
#                 )

#                 print(f"📊 Previous staging version: {model_version.version}")

#                 staging_run_id = model_version.run_id
#                 run = client.get_run(staging_run_id)

#                 print("📊 Staging run metrics:", run.data.metrics)

#                 staging_score = run.data.metrics.get(
#                     "eval_balanced_accuracy",
#                     run.data.metrics.get("balanced_accuracy", 0.0)
#                 )

#             except Exception:
#                 print("⚠️ No staging model found yet (first run)")

#             print(f"📊 Current staging score: {staging_score}")


#             # ---------------- PROMOTION ----------------
#             current_score = metrics["eval_balanced_accuracy"]

#             if current_score > staging_score:

#                 print("🚀 Promoting new model to staging")

#                 result = mlflow.sklearn.log_model(
#                     sk_model=model,
#                     name="model",
#                     registered_model_name=self.model_name
#                 )

#                 version = result.registered_model_version

#                 client.set_registered_model_alias(
#                     name=self.model_name,
#                     alias="staging",
#                     version=version
#                 )

#                 print(f"✅ Registered & promoted version {version}")

#             else:
#                 print(f"❌ Not promoted ({current_score:.4f} <= {staging_score:.4f})")
# import os
# import pandas as pd
# import numpy as np
# import mlflow
# import mlflow.sklearn
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# from pathlib import Path
# from dotenv import load_dotenv
# from mlflow.tracking import MlflowClient

# from sklearn.metrics import (
#     recall_score, f1_score, accuracy_score,
#     roc_auc_score, confusion_matrix,
#     classification_report, RocCurveDisplay,
#     balanced_accuracy_score, precision_score
# )


# class ModelEvaluator:

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
#         self.eval_dir    = Path("artifacts/evaluation")
#         self.eval_dir.mkdir(parents=True, exist_ok=True)

#         self.model_name = "ASD_BEST_MODEL"

#     def get_best_run(self):

#         client = MlflowClient()
#         exp    = client.get_experiment_by_name("model_training_experiment")

#         runs = client.search_runs(
#             experiment_ids=[exp.experiment_id],
#             filter_string="tags.mlflow.runName = 'best_model_summary'",
#             order_by=["start_time DESC"],
#             max_results=1
#         )

#         if not runs:
#             raise Exception("No best_model_summary run found. Run training first.")

#         best_run_id = runs[0].data.params.get("best_run_id")

#         if not best_run_id or best_run_id == "none":
#             raise Exception("best_run_id is 'none'. Check training output.")

#         print(f"Best training run ID: {best_run_id}")
#         return best_run_id

#     def load_data(self):

#         test_df = pd.read_csv(self.feature_dir / "test_features.csv")
#         X_test  = test_df.drop("label", axis=1).values
#         y_test  = test_df["label"].values

#         print(f"Test size        : {len(y_test)}")
#         print(f"Test distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

#         run_id = self.get_best_run()
#         client = MlflowClient()

#         local_dir  = client.download_artifacts(run_id, "model")
#         model_file = next(
#             (f for f in os.listdir(local_dir) if f.endswith(".pkl")), None
#         )

#         if model_file is None:
#             raise Exception(f"No .pkl file found in artifact dir: {local_dir}")

#         model = joblib.load(os.path.join(local_dir, model_file))
#         print(f"Model loaded: {model_file}")

#         # Retrieve holdout balanced_accuracy from training for gap comparison
#         training_run          = client.get_run(run_id)
#         holdout_accuracy  = training_run.data.metrics.get(
#             "holdout_accuracy", None
#         )

#         return X_test, y_test, model, run_id, holdout_accuracy

#     def evaluate(self):

#         X_test, y_test, model, run_id, holdout_accuracy = self.load_data()
#         client = MlflowClient()

#         with mlflow.start_run(run_name="evaluation_stage") as eval_run:

#             mlflow.log_param("source_run_id", run_id)
#             mlflow.log_param(
#                 "model_type",
#                 model.named_steps["model"].__class__.__name__
#             )

#             y_pred = model.predict(X_test)
#             y_prob = model.predict_proba(X_test)[:, 1]

#             eval_metrics = {
#                 "eval_accuracy":          accuracy_score(y_test, y_pred),
#                 "eval_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
#                 "eval_auc":               roc_auc_score(y_test, y_prob),
#                 "eval_f1":                f1_score(y_test, y_pred,        pos_label=1),
#                 "eval_recall":            recall_score(y_test, y_pred,     pos_label=1),
#                 "eval_precision":         precision_score(y_test, y_pred,  pos_label=1),
#             }

#             # Holdout → Test gap — informational only
#             if holdout_accuracy is not None:
#                 gap = holdout_accuracy - eval_metrics["eval_accuracy"]
#                 eval_metrics["holdout_test_gap"] = gap
#                 print(f"Holdout Accuracy : {holdout_accuracy:.4f}")
#                 print(f"Test Accuracy    : {eval_metrics['eval_accuracy']:.4f}")
#                 print(f"Gap                       : {gap:.4f}  {'[OVERFIT]' if gap > 0.10 else '[OK]'}")

#             for k, v in eval_metrics.items():
#                 mlflow.log_metric(k, v)

#             print(f"\nTest metrics: {eval_metrics}")

#             # ---------------- CONFUSION MATRIX ----------------
#             cm         = confusion_matrix(y_test, y_pred)
#             cm_percent = cm / cm.sum(axis=1, keepdims=True)
#             labels     = np.array([
#                 [f"{cm[i,j]}\n({cm_percent[i,j]*100:.1f}%)"
#                  for j in range(cm.shape[1])]
#                 for i in range(cm.shape[0])
#             ])

#             plt.figure(figsize=(7, 6))
#             sns.set(style="whitegrid", font_scale=1.2)
#             sns.heatmap(
#                 cm, annot=labels, fmt="", cmap="YlGnBu",
#                 xticklabels=["Non-ASD", "ASD"],
#                 yticklabels=["Non-ASD", "ASD"],
#                 linewidths=0.5, linecolor="gray"
#             )
#             plt.title("Confusion Matrix — Test Set", fontsize=16, fontweight="bold")
#             plt.xlabel("Predicted Label", fontsize=12)
#             plt.ylabel("Actual Label",    fontsize=12)
#             plt.tight_layout()

#             cm_path = self.eval_dir / "confusion_matrix.png"
#             plt.savefig(cm_path, dpi=300)
#             plt.close()
#             mlflow.log_artifact(str(cm_path))

#             # ---------------- ROC CURVE ----------------
#             plt.figure()
#             RocCurveDisplay.from_predictions(y_test, y_prob)
#             plt.title("ROC Curve — Test Set")
#             roc_path = self.eval_dir / "roc_curve.png"
#             plt.savefig(roc_path)
#             plt.close()
#             mlflow.log_artifact(str(roc_path))

#             # ---------------- CLASSIFICATION REPORT ----------------
#             report      = classification_report(y_test, y_pred, target_names=["Non-ASD", "ASD"])
#             report_path = self.eval_dir / "classification_report.txt"
#             with open(report_path, "w") as f:
#                 f.write(report)
#             mlflow.log_artifact(str(report_path))
#             print(f"\nClassification Report:\n{report}")

            

#             # ---------------- STAGING COMPARISON ----------------
#             # Promote if test balanced_accuracy beats current staging model
#             staging_score = 0.0

#             try:
#                 model_version = client.get_model_version_by_alias(
#                     name=self.model_name, alias="staging"
#                 )
#                 staging_run   = client.get_run(model_version.run_id)
#                 staging_score = staging_run.data.metrics.get("eval_accuracy", 0.0)
#                 print(f"\nCurrent staging Accuracy : {staging_score:.4f}")
#                 print(f"New model Accuracy       : {eval_metrics['eval_accuracy']:.4f}")

#             except Exception:
#                 print("\nNo staging model found yet (first run).")

#             if eval_metrics["eval_accuracy"] > staging_score:
#                 print("Promoting new model to staging.")

#                 result  = mlflow.sklearn.log_model(
#                     sk_model=model,
#                     name="model",
#                     registered_model_name=self.model_name
#                 )
#                 version = result.registered_model_version

#                 client.set_registered_model_alias(
#                     name=self.model_name,
#                     alias="staging",
#                     version=version
#                 )
#                 print(f"Registered and promoted: version {version}")

#             else:
#                 print(
#                     f"Not promoted: score {eval_metrics['eval_accuracy']:.4f}"
#                     f" <= staging {staging_score:.4f}"
#                 )

#             return eval_metrics
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from pathlib import Path
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from imblearn.pipeline import Pipeline

from sklearn.metrics import (
    recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay,
    balanced_accuracy_score, precision_score
)


class ModelEvaluator:

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
        self.eval_dir    = Path("artifacts/evaluation")
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        

        self.model_name = "ASD_BEST_MODEL"

    def get_best_run(self):

        client = MlflowClient()
        exp    = client.get_experiment_by_name("model_training_experiment")

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.mlflow.runName = 'best_model_summary'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            raise Exception("No best_model_summary run found. Run training first.")

        best_run_id = runs[0].data.params.get("best_run_id")

        if not best_run_id or best_run_id == "none":
            raise Exception("best_run_id is 'none'. Check training output.")

        print(f"Best training run ID: {best_run_id}")
        return best_run_id

    def load_data(self):

        test_df = pd.read_csv(self.feature_dir / "test_features.csv")
        selected_features = joblib.load("artifacts/models/selected_features.pkl")
        X_test = test_df.drop("label", axis=1)
        y_test  = test_df["label"].values

        print(f"Test size        : {len(y_test)}")
        print(f"Test distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

        run_id = self.get_best_run()
        client = MlflowClient()

        local_dir  = client.download_artifacts(run_id, "model")
        model_file = next(
            (f for f in os.listdir(local_dir) if f.endswith(".pkl")), None
        )

        if model_file is None:
            raise Exception(f"No .pkl file found in artifact dir: {local_dir}")

        model = joblib.load(os.path.join(local_dir, model_file))
        print(f"Model loaded: {model_file}")

        training_run     = client.get_run(run_id)
        holdout_accuracy = training_run.data.metrics.get("holdout_accuracy", None)

        return X_test, y_test, model, run_id, holdout_accuracy

    # -------------------------
    # Helper — extract class-1
    # SHAP array regardless of
    # SHAP version return shape
    # -------------------------
    def extract_shap_class1(self, shap_values):
        if isinstance(shap_values, list):
            # Older SHAP → list [array(n,f), array(n,f)]
            return shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Newer SHAP → 3D array (n_samples, n_features, n_classes)
            return shap_values[:, :, 1]
        else:
            # Single output / already 2D
            return shap_values

    def evaluate(self):

        X_test, y_test, model, run_id, holdout_accuracy = self.load_data()
        client = MlflowClient()

        with mlflow.start_run(run_name="evaluation_stage") as eval_run:

            mlflow.log_param("source_run_id", run_id)
            # ---------------- DRIFT DETECTION ----------------
            print("\n--- Drift Detection ---")

            train_df = pd.read_csv(self.feature_dir / "train_features.csv")
            test_df  = pd.read_csv(self.feature_dir / "test_features.csv")

            train_features = train_df.drop("label", axis=1)
            test_features  = test_df.drop("label", axis=1)

            # ---------- PCA DRIFT ----------
            pipeline_without_model = Pipeline(model.steps[:-1])

            X_train_transformed = pipeline_without_model.transform(train_features.values)
            X_test_transformed  = pipeline_without_model.transform(test_features.values)

            pca_drift = np.abs(
                X_train_transformed.mean(axis=0) - X_test_transformed.mean(axis=0)
            ).mean()

            print(f"PCA Drift Score  : {pca_drift:.4f}")
            mlflow.log_metric("pca_drift", pca_drift)

            # ---------- DRIFT DECISION ----------
            DRIFT_THRESHOLD = 0.15
            is_drift = False

            if pca_drift > DRIFT_THRESHOLD:
                print("⚠️ Drift detected!")
                is_drift = True
                mlflow.log_param("drift_status", "high")
            else:
                print("✅ No significant drift")
                mlflow.log_param("drift_status", "low")
            mlflow.log_param(
                "model_type",
                model.named_steps["model"].__class__.__name__
            )

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            eval_metrics = {
                "eval_accuracy":          accuracy_score(y_test, y_pred),
                "eval_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "eval_auc":               roc_auc_score(y_test, y_prob),
                "eval_f1":                f1_score(y_test, y_pred,        pos_label=1),
                "eval_recall":            recall_score(y_test, y_pred,     pos_label=1),
                "eval_precision":         precision_score(y_test, y_pred,  pos_label=1),
            }

            # Holdout → Test gap
            is_overfit = False
            OVERFIT_THRESHOLD = 0.10

            if holdout_accuracy is not None:
                gap = holdout_accuracy - eval_metrics["eval_accuracy"]
                eval_metrics["holdout_test_gap"] = gap

                print(f"Holdout Accuracy : {holdout_accuracy:.4f}")
                print(f"Test Accuracy    : {eval_metrics['eval_accuracy']:.4f}")
                
                if gap > OVERFIT_THRESHOLD:
                    print(f"Gap              : {gap:.4f}  [OVERFIT ❌]")
                    is_overfit = True
                else:
                    print(f"Gap              : {gap:.4f}  [OK ✅]")

                # 🔥 Log status
                for k, v in eval_metrics.items():
                    mlflow.log_metric(k, v)
                print(f"\nTest metrics: {eval_metrics}")
                
                
                mlflow.log_param("overfit_status", "overfit" if is_overfit else "ok")

            # ---------------- CONFUSION MATRIX ----------------
            cm         = confusion_matrix(y_test, y_pred)
            cm_percent = cm / cm.sum(axis=1, keepdims=True)
            labels     = np.array([
                [f"{cm[i,j]}\n({cm_percent[i,j]*100:.1f}%)"
                 for j in range(cm.shape[1])]
                for i in range(cm.shape[0])
            ])

            plt.figure(figsize=(7, 6))
            sns.set(style="whitegrid", font_scale=1.2)
            sns.heatmap(
                cm, annot=labels, fmt="", cmap="YlGnBu",
                xticklabels=["Non-ASD", "ASD"],
                yticklabels=["Non-ASD", "ASD"],
                linewidths=0.5, linecolor="gray"
            )
            plt.title("Confusion Matrix — Test Set", fontsize=16, fontweight="bold")
            plt.xlabel("Predicted Label", fontsize=12)
            plt.ylabel("Actual Label",    fontsize=12)
            plt.tight_layout()

            cm_path = self.eval_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=300)
            plt.close()
            mlflow.log_artifact(str(cm_path))

            # ---------------- ROC CURVE ----------------
            plt.figure()
            RocCurveDisplay.from_predictions(y_test, y_prob)
            plt.title("ROC Curve — Test Set")
            roc_path = self.eval_dir / "roc_curve.png"
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(str(roc_path))

            # ---------------- CLASSIFICATION REPORT ----------------
            report      = classification_report(y_test, y_pred, target_names=["Non-ASD", "ASD"])
            report_path = self.eval_dir / "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(str(report_path))
            print(f"\nClassification Report:\n{report}")

            # ---------------- SHAP EXPLANATION ----------------
            best_model_step = model.named_steps["model"]
            model_type      = best_model_step.__class__.__name__

            try:
                print("\n--- SHAP Analysis Started ---")

                # =====================================================
                # 1. TRANSFORM TEST DATA THROUGH FULL PREPROCESS PIPELINE
                # =====================================================
                pipeline_without_model = Pipeline(model.steps[:-1])
                X_transformed = pipeline_without_model.transform(X_test)

                # PCA step
                pca_step = model.named_steps["pca"]
                n_components = pca_step.n_components_
                feature_names = [f"PC{i+1}" for i in range(n_components)]

                # =====================================================
                # 2. COMPUTE SHAP VALUES ON PCA COMPONENTS
                # =====================================================
                if model_type == "RandomForestClassifier":
                    explainer = shap.TreeExplainer(best_model_step)
                    shap_values = explainer.shap_values(X_transformed)

                else:
                    background = shap.sample(
                        X_transformed,
                        min(50, len(X_transformed)),
                        random_state=42
                    )

                    explainer = shap.KernelExplainer(
                        best_model_step.predict_proba,
                        background
                    )

                    shap_values = explainer.shap_values(
                        X_transformed,
                        nsamples=100
                    )

                shap_asd = self.extract_shap_class1(shap_values)

                print("SHAP PCA shape :", shap_asd.shape)

                # =====================================================
                # 3. PCA COMPONENT SHAP BAR PLOT
                # =====================================================
                plt.figure(figsize=(10, 6))

                shap.summary_plot(
                    shap_asd,
                    X_transformed,
                    feature_names=feature_names,
                    plot_type="bar",
                    max_display=15,
                    show=False
                )

                plt.title(f"SHAP Feature Importance (PCA) — {model_type}")
                plt.tight_layout()

                shap_bar_path = self.eval_dir / "shap_summary_bar.png"
                plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
                plt.close()

                mlflow.log_artifact(str(shap_bar_path))

                # =====================================================
                # 4. PCA COMPONENT SHAP DOT PLOT
                # =====================================================
                plt.figure(figsize=(10, 6))

                shap.summary_plot(
                    shap_asd,
                    X_transformed,
                    feature_names=feature_names,
                    plot_type="dot",
                    max_display=15,
                    show=False
                )

                plt.title(f"SHAP Summary Dot Plot (PCA) — {model_type}")
                plt.tight_layout()

                shap_dot_path = self.eval_dir / "shap_summary_dot.png"
                plt.savefig(shap_dot_path, dpi=300, bbox_inches="tight")
                plt.close()

                mlflow.log_artifact(str(shap_dot_path))

                # =====================================================
                # 5. SAVE RAW SHAP VALUES
                # =====================================================
                shap_array_path = self.eval_dir / "shap_values.npy"
                np.save(str(shap_array_path), shap_asd)
                mlflow.log_artifact(str(shap_array_path))

                mean_shap = np.abs(shap_asd).mean(axis=0)
                top_component = feature_names[np.argmax(mean_shap)]

                mlflow.log_param("top_shap_component", top_component)

                print("Top PCA component :", top_component)

                # =====================================================
                # 6. BACK PROJECT SHAP TO ORIGINAL FEATURES
                # =====================================================
                pca_components = pca_step.components_        # (pc, feature)
                shap_original = np.dot(shap_asd, pca_components)

                # =====================================================
                # 7. GET TRUE FEATURE NAMES AFTER VARIANCE THRESHOLD
                # =====================================================
                selected_features = joblib.load(
                    "artifacts/models/selected_features.pkl"
                )

                # variance_mask = model.named_steps["variance"].get_support()

                original_features = np.array(selected_features)

                # =====================================================
                # 8. GET DATA BEFORE PCA (scaled + variance only)
                # =====================================================
                pre_pca_pipeline = Pipeline([
                    ("scaler", model.named_steps["scaler"]),
                    ("variance", model.named_steps["variance"])
                ])

                X_before_pca = pre_pca_pipeline.transform(X_test)

                print("Original SHAP shape :", shap_original.shape)
                print("Original Data shape :", X_before_pca.shape)

                # =====================================================
                # 9. ORIGINAL FEATURE SHAP DOT PLOT
                # =====================================================
                plt.figure(figsize=(10, 6))

                shap.summary_plot(
                    shap_original,
                    X_before_pca,
                    feature_names=original_features,
                    plot_type="dot",
                    max_display=15,
                    show=False
                )

                plt.title(f"SHAP Summary Dot Plot (Original Features)")
                plt.tight_layout()

                orig_dot_path = self.eval_dir / "shap_original_summary_dot.png"
                plt.savefig(orig_dot_path, dpi=300, bbox_inches="tight")
                plt.close()

                mlflow.log_artifact(str(orig_dot_path))

                # =====================================================
                # 10. ORIGINAL FEATURE BAR PLOT
                # =====================================================
                plt.figure(figsize=(10, 6))

                shap.summary_plot(
                    shap_original,
                    X_before_pca,
                    feature_names=original_features,
                    plot_type="bar",
                    max_display=15,
                    show=False
                )

                plt.title("SHAP Feature Importance (Original Features)")
                plt.tight_layout()

                orig_bar_path = self.eval_dir / "shap_original_summary_bar.png"
                plt.savefig(orig_bar_path, dpi=300, bbox_inches="tight")
                plt.close()

                mlflow.log_artifact(str(orig_bar_path))

                # =====================================================
                # 11. FEATURE IMPORTANCE TABLE
                # =====================================================
                original_importance = np.abs(shap_original).mean(axis=0)

                original_importance_df = pd.DataFrame({
                    "feature": original_features,
                    "importance": original_importance
                }).sort_values("importance", ascending=False)

                print("\nTop 15 Original Features")
                print(original_importance_df.head(15).to_string(index=False))

                orig_csv = self.eval_dir / "shap_original_feature_importance.csv"
                original_importance_df.to_csv(orig_csv, index=False)
                mlflow.log_artifact(str(orig_csv))

                # =====================================================
                # 12. GROUPED FEATURE IMPORTANCE
                # =====================================================
                feature_groups = {
                    "LBP": ["lbp"],
                    "GLCM": ["glcm"],
                    "GFCC": ["gfcc"]
                }

                group_importance = {}

                for group, keywords in feature_groups.items():

                    mask = original_importance_df["feature"].str.lower().apply(
                        lambda x: any(k in x for k in keywords)
                    )

                    group_importance[group] = original_importance_df.loc[
                        mask, "importance"
                    ].sum()

                group_df = pd.DataFrame(
                    list(group_importance.items()),
                    columns=["group", "importance"]
                ).sort_values("importance", ascending=False)

                print("\nGrouped Importance")
                print(group_df.to_string(index=False))

                group_csv = self.eval_dir / "shap_group_importance.csv"
                group_df.to_csv(group_csv, index=False)
                mlflow.log_artifact(str(group_csv))

                # =====================================================
                # 13. GROUP PLOT
                # =====================================================
                plt.figure(figsize=(6, 4))

                plt.barh(
                    group_df["group"][::-1],
                    group_df["importance"][::-1]
                )

                plt.title("Feature Group Contribution")
                plt.tight_layout()

                group_plot = self.eval_dir / "shap_group_importance.png"
                plt.savefig(group_plot, dpi=300, bbox_inches="tight")
                plt.close()

                mlflow.log_artifact(str(group_plot))

                top_group = group_df.iloc[0]["group"]
                mlflow.log_param("top_feature_group", top_group)

                print("Top Group :", top_group)
                print("SHAP Analysis Completed Successfully")

            except Exception as e:
                print(f"\nSHAP computation failed: {e}")
                mlflow.log_param("shap_status", f"failed: {str(e)}")

            # ---------------- STAGING COMPARISON ----------------
            staging_score = 0.0

            try:
                model_version = client.get_model_version_by_alias(
                    name=self.model_name, alias="staging"
                )
                staging_run   = client.get_run(model_version.run_id)
                staging_score = staging_run.data.metrics.get("eval_accuracy", 0.0)

                print(f"\nCurrent staging Accuracy : {staging_score:.4f}")
                print(f"New model Accuracy       : {eval_metrics['eval_accuracy']:.4f}")

            except Exception:
                print("\nNo staging model found yet (first run).")


            # 🔥 FINAL DECISION LOGIC
            if (not is_overfit) and (eval_metrics["eval_accuracy"] > staging_score):

                print("✅ Promoting new model to staging (better + no overfit)")

                result = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    registered_model_name=self.model_name
                )
                version = result.registered_model_version

                client.set_registered_model_alias(
                    name=self.model_name,
                    alias="staging",
                    version=version
                )

                print(f"Registered and promoted: version {version}")

            else:
                print("\n❌ Model NOT promoted")

                if is_overfit:
                    print("Reason: Overfitting detected (holdout → test gap too high)")

                if eval_metrics["eval_accuracy"] <= staging_score:
                    print(
                        f"Reason: Not better than staging "
                        f"({eval_metrics['eval_accuracy']:.4f} <= {staging_score:.4f})"
                    )