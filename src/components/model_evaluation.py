
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
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

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
        X_test  = test_df.drop("label", axis=1).values
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

        # Retrieve holdout balanced_accuracy from training for gap comparison
        training_run          = client.get_run(run_id)
        holdout_accuracy  = training_run.data.metrics.get(
            "holdout_accuracy", None
        )

        return X_test, y_test, model, run_id, holdout_accuracy

    def evaluate(self):

        X_test, y_test, model, run_id, holdout_accuracy = self.load_data()
        client = MlflowClient()

        with mlflow.start_run(run_name="evaluation_stage") as eval_run:

            mlflow.log_param("source_run_id", run_id)
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

            # Holdout → Test gap — informational only
            if holdout_accuracy is not None:
                gap = holdout_accuracy - eval_metrics["eval_accuracy"]
                eval_metrics["holdout_test_gap"] = gap
                print(f"Holdout Accuracy : {holdout_accuracy:.4f}")
                print(f"Test Accuracy    : {eval_metrics['eval_accuracy']:.4f}")
                print(f"Gap                       : {gap:.4f}  {'[OVERFIT]' if gap > 0.10 else '[OK]'}")

            for k, v in eval_metrics.items():
                mlflow.log_metric(k, v)

            print(f"\nTest metrics: {eval_metrics}")

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

            # ---------------- STAGING COMPARISON ----------------
            # Promote if test balanced_accuracy beats current staging model
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

            if eval_metrics["eval_accuracy"] > staging_score:
                print("Promoting new model to staging.")

                result  = mlflow.sklearn.log_model(
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
                print(
                    f"Not promoted: score {eval_metrics['eval_accuracy']:.4f}"
                    f" <= staging {staging_score:.4f}"
                )

            return eval_metrics