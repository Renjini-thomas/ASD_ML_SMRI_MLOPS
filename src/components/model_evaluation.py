
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    balanced_accuracy_score,
    precision_score
)


class ModelEvaluator:

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
        self.eval_dir = Path("artifacts/evaluation")
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = "ASD_BEST_MODEL"

    # ---------------- GET BEST RUN ----------------
    def get_best_run(self):

        client = MlflowClient()
        exp = client.get_experiment_by_name("model_training_experiment")

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.mlflow.runName = 'best_model_summary'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if len(runs) == 0:
            raise Exception("❌ No best_model_summary run found")

        summary_run = runs[0]

        best_run_id = summary_run.data.params.get("best_run_id")

        if best_run_id is None:
            raise Exception("❌ best_run_id not found in summary run")

        print("⭐ Best Training Run ID:", best_run_id)

        return best_run_id

    # ---------------- LOAD DATA + MODEL ----------------
    def load_data(self):

        test_df = pd.read_csv(self.feature_dir / "test_features.csv")

        X_test = test_df.drop("label", axis=1)
        y_test = test_df["label"]

        run_id = self.get_best_run()

        model_uri = f"runs:/{run_id}/model"
        print(f"📦 Loading model from: {model_uri}")

        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            raise Exception(f"❌ Model not found at {model_uri}")

        return X_test, y_test, model, run_id

    # ---------------- EVALUATION ----------------
    def evaluate(self):

        X_test, y_test, model, run_id = self.load_data()
        client = MlflowClient()

        with mlflow.start_run(run_name="evaluation_stage"):

            # ---------------- BASIC INFO ----------------
            mlflow.log_param("source_run_id", run_id)
            mlflow.log_param("model_type", model.named_steps["model"].__class__.__name__)

            # ---------------- PREDICTIONS ----------------
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # ---------------- METRICS ----------------
            metrics = {
                "eval_recall": recall_score(y_test, y_pred),
                "eval_f1": f1_score(y_test, y_pred),
                "eval_accuracy": accuracy_score(y_test, y_pred),
                "eval_auc": roc_auc_score(y_test, y_prob),
                "eval_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "eval_precision": precision_score(y_test, y_pred),
            }

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # ---------------- CONFUSION MATRIX ----------------
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title("Confusion Matrix")

            cm_path = self.eval_dir / "confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()

            mlflow.log_artifact(str(cm_path))

            # ---------------- ROC CURVE ----------------
            plt.figure()
            RocCurveDisplay.from_predictions(y_test, y_prob)

            roc_path = self.eval_dir / "roc_curve.png"
            plt.savefig(roc_path)
            plt.close()

            mlflow.log_artifact(str(roc_path))

            # ---------------- CLASSIFICATION REPORT ----------------
            report = classification_report(y_test, y_pred)

            report_path = self.eval_dir / "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)

            mlflow.log_artifact(str(report_path))

            print("✅ Evaluation metrics & artifacts logged")

            # ---------------- STAGING MODEL CHECK ----------------
            staging_score = 0.0

            try:
                staging_model = client.get_model_version_by_alias(
                    name=self.model_name,
                    alias="staging"
                )

                staging_score = float(
                    staging_model.tags.get("eval_balanced_accuracy", 0)
                )

                print(f"📊 Current staging score: {staging_score}")

            except Exception:
                print("⚠️ No staging model found (first run)")

            # ---------------- PROMOTION LOGIC ----------------
            current_score = metrics["eval_balanced_accuracy"]

            if current_score > staging_score:

                print("🚀 Promoting new model to staging")

                result = mlflow.register_model(
                    model_uri=f"runs:/{run_id}/model",
                    name=self.model_name
                )

                # Tag performance
                client.set_model_version_tag(
                    name=self.model_name,
                    version=result.version,
                    key="eval_balanced_accuracy",
                    value=str(round(current_score, 6))
                )

                # Set alias
                client.set_registered_model_alias(
                    name=self.model_name,
                    alias="staging",
                    version=result.version
                )

                print(f"✅ Promoted (score={current_score:.4f})")

            else:
                print(f"❌ Not promoted ({current_score:.4f} <= {staging_score:.4f})")

