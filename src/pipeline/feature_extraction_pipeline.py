from joblib import Logger

from src.components.feature_extraction import FeatureExtractor
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys  
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("feature_extraction_experiment")


class FeatureExtractionPipeline:
    def main(self):
        try:
            logger.info("==== Feature Extraction Pipeline Started ====")

            obj = FeatureExtractor()

            with mlflow.start_run(run_name="feature_extraction_run"):

                train_path = "artifacts/preprocessing/train"
                test_path = "artifacts/preprocessing/test"
                os.makedirs("artifacts/features", exist_ok=True)

                # PARAMS
                mlflow.log_param("image_size", 224)
                # mlflow.log_param("lbp_points", 8)
                # mlflow.log_param("lbp_radius", 1)
                # mlflow.log_param("glcm_features", 7)
                # mlflow.log_param("lbp_features", 256)
                # mlflow.log_param("gfcc_features",6)
                mlflow.log_param("feature_extractor", "DenseNet121 RadImageNet")
                # mlflow.log_param("pretrained", True)

                # ================= TRAIN =================
                logger.info("Extracting features from training data...")
                train_df = obj.process_dataset(train_path)
                train_df.to_csv("artifacts/features/train_features.csv", index=False)

                # ================= TEST =================
                logger.info("Extracting features from testing data...")
                test_df = obj.process_dataset(test_path)    
                test_df.to_csv("artifacts/features/test_features.csv", index=False)

                # METRICS (AFTER BOTH READY)
                mlflow.log_metric("num_train_samples", len(train_df))
                mlflow.log_metric("num_test_samples", len(test_df))
                mlflow.log_metric("num_features", train_df.shape[1] - 1)

                # ARTIFACTS
                mlflow.log_artifact("artifacts/features/train_features.csv")
                mlflow.log_artifact("artifacts/features/test_features.csv")

                # FEATURE NAMES
                feature_names = list(train_df.columns)
                with open("artifacts/features/feature_names.txt", "w") as f:
                    f.write("\n".join(feature_names))

                mlflow.log_artifact("artifacts/features/feature_names.txt")

                logger.info("==== Feature Extraction Pipeline Completed ====")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = FeatureExtractionPipeline()
    obj.main()