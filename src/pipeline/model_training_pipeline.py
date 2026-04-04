import sys
import mlflow

from src.components.model_training import ModelTrainer
from src.utils.logger import logger
from src.utils.exception import CustomException


class ModelTrainingPipeline:

    def main(self):

        try:
            logger.info("===== Model Training Pipeline Started =====")

            trainer = ModelTrainer()

            best_run_id, best_model_name, best_score = trainer.run()

            # ✅ Log global best (for next stage)
            with mlflow.start_run(run_name="best_model_summary"):

                mlflow.log_param("best_run_id", best_run_id)
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_balanced_accuracy", best_score)

            logger.info(f"🏆 Best Model: {best_model_name}")
            logger.info(f"Run ID: {best_run_id}")
            logger.info(f"Score: {best_score}")

            logger.info("===== Model Training Pipeline Completed =====")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = ModelTrainingPipeline()
    obj.main()