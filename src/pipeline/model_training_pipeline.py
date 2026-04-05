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
            
            trainer.run()
            logger.info("===== Model Training Pipeline Completed =====")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = ModelTrainingPipeline()
    obj.main()