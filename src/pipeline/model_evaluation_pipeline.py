import sys

from src.components.model_evaluation import ModelEvaluator
from src.utils.logger import logger
from src.utils.exception import CustomException


class ModelEvaluationPipeline:

    def main(self):

        try:
            logger.info("===== Evaluation Started =====")

            evaluator = ModelEvaluator()
            evaluator.evaluate()

            logger.info("===== Evaluation Completed =====")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = ModelEvaluationPipeline()
    obj.main()