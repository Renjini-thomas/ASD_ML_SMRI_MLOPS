from src.components.preprocessing import DataPreprocessing, ConfigurationManager
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys


class DataPreprocessingPipeline:
    def main(self):
        try:
            logger.info("==== Preprocessing Pipeline Started ====")

            config = ConfigurationManager().get_preprocessing_config()
            obj = DataPreprocessing(config)
            obj.run()

            logger.info("==== Preprocessing Pipeline Completed ====")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataPreprocessingPipeline()
    obj.main()