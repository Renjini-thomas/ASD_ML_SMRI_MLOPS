from src.components.data_ingestion import DataIngestion, ConfigurationManager
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info("==== Data Ingestion Pipeline Started ====")

            # Load configuration
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()

            # Initialize and run ingestion
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.initiate_data_ingestion()

            logger.info("==== Data Ingestion Pipeline Completed ====")

        except Exception as e:
            raise CustomException(e, sys)
if __name__ == "__main__":
    obj = DataIngestionTrainingPipeline()
    obj.main()