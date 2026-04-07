import os
import shutil
from dataclasses import dataclass

from src.utils.logger import logger
from src.utils.exception import CustomException
import sys
import yaml


# =========================
# 📌 Config Entity
# =========================
@dataclass
class DataIngestionConfig:
    root_dir: str
    source_dir: str
    local_data_file: str


# =========================
# 📌 Configuration Manager
# =========================
class ConfigurationManager:
    def __init__(self, config_filepath="config.yaml"):
        try:
            with open(config_filepath, "r") as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config["data_ingestion"]

            return DataIngestionConfig(
                root_dir=config["root_dir"],
                source_dir=config["source_dir"],
                local_data_file=config["local_data_file"],
            )
        except Exception as e:
            raise CustomException(e, sys)


# =========================
# 📌 Data Ingestion Class
# =========================
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            logger.info("Starting data ingestion...")

            # Validate source path FIRST
            if not os.path.exists(self.config.source_dir):
                raise CustomException(
                    f"Source path does not exist: {self.config.source_dir}",
                    sys
                )

            logger.info(f"Source path verified: {self.config.source_dir}")

            # Create root directory
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Created root dir: {self.config.root_dir}")

            # Remove old data if exists
            if os.path.exists(self.config.local_data_file):
                logger.info("Existing data found. Removing old data...")
                shutil.rmtree(self.config.local_data_file)

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)

            logger.info(f"Copying data from {self.config.source_dir}")
            logger.info(f"Destination: {self.config.local_data_file}")

            # Copy dataset
            shutil.copytree(
                self.config.source_dir,
                self.config.local_data_file
            )

            # Verify copy success
            if not os.path.exists(self.config.local_data_file):
                raise CustomException(
                    "Data copy failed. Destination not created.",
                    sys
                )

            logger.info("Data ingestion completed successfully!")

        except Exception as e:
            raise CustomException(e, sys)