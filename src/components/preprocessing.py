import os
import random
import shutil
from dataclasses import dataclass
import sys
import yaml

import nibabel as nib
import numpy as np
import cv2

from src.utils.logger import logger
from src.utils.exception import CustomException


# =========================
# 📌 Config Entity
# =========================
@dataclass
class PreprocessingConfig:
    root_dir: str
    input_dir: str
    train_dir: str
    test_dir: str
    test_size: float
    random_state: int
    num_slices: int


# =========================
# 📌 Config Manager
# =========================
class ConfigurationManager:
    def __init__(self, config_filepath="config.yaml"):
        try:
            with open(config_filepath, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def get_preprocessing_config(self) -> PreprocessingConfig:
        try:
            config = self.config["preprocessing"]

            return PreprocessingConfig(
                root_dir=config["root_dir"],
                input_dir=config["input_dir"],
                train_dir=config["train_dir"],
                test_dir=config["test_dir"],
                test_size=config["test_size"],
                random_state=config["random_state"],
                num_slices=config["num_slices"]
            )
        except Exception as e:
            raise CustomException(e, sys)


# =========================
# 📌 Preprocessing Class
# =========================
class DataPreprocessing:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    # -------------------------
    # Reorient to RAS
    # -------------------------
    def reorient_to_ras(self, img):
        try:
            return nib.as_closest_canonical(img)
        except Exception as e:
            raise CustomException(e, sys)

    # # -------------------------
    # # Remove Tilt (Resampling)
    # # -------------------------
    # def resample_to_identity(self, img):
    #     try:
    #         target_affine = np.eye(4)

    #         resampled_img = resample_img(
    #             img,
    #             target_affine=target_affine,
    #             interpolation="continuous"
    #         )

    #         return resampled_img

    #     except Exception as e:
    #         raise CustomException(e, sys)

    # -------------------------
    # Train-Test Split
    # -------------------------
    def train_test_split(self):
        try:
            logger.info("Starting train-test split...")

            classes = os.listdir(self.config.input_dir)

            for cls in classes:
                class_path = os.path.join(self.config.input_dir, cls)
                files = os.listdir(class_path)

                random.seed(self.config.random_state)
                random.shuffle(files)

                split_idx = int(len(files) * (1 - self.config.test_size))

                train_files = files[:split_idx]
                test_files = files[split_idx:]

                train_cls_path = os.path.join(self.config.train_dir, cls)
                test_cls_path = os.path.join(self.config.test_dir, cls)

                os.makedirs(train_cls_path, exist_ok=True)
                os.makedirs(test_cls_path, exist_ok=True)

                for file in train_files:
                    shutil.copy(os.path.join(class_path, file), train_cls_path)

                for file in test_files:
                    shutil.copy(os.path.join(class_path, file), test_cls_path)

            logger.info("Train-test split completed!")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------
    # Normalize Slice
    # -------------------------
    def normalize_slice(self, slice_2d):
        slice_2d = (slice_2d - np.min(slice_2d)) / (
            np.max(slice_2d) - np.min(slice_2d) + 1e-8
        )
        return (slice_2d * 255).astype(np.uint8)

    # -------------------------
    # Train Slice Extraction (11 slices)
    # -------------------------
    def extract_train_slices(self):
        try:
            logger.info("Extracting TRAIN slices...")

            for cls in os.listdir(self.config.train_dir):
                class_path = os.path.join(self.config.train_dir, cls)

                for file in os.listdir(class_path):

                    file_path = os.path.join(class_path, file)

                    if not file.lower().endswith((".nii", ".nii.gz")):
                        continue

                    try:
                        img = nib.load(file_path)
                        img = self.reorient_to_ras(img)

                        volume = np.array(img.get_fdata())
                        del img  # release file lock

                        mid = volume.shape[0] // 2
                        subject_name = os.path.splitext(file)[0]

                        for i in range(-(self.config.num_slices // 2),
                                    (self.config.num_slices // 2) + 1):

                            slice_idx = mid + i

                            if slice_idx < 0 or slice_idx >= volume.shape[0]:
                                continue

                            slice_2d = volume[slice_idx, :, :]
                            slice_2d = np.rot90(slice_2d)
                            slice_2d = self.normalize_slice(slice_2d)


                            filename = f"{subject_name}_slice_{slice_idx}.png"
                            cv2.imwrite(os.path.join(class_path, filename), slice_2d)

                        # 🔥 FORCE DELETE
                        os.remove(file_path)

                    except Exception as e:
                        logger.warning(f"Skipping file: {file_path} | Error: {e}")
                        continue
                logger.info("Train slices extracted!")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------
    # Test Slice Extraction (1 slice)
    # -------------------------
    def extract_test_slice(self):
        try:
            logger.info("Extracting TEST slice...")

            for cls in os.listdir(self.config.test_dir):
                class_path = os.path.join(self.config.test_dir, cls)

                for file in os.listdir(class_path):

                    file_path = os.path.join(class_path, file)

                    if not file.lower().endswith((".nii", ".nii.gz")):
                        continue

                    try:
                        img = nib.load(file_path)
                        img = self.reorient_to_ras(img)

                        volume = np.array(img.get_fdata())
                        del img

                        mid = volume.shape[0] // 2

                        slice_indices = [mid - 2, mid, mid + 2]

                        for i, idx in enumerate(slice_indices):

                            if idx < 0 or idx >= volume.shape[0]:
                                continue

                            slice_2d = volume[idx, :, :]
                            slice_2d = np.rot90(slice_2d)
                            slice_2d = self.normalize_slice(slice_2d)

                            subject_name = os.path.splitext(file)[0]
                            filename = f"{subject_name}_slice_{i}.png"

                            cv2.imwrite(os.path.join(class_path, filename), slice_2d)
                        

                        subject_name = os.path.splitext(file)[0]
                        filename = f"{subject_name}_mid.png"

                        cv2.imwrite(os.path.join(class_path, filename), slice_2d)

                        os.remove(file_path)

                    except Exception as e:
                        logger.warning(f"Skipping file: {file_path} | Error: {e}")
                        continue

            logger.info("Test slice extracted!")

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------
    # Run Pipeline
    # -------------------------
    def run(self):
        try:
            logger.info("Starting preprocessing pipeline...")

            os.makedirs(self.config.root_dir, exist_ok=True)

            self.train_test_split()
            self.extract_train_slices()
            self.extract_test_slice()

            logger.info("Preprocessing completed successfully!")

        except Exception as e:
            raise CustomException(e, sys)