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

                        if 0 <= mid < volume.shape[0]:

                            slice_2d = volume[mid, :, :]
                            slice_2d = np.rot90(slice_2d)
                            slice_2d = self.normalize_slice(slice_2d)

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
# import os
# import random
# import shutil
# from dataclasses import dataclass
# import sys
# import yaml

# import nibabel as nib
# import numpy as np
# import cv2


# # =========================
# # 📌 Config Entity
# # =========================
# @dataclass
# class PreprocessingConfig:
#     root_dir: str
#     input_dir: str
#     train_dir: str
#     test_dir: str
#     test_size: float
#     random_state: int
#     num_slices: int


# # =========================
# # 📌 Config Manager
# # =========================
# class ConfigurationManager:
#     def __init__(self, config_filepath="config.yaml"):
#         with open(config_filepath, "r") as f:
#             self.config = yaml.safe_load(f)

#     def get_preprocessing_config(self) -> PreprocessingConfig:
#         config = self.config["preprocessing"]

#         return PreprocessingConfig(
#             root_dir=config["root_dir"],
#             input_dir=config["input_dir"],
#             train_dir=config["train_dir"],
#             test_dir=config["test_dir"],
#             test_size=config["test_size"],
#             random_state=config["random_state"],
#             num_slices=config["num_slices"]
#         )


# # =========================
# # 📌 Preprocessing Class
# # =========================
# class DataPreprocessing:
#     def __init__(self, config: PreprocessingConfig):
#         self.config = config

#     # -------------------------
#     # Reorient to RAS
#     # -------------------------
#     def reorient_to_ras(self, img):
#         return nib.as_closest_canonical(img)

#     # -------------------------
#     # Train-Test Split
#     # -------------------------
#     def train_test_split(self):
#         print("Splitting dataset...")

#         for cls in os.listdir(self.config.input_dir):
#             class_path = os.path.join(self.config.input_dir, cls)
#             files = os.listdir(class_path)

#             random.seed(self.config.random_state)
#             random.shuffle(files)

#             split_idx = int(len(files) * (1 - self.config.test_size))

#             train_files = files[:split_idx]
#             test_files = files[split_idx:]

#             train_cls_path = os.path.join(self.config.train_dir, cls)
#             test_cls_path = os.path.join(self.config.test_dir, cls)

#             os.makedirs(train_cls_path, exist_ok=True)
#             os.makedirs(test_cls_path, exist_ok=True)

#             for f in train_files:
#                 shutil.copy(os.path.join(class_path, f), train_cls_path)

#             for f in test_files:
#                 shutil.copy(os.path.join(class_path, f), test_cls_path)

#         print("Train-test split done!")

#     # -------------------------
#     # Normalize Slice
#     # -------------------------
#     def normalize_slice(self, slice_2d):
#         slice_2d = (slice_2d - np.min(slice_2d)) / (
#             np.max(slice_2d) - np.min(slice_2d) + 1e-8
#         )
#         return (slice_2d * 255).astype(np.uint8)

#     # -------------------------
#     # Train Slice Extraction
#     # -------------------------
#     def extract_train_slices(self):
#         print("Extracting train slices...")

#         for cls in os.listdir(self.config.train_dir):
#             class_path = os.path.join(self.config.train_dir, cls)

#             for file in os.listdir(class_path):

#                 if not file.endswith((".nii", ".nii.gz")):
#                     continue

#                 file_path = os.path.join(class_path, file)

#                 try:
#                     img = nib.load(file_path)
#                     img = self.reorient_to_ras(img)

#                     volume = np.array(img.get_fdata())
#                     del img

#                     mid = volume.shape[0] // 2
#                     subject_name = os.path.splitext(file)[0]

#                     for i in range(-(self.config.num_slices // 2),
#                                    (self.config.num_slices // 2) + 1):

#                         idx = mid + i
#                         if idx < 0 or idx >= volume.shape[0]:
#                             continue

#                         slice_2d = volume[idx, :, :]
#                         slice_2d = np.rot90(slice_2d)
#                         slice_2d = self.normalize_slice(slice_2d)

#                         filename = f"{subject_name}_slice_{idx}.png"
#                         cv2.imwrite(os.path.join(class_path, filename), slice_2d)

#                     os.remove(file_path)

#                 except:
#                     continue

#         print("Train slices done!")

#     # -------------------------
#     # Test Slice Extraction
#     # -------------------------
#     def extract_test_slice(self):
#         print("Extracting test slice...")

#         for cls in os.listdir(self.config.test_dir):
#             class_path = os.path.join(self.config.test_dir, cls)

#             for file in os.listdir(class_path):

#                 if not file.endswith((".nii", ".nii.gz")):
#                     continue

#                 file_path = os.path.join(class_path, file)

#                 try:
#                     img = nib.load(file_path)
#                     img = self.reorient_to_ras(img)

#                     volume = np.array(img.get_fdata())
#                     del img

#                     mid = volume.shape[0] // 2

#                     slice_2d = volume[mid, :, :]
#                     slice_2d = np.rot90(slice_2d)
#                     slice_2d = self.normalize_slice(slice_2d)

#                     subject_name = os.path.splitext(file)[0]
#                     filename = f"{subject_name}_mid.png"

#                     cv2.imwrite(os.path.join(class_path, filename), slice_2d)

#                     os.remove(file_path)

#                 except:
#                     continue

#         print("Test slice done!")

#     # =========================
#     # 🔥 AUGMENTATION SECTION
#     # =========================

#     def augment_image(self, img):
#         h, w = img.shape

#         # rotation
#         angle = random.uniform(-10, 10)
#         M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
#         img = cv2.warpAffine(img, M, (w, h))

#         # brightness
#         alpha = random.uniform(0.9, 1.1)
#         beta = random.randint(-10, 10)
#         img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

#         # noise
#         noise = np.random.normal(0, 5, img.shape)
#         img = np.clip(img + noise, 0, 255).astype(np.uint8)

#         return img

#     def group_by_subject(self, images):
#         subjects = {}

#         for img in images:
#             subject = img.split("_slice_")[0]

#             if subject not in subjects:
#                 subjects[subject] = []

#             subjects[subject].append(img)

#         return subjects

#     def augment_dataset_subjectwise(self, target_counts):
#         print("Starting smart augmentation...")

#         for cls in os.listdir(self.config.train_dir):

#             class_path = os.path.join(self.config.train_dir, cls)
#             images = [f for f in os.listdir(class_path) if f.endswith(".png")]

#             current_count = len(images)
#             target_count = target_counts.get(cls, current_count)

#             if current_count >= target_count:
#                 continue

#             needed = target_count - current_count

#             subjects = self.group_by_subject(images)
#             subject_keys = list(subjects.keys())

#             num_subjects = len(subject_keys)

#             aug_per_subject = needed // num_subjects
#             extra = needed % num_subjects

#             count = 0

#             print(f"{cls}: need {needed} images")

#             for idx, subject in enumerate(subject_keys):

#                 subject_images = subjects[subject]

#                 num_aug = aug_per_subject + (1 if idx < extra else 0)

#                 for i in range(num_aug):

#                     img_name = subject_images[i % len(subject_images)]
#                     img_path = os.path.join(class_path, img_name)

#                     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                     aug_img = self.augment_image(img)

#                     new_name = f"aug_{subject}_{count}.png"
#                     cv2.imwrite(os.path.join(class_path, new_name), aug_img)

#                     count += 1

#         print("Augmentation done!")

#     # -------------------------
#     # RUN PIPELINE
#     # -------------------------
#     def run(self):
#         print("Starting pipeline...")

#         os.makedirs(self.config.root_dir, exist_ok=True)

#         self.train_test_split()
#         self.extract_train_slices()
#         self.extract_test_slice()

#         # 🔥 TARGET COUNTS (your strategy)
#         target_counts = {
#             "autistic": 1500,
#             "non-autistic": 1400
#         }

#         self.augment_dataset_subjectwise(target_counts)

#         print("Pipeline completed successfully!")