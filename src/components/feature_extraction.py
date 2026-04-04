import os
import numpy as np
import cv2
import pandas as pd

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


class FeatureExtractor:

    def __init__(self, image_size=224):
        self.image_size = image_size

    # -------------------------
    # GLCM FEATURES
    # -------------------------
    def extract_glcm(self, image):

        image = image.astype(np.uint8)

        glcm = graycomatrix(
            image,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        features = {}

        features["glcm_contrast"] = graycoprops(glcm, 'contrast')[0, 0]
        features["glcm_correlation"] = graycoprops(glcm, 'correlation')[0, 0]
        features["glcm_energy"] = graycoprops(glcm, 'energy')[0, 0]
        features["glcm_homogeneity"] = graycoprops(glcm, 'homogeneity')[0, 0]

        # additional stats
        features["glcm_mean"] = np.mean(glcm)
        features["glcm_variance"] = np.var(glcm)
        features["glcm_entropy"] = -np.sum(glcm * np.log2(glcm + 1e-10))

        return features

    # -------------------------
    # LBP FEATURES
    # -------------------------
    def extract_lbp(self, image):

        lbp = local_binary_pattern(image, P=8, R=1, method='uniform')

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=256,
            range=(0, 256)
        )

        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-8)

        features = {}

        for i in range(256):
            features[f"lbp_{i}"] = hist[i]

        return features

    # -------------------------
    # PROCESS SINGLE IMAGE
    # -------------------------
    def process_image(self, image_path):

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.image_size, self.image_size))

        features = {}

        features.update(self.extract_glcm(img))
        features.update(self.extract_lbp(img))

        return features

    # -------------------------
    # PROCESS DATASET
    # -------------------------
    def process_dataset(self, base_path):

        dataset = []

        for cls in os.listdir(base_path):

            cls_path = os.path.join(base_path, cls)

            if not os.path.isdir(cls_path):
                continue

            label = 1 if cls == "autistic" else 0

            for file in os.listdir(cls_path):

                if not file.endswith(".png"):
                    continue

                img_path = os.path.join(cls_path, file)

                features = self.process_image(img_path)

                features["label"] = label

                dataset.append(features)

        return pd.DataFrame(dataset)