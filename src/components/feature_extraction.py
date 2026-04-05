import os
import numpy as np
import cv2
import pandas as pd

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import opening, closing, disk

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

        lbp = local_binary_pattern(image, P=8, R=1, method='default')

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
    # GFCC FEATURES (Corpus Callosum Geometry)
    # -------------------------
    def extract_gfcc(self, image):

    

        features = {}

        thresholds = threshold_multiotsu(image, classes=3)
        segmented = np.digitize(image, bins=thresholds)

        cc_mask = (segmented == 2).astype(np.uint8)

        cc_mask = opening(cc_mask, disk(3))
        cc_mask = closing(cc_mask, disk(3))

        labeled = label(cc_mask)
        regions = regionprops(labeled)

        if len(regions) == 0:
            return {
                "gfcc_area": 0,
                "gfcc_perimeter": 0,
                "gfcc_major_axis": 0,
                "gfcc_minor_axis": 0,
                "gfcc_solidity": 0,
                "gfcc_extent": 0,
                # "gfcc_eccentricity": 0,
                # "gfcc_circularity": 0,
                # "gfcc_axis_ratio": 0,
                # "gfcc_convex_ratio": 0,
                # "gfcc_bbox_ratio": 0
            }

        region = max(regions, key=lambda r: r.area)

        area = region.area
        perimeter = region.perimeter + 1e-8
        major = region.major_axis_length + 1e-8
        minor = region.minor_axis_length + 1e-8
        convex_area = region.convex_area + 1e-8

        minr, minc, maxr, maxc = region.bbox
        width = maxc - minc + 1e-8
        height = maxr - minr + 1e-8

        # Existing features
        features["gfcc_area"] = area
        features["gfcc_perimeter"] = perimeter
        features["gfcc_major_axis"] = major
        features["gfcc_minor_axis"] = minor
        features["gfcc_solidity"] = region.solidity
        features["gfcc_extent"] = region.extent

        # 🔥 New features
        # features["gfcc_eccentricity"] = region.eccentricity

        # features["gfcc_circularity"] = (4 * np.pi * area) / (perimeter ** 2)

        # features["gfcc_axis_ratio"] = major / minor

        # features["gfcc_convex_ratio"] = area / convex_area

        # features["gfcc_bbox_ratio"] = width / height

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
        features.update(self.extract_gfcc(img))

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