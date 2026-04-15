
import os
import numpy as np
import cv2
import pandas as pd

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import opening, closing, disk

class FeatureExtractor:

    def __init__(self, image_size=256):
        self.image_size = image_size

    # -------------------------
    # GLCM FEATURES
    # -------------------------
    def extract_glcm(self, image):

        image = image.astype(np.uint8)

        glcm = graycomatrix(
            image=image,
            distances=[1,2,3],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        features = {}

        features["glcm_contrast"] = graycoprops(glcm, 'contrast').mean()
        features["glcm_correlation"] = graycoprops(glcm, 'correlation').mean()
        features["glcm_energy"] = graycoprops(glcm, 'energy').mean()
        features["glcm_homogeneity"] = graycoprops(glcm, 'homogeneity').mean()

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

        # #New features
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

# RESNET50 FEATURE EXTRACTION

# import os
# import numpy as np
# import cv2
# import pandas as pd
# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image


# class FeatureExtractor:

#     def __init__(self, image_size=224):
#         self.image_size = image_size

#         # Device (GPU if available)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # 🔥 Load pretrained ResNet18
#         self.model = models.resnet18(pretrained=True)

#         # Remove final classification layer (fc)
#         self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

#         self.model.to(self.device)
#         self.model.eval()

#         # Dynamically get feature size (should be 512 for ResNet18)
#         self.feature_dim = 512

#         # ✅ Correct preprocessing pipeline
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),   # resize first
#             transforms.ToTensor(),           # then tensor
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])

#     # -------------------------
#     # FEATURE EXTRACTION
#     # -------------------------
#     def extract_resnet_features(self, image):

#         # Convert grayscale → RGB
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#         # 🔥 FIX: Convert NumPy → PIL
#         image = Image.fromarray(image)

#         # Apply transforms
#         image = self.transform(image)

#         # Add batch dimension
#         image = image.unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             features = self.model(image)

#         return features.cpu().numpy().flatten()  # (512,)

#     # -------------------------
#     # PROCESS SINGLE IMAGE
#     # -------------------------
#     def process_image(self, image_path):

#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#         if img is None:
#             raise ValueError(f"Image not found: {image_path}")

#         return self.extract_resnet_features(img)

#     # -------------------------
#     # PROCESS DATASET
#     # -------------------------
#     def process_dataset(self, base_path):

#         dataset = []
#         labels = []

#         for cls in os.listdir(base_path):

#             cls_path = os.path.join(base_path, cls)

#             if not os.path.isdir(cls_path):
#                 continue

#             label = 1 if cls == "autistic" else 0

#             for file in os.listdir(cls_path):

#                 if not file.endswith(".png"):
#                     continue

#                 img_path = os.path.join(cls_path, file)

#                 print(f"Processing: {img_path}")

#                 try:
#                     features = self.process_image(img_path)
#                     dataset.append(features)
#                     labels.append(label)

#                 except Exception as e:
#                     print(f"Skipping {img_path} | Error: {e}")

#         # 🔥 Safety check
#         if len(dataset) == 0:
#             raise ValueError("No features extracted. Check preprocessing or image format.")

#         X = np.array(dataset)
#         y = np.array(labels)

#         print(f"✅ Extracted features shape: {X.shape}")

#         # Create column names dynamically
#         feature_names = [f"resnet18_feature_{i}" for i in range(self.feature_dim)]

#         df = pd.DataFrame(X, columns=feature_names)
#         df["label"] = y

#         return df
# import os
# import numpy as np
# import cv2
# import pandas as pd
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image


# # ============================================================
# # PATH TO RadImageNet WEIGHTS
# # Resolves to: ASD_ML_SMRI_MLOPS/pretrained_model/DenseNet121.pt
# # ============================================================
# BASE_DIR            = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# RADIMAGENET_WEIGHTS = os.path.join(BASE_DIR, "pretrained_model", "DenseNet121.pt")


# class FeatureExtractor:

#     def __init__(self, image_size=224):
#         self.image_size  = image_size
#         self.feature_dim = 1024   # DenseNet121 outputs 1024-dim features

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {self.device}")

#         # --------------------------------------------------
#         # LOAD DenseNet121 BACKBONE
#         # --------------------------------------------------
#         self.model = models.densenet121(pretrained=False)

#         # Replace classifier with Identity to get raw 1024-dim
#         # feature vectors instead of 1000-class probabilities
#         self.model.classifier = nn.Identity()

#         # --------------------------------------------------
#         # LOAD RadImageNet WEIGHTS
#         # --------------------------------------------------
#         print(f"Looking for RadImageNet weights at:\n  {RADIMAGENET_WEIGHTS}")

#         if os.path.exists(RADIMAGENET_WEIGHTS):

#             checkpoint = torch.load(
#                 RADIMAGENET_WEIGHTS,
#                 map_location=self.device
#             )

#             # .pt files can be saved as full model or state_dict
#             # — handle both cases safely
#             if isinstance(checkpoint, dict):
#                 self.model.load_state_dict(checkpoint, strict=False)
#             else:
#                 self.model.load_state_dict(checkpoint.state_dict(), strict=False)

#             print("RadImageNet DenseNet121 weights loaded successfully.")

#         else:
#             print(
#                 "WARNING: RadImageNet weights not found.\n"
#                 "Falling back to random initialization.\n"
#                 "Check that DenseNet121.pt is inside pretrained_model/ folder."
#             )

#         self.model.to(self.device)
#         self.model.eval()   # disables dropout + batchnorm train mode

#         # --------------------------------------------------
#         # PREPROCESSING PIPELINE
#         # --------------------------------------------------
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])

#     # -------------------------
#     # FEATURE EXTRACTION
#     # -------------------------
#     def extract_features(self, image):

#         # Step 1: Grayscale → RGB (DenseNet expects 3 channels)
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#         # Step 2: NumPy → PIL (torchvision transforms expect PIL)
#         image = Image.fromarray(image)

#         # Step 3: Resize, ToTensor, Normalize
#         image = self.transform(image)

#         # Step 4: Add batch dimension → (1, 3, 224, 224)
#         image = image.unsqueeze(0).to(self.device)

#         # Step 5: Forward pass — no gradients needed
#         with torch.no_grad():
#             features = self.model(image)

#         # Step 6: Flatten → (1024,)
#         return features.cpu().numpy().flatten()

#     # -------------------------
#     # PROCESS SINGLE IMAGE
#     # -------------------------
#     def process_image(self, image_path):

#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#         if img is None:
#             raise ValueError(f"Image not found: {image_path}")

#         return self.extract_features(img)

#     # -------------------------
#     # PROCESS DATASET
#     # -------------------------
#     def process_dataset(self, base_path):

#         dataset = []
#         labels  = []

#         for cls in os.listdir(base_path):

#             cls_path = os.path.join(base_path, cls)

#             if not os.path.isdir(cls_path):
#                 continue

#             label = 1 if cls == "autistic" else 0

#             for file in os.listdir(cls_path):

#                 if not file.lower().endswith(".png"):
#                     continue

#                 img_path = os.path.join(cls_path, file)

#                 print(f"Processing: {img_path}")

#                 try:
#                     features = self.process_image(img_path)
#                     dataset.append(features)
#                     labels.append(label)

#                 except Exception as e:
#                     print(f"Skipping {img_path} | Error: {e}")

#         # Safety check
#         if len(dataset) == 0:
#             raise ValueError("No features extracted. Check preprocessing or image format.")

#         X = np.array(dataset)   # (N, 1024)
#         y = np.array(labels)    # (N,)

#         print(f"Extracted features shape: {X.shape}")

#         feature_names = [f"densenet121_feature_{i}" for i in range(self.feature_dim)]

#         df = pd.DataFrame(X, columns=feature_names)
#         df["label"] = y

#         return df

# DENSENET121+GLCM+LBP+GFCC FEATURE EXTRACTION
# import os
# import numpy as np
# import cv2
# import pandas as pd
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image

# from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
# from skimage.filters import threshold_multiotsu
# from skimage.measure import label, regionprops
# from skimage.morphology import opening, closing, disk


# BASE_DIR            = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# RADIMAGENET_WEIGHTS = os.path.join(BASE_DIR, "pretrained_model", "DenseNet121.pt")


# class FeatureExtractor:

#     def __init__(self, image_size=224):
#         self.image_size  = image_size
#         self.feature_dim = 1024

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {self.device}")

#         # --------------------------------------------------
#         # LOAD DenseNet121
#         # --------------------------------------------------
#         self.model = models.densenet121(pretrained=False)
#         self.model.classifier = nn.Identity()

#         print(f"Looking for RadImageNet weights at:\n  {RADIMAGENET_WEIGHTS}")

#         if os.path.exists(RADIMAGENET_WEIGHTS):
#             checkpoint = torch.load(RADIMAGENET_WEIGHTS, map_location=self.device)
#             if isinstance(checkpoint, dict):
#                 self.model.load_state_dict(checkpoint, strict=False)
#             else:
#                 self.model.load_state_dict(checkpoint.state_dict(), strict=False)
#             print("yes,RadImageNet DenseNet121 weights loaded successfully.")
#         else:
#             print(
#                 "WARNING: RadImageNet weights not found.\n"
#                 "Falling back to random initialization.\n"
#                 "Check that DenseNet121.pt is inside pretrained_model/ folder."
#             )

#         self.model.to(self.device)
#         self.model.eval()

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])

#     # -------------------------
#     # DENSENET121 FEATURES
#     # -------------------------
#     def extract_densenet_features(self, image):
#         """Takes grayscale numpy image, returns 1024-dim feature vector."""

#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#         image = Image.fromarray(image)
#         image = self.transform(image)
#         image = image.unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             features = self.model(image)

#         return features.cpu().numpy().flatten()   # (1024,)

#     # -------------------------
#     # GLCM FEATURES
#     # -------------------------
#     def extract_glcm(self, image):

#         image = image.astype(np.uint8)

#         glcm = graycomatrix(
#             image,
#             distances=[1],
#             angles=[0],
#             levels=256,
#             symmetric=True,
#             normed=True
#         )

#         features = {}
#         features["glcm_contrast"]    = graycoprops(glcm, 'contrast')[0, 0]
#         features["glcm_correlation"] = graycoprops(glcm, 'correlation')[0, 0]
#         features["glcm_energy"]      = graycoprops(glcm, 'energy')[0, 0]
#         features["glcm_homogeneity"] = graycoprops(glcm, 'homogeneity')[0, 0]
#         features["glcm_mean"]        = np.mean(glcm)
#         features["glcm_variance"]    = np.var(glcm)
#         features["glcm_entropy"]     = -np.sum(glcm * np.log2(glcm + 1e-10))

#         return features

#     # -------------------------
#     # LBP FEATURES
#     # -------------------------
#     def extract_lbp(self, image):

#         lbp = local_binary_pattern(image, P=8, R=1, method='default')

#         hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
#         hist     = hist.astype("float")
#         hist    /= (hist.sum() + 1e-8)

#         return {f"lbp_{i}": hist[i] for i in range(256)}

#     # -------------------------
#     # GFCC FEATURES
#     # -------------------------
#     def extract_gfcc(self, image):

#         features = {}

#         thresholds = threshold_multiotsu(image, classes=3)
#         segmented  = np.digitize(image, bins=thresholds)

#         cc_mask = (segmented == 2).astype(np.uint8)
#         cc_mask = opening(cc_mask, disk(3))
#         cc_mask = closing(cc_mask, disk(3))

#         labeled = label(cc_mask)
#         regions = regionprops(labeled)

#         if len(regions) == 0:
#             return {
#                 "gfcc_area":       0,
#                 "gfcc_perimeter":  0,
#                 "gfcc_major_axis": 0,
#                 "gfcc_minor_axis": 0,
#                 "gfcc_solidity":   0,
#                 "gfcc_extent":     0,
#             }

#         region = max(regions, key=lambda r: r.area)

#         features["gfcc_area"]       = region.area
#         features["gfcc_perimeter"]  = region.perimeter + 1e-8
#         features["gfcc_major_axis"] = region.major_axis_length + 1e-8
#         features["gfcc_minor_axis"] = region.minor_axis_length + 1e-8
#         features["gfcc_solidity"]   = region.solidity
#         features["gfcc_extent"]     = region.extent

#         return features

#     # -------------------------
#     # PROCESS SINGLE IMAGE
#     # -------------------------
#     def process_image(self, image_path):

#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#         if img is None:
#             raise ValueError(f"Image not found: {image_path}")

#         img = cv2.resize(img, (self.image_size, self.image_size))

#         # --- DenseNet121 features (1024-dim) ---
#         densenet_feats = self.extract_densenet_features(img)
#         densenet_dict  = {
#             f"densenet_feature_{i}": densenet_feats[i]
#             for i in range(self.feature_dim)
#         }

#         # --- Handcrafted features (GLCM + LBP + GFCC) ---
#         handcrafted_dict = {}
#         handcrafted_dict.update(self.extract_glcm(img))
#         handcrafted_dict.update(self.extract_lbp(img))
#         handcrafted_dict.update(self.extract_gfcc(img))

#         # --- Combine ---
#         combined = {}
#         combined.update(densenet_dict)
#         combined.update(handcrafted_dict)

#         return combined

#     # -------------------------
#     # PROCESS DATASET
#     # -------------------------
#     def process_dataset(self, base_path):

#         dataset = []

#         for cls in os.listdir(base_path):

#             cls_path = os.path.join(base_path, cls)

#             if not os.path.isdir(cls_path):
#                 continue

#             label = 1 if cls == "autistic" else 0

#             for file in os.listdir(cls_path):

#                 if not file.lower().endswith(".png"):
#                     continue

#                 img_path = os.path.join(cls_path, file)
#                 print(f"Processing: {img_path}")

#                 try:
#                     features          = self.process_image(img_path)
#                     features["label"] = label
#                     dataset.append(features)

#                 except Exception as e:
#                     print(f"Skipping {img_path} | Error: {e}")

#         if len(dataset) == 0:
#             raise ValueError("No features extracted. Check preprocessing or image format.")

#         df = pd.DataFrame(dataset)
#         print(f"Extracted feature shape: {df.drop('label', axis=1).shape}")
#         print(f"  DenseNet121 features : 1024")
#         print(f"  GLCM features        : 7")
#         print(f"  LBP features         : 256")
#         print(f"  GFCC features        : 6")
#         print(f"  Total                : {df.shape[1] - 1}")

#         return df