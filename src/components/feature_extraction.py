# import cv2
# import numpy as np
# import pandas as pd
# import mlflow
# import os

# from pathlib import Path
# from tqdm import tqdm
# from scipy.stats import skew, kurtosis
# from dotenv import load_dotenv

# from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# class FeatureExtraction:

#     def __init__(self):

#         self.train_dir = Path("data/augmented/train")
#         self.test_dir = Path("data/preprocessed/test")

#         self.output_dir = Path("artifacts/features")
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#     # ===================== GLCM =====================
#     def extract_glcm(self, img):

#         distances = [1,2,4]
#         angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

#         glcm = graycomatrix(
#             img,
#             distances=distances,
#             angles=angles,
#             levels=256,
#             symmetric=True,
#             normed=True
#         )

#         feats = []

#         for prop in ['contrast','correlation','energy','homogeneity']:
#             vals = graycoprops(glcm, prop)
#             feats.extend(vals.flatten())

#         return feats

#     # ===================== LBP =====================
#     def extract_lbp(self, img):

#         lbp = local_binary_pattern(img, 8, 1, method="uniform")

#         hist,_ = np.histogram(
#             lbp.ravel(),
#             bins=np.arange(0,11),
#             range=(0,10)
#         )

#         hist = hist.astype("float")
#         hist /= (hist.sum()+1e-6)

#         return hist.tolist()

#     # ===================== INTENSITY STATS =====================
#     def extract_stats(self, img):

#         pixels = img.flatten()

#         hist = np.histogram(pixels, bins=256)[0]/len(pixels)
#         entropy = -np.sum((hist+1e-6)*np.log2(hist+1e-6))

#         return [
#             np.mean(pixels),
#             np.std(pixels),
#             skew(pixels),
#             kurtosis(pixels),
#             entropy
#         ]

#     # ===================== MORPHOLOGY =====================
#     def extract_gfcc(self, img):

#         _, thresh = cv2.threshold(img,0,255,cv2.THRESH_OTSU)

#         contours,_ = cv2.findContours(
#             thresh,
#             cv2.RETR_EXTERNAL,
#             cv2.CHAIN_APPROX_SIMPLE
#         )

#         if len(contours)==0:
#             return [0]*7

#         cc = max(contours, key=cv2.contourArea)

#         area = cv2.contourArea(cc)
#         perimeter = cv2.arcLength(cc,True)

#         x,y,w,h = cv2.boundingRect(cc)
#         aspect_ratio = w/(h+1e-6)

#         hull = cv2.convexHull(cc)
#         solidity = area/(cv2.contourArea(hull)+1e-6)

#         extent = area/(w*h+1e-6)

#         equiv_diameter = np.sqrt(4*area/np.pi)

#         return [
#             area,
#             perimeter,
#             w,
#             aspect_ratio,
#             solidity,
#             extent,
#             equiv_diameter
#         ]

#     # ===================== EDGE MAGNITUDE =====================
#     def extract_edges(self, img):

#         sobelx = cv2.Sobel(img, cv2.CV_64F,1,0)
#         sobely = cv2.Sobel(img, cv2.CV_64F,0,1)

#         mag = np.sqrt(sobelx**2 + sobely**2)

#         return [
#             np.mean(mag),
#             np.std(mag),
#             np.mean(mag > np.mean(mag))
#         ]

#     # ===================== HEMISPHERIC SYMMETRY ⭐⭐⭐⭐⭐
#     def extract_symmetry(self, img):

#         h,w = img.shape

#         if w % 2 != 0:
#             img = img[:, :-1]
#             w -= 1

#         left = img[:, :w//2]
#         right = np.fliplr(img[:, w//2:])

#         diff = np.abs(left-right)

#         corr = np.corrcoef(left.flatten(), right.flatten())[0,1]

#         return [
#             np.mean(diff),
#             np.std(diff),
#             np.max(diff),
#             corr
#         ]

#     # ===================== MIDLINE PROFILE ⭐⭐⭐⭐⭐
#     def extract_midline(self, img):

#         h,w = img.shape

#         mid = img[:, w//2-2:w//2+2]

#         return [
#             np.mean(mid),
#             np.std(mid),
#             np.sum(mid > np.mean(img))
#         ]

#     # ===================== VERTICAL REGIONS ⭐⭐⭐⭐
#     def extract_vertical_regions(self, img):

#         h,w = img.shape

#         regions = [
#             img[:h//4,:],
#             img[h//4:h//2,:],
#             img[h//2:3*h//4,:],
#             img[3*h//4:,:]
#         ]

#         feats=[]

#         for r in regions:
#             feats.append(np.mean(r))
#             feats.append(np.std(r))

#         return feats

#     # ===================== EDGE DIRECTION ⭐⭐⭐⭐
#     def extract_edge_direction(self, img):

#         sobelx = cv2.Sobel(img, cv2.CV_64F,1,0)
#         sobely = cv2.Sobel(img, cv2.CV_64F,0,1)

#         return [
#             np.mean(np.abs(sobelx)),
#             np.mean(np.abs(sobely)),
#             np.std(sobelx),
#             np.std(sobely)
#         ]

#     # ===================== PROCESS =====================
#     def process_dataset(self, base_dir):

#         rows=[]

#         for class_dir in base_dir.iterdir():

#             label = class_dir.name
#             image_dir = class_dir / "images"

#             for file in tqdm(list(image_dir.glob("*"))):

#                 if file.suffix==".npy":
#                     img = np.load(file)
#                 else:
#                     img = cv2.imread(str(file),0)

#                 img = cv2.normalize(
#                     img,None,0,255,
#                     cv2.NORM_MINMAX
#                 ).astype("uint8")

#                 features = (
#                     self.extract_glcm(img)
#                     + self.extract_lbp(img)
#                     + self.extract_stats(img)
#                     + self.extract_gfcc(img)
#                     + self.extract_edges(img)
#                     + self.extract_symmetry(img)
#                     + self.extract_midline(img)
#                     + self.extract_vertical_regions(img)
#                     + self.extract_edge_direction(img)
#                 )

#                 rows.append(features + [label])

#         # ================= FEATURE NAMES =================

#                 glcm_names = [
#                     f"glcm_{prop}_d{d}_a{ang}"
#                     for prop in ["contrast","correlation","energy","homogeneity"]
#                     for d in [1,2,4]
#                     for ang in ["0","45","90","135"]
#                 ]

#                 lbp_names = [f"lbp_bin_{i}" for i in range(10)]

#                 stat_names = [
#                     "intensity_mean",
#                     "intensity_std",
#                     "intensity_skew",
#                     "intensity_kurtosis",
#                     "intensity_entropy"
#                 ]

#                 gfcc_names = [
#                     "shape_area",
#                     "shape_perimeter",
#                     "shape_width",
#                     "shape_aspect_ratio",
#                     "shape_solidity",
#                     "shape_extent",
#                     "shape_equiv_diameter"
#                 ]

#                 edge_mag_names = [
#                     "edge_mag_mean",
#                     "edge_mag_std",
#                     "edge_mag_density"
#                 ]

#                 symmetry_names = [
#                     "symmetry_mean_diff",
#                     "symmetry_std_diff",
#                     "symmetry_max_diff",
#                     "symmetry_correlation"
#                 ]

#                 midline_names = [
#                     "midline_mean",
#                     "midline_std",
#                     "midline_high_intensity_ratio"
#                 ]

#                 vertical_names = []

#                 for i in range(4):
#                     vertical_names.append(f"vertical_region_{i}_mean")
#                     vertical_names.append(f"vertical_region_{i}_std")

#                 edge_dir_names = [
#                     "edge_dir_x_mean",
#                     "edge_dir_y_mean",
#                     "edge_dir_x_std",
#                     "edge_dir_y_std"
#                 ]

#                 columns = (
#                     glcm_names
#                     + lbp_names
#                     + stat_names
#                     + gfcc_names
#                     + edge_mag_names
#                     + symmetry_names
#                     + midline_names
#                     + vertical_names
#                     + edge_dir_names
#                     + ["label"]
#                 )

#         return pd.DataFrame(rows, columns=columns)

#     # ===================== RUN =====================
#     def run(self):

#         load_dotenv()

#         os.environ["MLFLOW_TRACKING_USERNAME"]=os.getenv("DAGSHUB_USERNAME")
#         os.environ["MLFLOW_TRACKING_PASSWORD"]=os.getenv("DAGSHUB_TOKEN")

#         mlflow.set_tracking_uri(
#             "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
#         )

#         mlflow.set_experiment("ASD_CLASSICAL_ML")

#         with mlflow.start_run(run_name="feature_extraction_sagittal_v2"):

#             train_df = self.process_dataset(self.train_dir)
#             test_df = self.process_dataset(self.test_dir)

#             train_path = self.output_dir / "train_features.csv"
#             test_path = self.output_dir / "test_features.csv"

#             train_df.to_csv(train_path,index=False)
#             test_df.to_csv(test_path,index=False)

#             mlflow.log_param("slice_type","mid_sagittal")
#             mlflow.log_param("feature_count", train_df.shape[1]-1)

#             mlflow.log_artifact(str(train_path))
#             mlflow.log_artifact(str(test_path))

#             print("Feature Extraction Completed ✅")
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import mlflow
import os

from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import cv2


class FeatureExtraction:

    def __init__(self):

        self.train_dir = Path("data/preprocessed/train")
        self.test_dir = Path("data/preprocessed/test")

        self.output_dir = Path("artifacts/features")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DEVICE
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # LOAD DENSENET121
        model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )

        # USE ONLY FEATURE BACKBONE
        self.backbone = model.features

        self.backbone.to(self.device)
        self.backbone.eval()

        # TRANSFORM
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # ================= FEATURE EXTRACTION =================
    def extract_feature(self, img):

        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():

            feat_map = self.backbone(img)

            feat_map = torch.relu(feat_map)

            avg_pool = torch.mean(feat_map, dim=(2, 3))
            max_pool = torch.amax(feat_map, dim=(2, 3))
            std_pool = torch.std(feat_map, dim=(2, 3))

            feat = torch.cat([avg_pool, max_pool, std_pool], dim=1)

        return feat.squeeze().cpu().numpy()

    # ================= PROCESS DATASET =================
    def process_dataset(self, base_dir):

        rows = []

        for class_dir in base_dir.iterdir():

            if not class_dir.is_dir():
                continue

            label = class_dir.name
            files = list(class_dir.glob("*.png"))

            print(f"{label} → {len(files)} images found")

            subject_dict = {}

            for file in tqdm(files, desc=f"Processing {label}"):

                subject_id = file.stem.split("_slice")[0]

                img = cv2.imread(str(file), 0)

                if img is None:
                    continue

                # img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                # img = (img * 255).astype("uint8")

                feat = self.extract_feature(img)

                subject_dict.setdefault(subject_id, []).append(feat)

            # SUBJECT LEVEL AGGREGATION
            for subject_id, feat_list in subject_dict.items():

                feat_stack = np.vstack(feat_list)

                mean_feat = np.mean(feat_stack, axis=0)
                max_feat = np.max(feat_stack, axis=0)
                std_feat = np.std(feat_stack, axis=0)

                # subject_feat = np.concatenate(
                #     [mean_feat, max_feat, std_feat]
                # )
                # subject_feat = np.concatenate([mean_feat, std_feat])
                subject_feat = mean_feat
                rows.append(list(subject_feat) + [label])

        if len(rows) == 0:
            raise ValueError("No features extracted!")

        feature_dim = len(rows[0]) - 1

        columns = [
            f"deep_feat_{i}" for i in range(feature_dim)
        ] + ["label"]

        df = pd.DataFrame(rows, columns=columns)

        return df, feature_dim

    # ================= RUN =================
    def run(self):

        load_dotenv()

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

        mlflow.set_tracking_uri(
            "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
        )

        mlflow.set_experiment("ASD_DEEP_FEATURES")

        with mlflow.start_run(run_name="densenet121_feature_extraction"):

            print("Extracting TRAIN deep features...")
            train_df, feature_dim = self.process_dataset(self.train_dir)

            print("Extracting TEST deep features...")
            test_df, _ = self.process_dataset(self.test_dir)

            train_path = self.output_dir / "train_features.csv"
            test_path = self.output_dir / "test_features.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            mlflow.log_param("feature_type", "DenseNet121")
            mlflow.log_param("feature_dim", feature_dim)
            mlflow.log_param("device", str(self.device))

            mlflow.log_artifact(str(train_path))
            mlflow.log_artifact(str(test_path))

            print("DenseNet Feature Extraction Completed ✅")
# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms

# import numpy as np
# import pandas as pd
# import mlflow
# import os

# from pathlib import Path
# from tqdm import tqdm
# from dotenv import load_dotenv
# import cv2


# class FeatureExtraction:

#     def __init__(self):

#         self.train_dir = Path("data/preprocessed/train")
#         self.test_dir  = Path("data/preprocessed/test")

#         self.output_dir = Path("artifacts/features")
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         # DEVICE
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )

#         # LOAD DENSENET121
#         model = models.densenet121(
#             weights=models.DenseNet121_Weights.IMAGENET1K_V1
#         )

#         # USE ONLY FEATURE BACKBONE
#         self.backbone = model.features
#         self.backbone.to(self.device)
#         self.backbone.eval()

#         # TRANSFORM
#         # ✅ PNG from preprocessing is already uint8 (z-score → min-max scaled)
#         # ✅ grayscale → 3-channel via repeat (required for ImageNet-pretrained model)
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),                            # → [0.0, 1.0], shape (1,H,W)
#             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # (1,H,W) → (3,H,W)
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])

#     # ================= FEATURE EXTRACTION =================
#     def extract_feature(self, img):
#         """
#         img : uint8 grayscale numpy array (already normalized PNG from preprocessing)

#         FIX 1 — Removed erroneous torch.relu():
#             DenseNet121's last dense block already ends with BatchNorm + ReLU.
#             Applying ReLU again was zeroing out negative activations that carry
#             real structural contrast information from MRI tissue boundaries.

#         FIX 2 — Returns only avg_pool (1024-d):
#             Smarter weighting is done at subject level (weighted mean by L2 norm),
#             keeping feature dimensionality at 1024 which is safe for ~760 subjects.
#         """
#         x = self.transform(img).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             feat_map = self.backbone(x)                       # (1, 1024, 7, 7)
#             # ✅ FIX 1: No extra relu — DenseNet already has BN+ReLU internally
#             avg_pool = torch.mean(feat_map, dim=(2, 3))       # (1, 1024)

#         return avg_pool.squeeze().cpu().numpy()               # (1024,)

#     # ================= PROCESS DATASET =================
#     def process_dataset(self, base_dir):

#         rows = []

#         for class_dir in sorted(base_dir.iterdir()):

#             if not class_dir.is_dir():
#                 continue

#             label = class_dir.name
#             files = sorted(class_dir.glob("*.png"))

#             print(f"\n{label} → {len(files)} slices found")

#             subject_dict = {}

#             for file in tqdm(files, desc=f"Processing {label}"):

#                 subject_id = file.stem.split("_slice")[0]

#                 # ✅ FIX 2: Removed double normalization.
#                 # preprocessing.py already does:
#                 #   z-score per slice → min-max scale → save uint8 PNG
#                 # Re-normalizing here was distorting the intensity distribution.
#                 img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

#                 if img is None:
#                     print(f"  ⚠ Could not read: {file.name}")
#                     continue

#                 # ✅ Skip near-empty / background slices
#                 # Edge mid-sagittal slices sometimes have very low variance
#                 # (mostly skull/background) — these add noise to aggregation
#                 if np.var(img) < 50:
#                     continue

#                 feat = self.extract_feature(img)
#                 subject_dict.setdefault(subject_id, []).append(feat)

#             # =============================================
#             # SUBJECT LEVEL AGGREGATION
#             # =============================================
#             for subject_id, feat_list in subject_dict.items():

#                 feat_stack = np.vstack(feat_list)   # (n_slices, 1024)

#                 # ✅ FIX 3: Weighted mean by L2 norm of each slice's feature vector.
#                 # Slices with more brain tissue produce higher activation norms
#                 # and contribute more to the subject representation.
#                 # Feature dim stays at 1024 — no dimensionality explosion.
#                 norms   = np.linalg.norm(feat_stack, axis=1)  # (n_slices,)
#                 weights = norms / (norms.sum() + 1e-8)         # normalize to sum=1

#                 subject_feat = np.average(
#                     feat_stack, axis=0, weights=weights
#                 )  # (1024,)

#                 rows.append(list(subject_feat) + [label])

#         if len(rows) == 0:
#             raise ValueError(
#                 "No features extracted! Check data paths and PNG files."
#             )

#         feature_dim = len(rows[0]) - 1
#         columns = [f"deep_feat_{i}" for i in range(feature_dim)] + ["label"]

#         df = pd.DataFrame(rows, columns=columns)

#         print(f"\nSubject counts:\n{df['label'].value_counts().to_string()}")

#         return df, feature_dim

#     # ================= RUN =================
#     def run(self):

#         load_dotenv()

#         os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

#         mlflow.set_tracking_uri(
#             "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
#         )

#         mlflow.set_experiment("ASD_DEEP_FEATURES")

#         with mlflow.start_run(run_name="densenet121_feature_extraction_v3"):

#             print("Extracting TRAIN deep features...")
#             train_df, feature_dim = self.process_dataset(self.train_dir)

#             print("\nExtracting TEST deep features...")
#             test_df, _ = self.process_dataset(self.test_dir)

#             train_path = self.output_dir / "train_features.csv"
#             test_path  = self.output_dir / "test_features.csv"

#             train_df.to_csv(train_path, index=False)
#             test_df.to_csv(test_path, index=False)

#             mlflow.log_param("model_backbone",       "DenseNet121_ImageNet")
#             mlflow.log_param("feature_dim",           feature_dim)
#             mlflow.log_param("pooling",               "global_avg_pool")
#             mlflow.log_param("aggregation",           "weighted_mean_l2norm")
#             mlflow.log_param("extra_relu_removed",    True)
#             mlflow.log_param("double_norm_removed",   True)
#             mlflow.log_param("empty_slice_filter",    "var<50")
#             mlflow.log_param("slices_per_subject",    11)
#             mlflow.log_param("slice_plane",           "mid_sagittal")
#             mlflow.log_param("device",                str(self.device))
#             mlflow.log_param("train_subjects",        len(train_df))
#             mlflow.log_param("test_subjects",         len(test_df))
#             mlflow.log_param(
#                 "train_class_dist",
#                 train_df["label"].value_counts().to_dict()
#             )

#             mlflow.log_artifact(str(train_path))
#             mlflow.log_artifact(str(test_path))

#             print(f"\nDenseNet Feature Extraction Completed ✅")
#             print(f"Feature dim   : {feature_dim}")
#             print(f"Train subjects: {len(train_df)}")
#             print(f"Test subjects : {len(test_df)}")
