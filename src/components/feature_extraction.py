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
# import cv2

# from pathlib import Path
# from tqdm import tqdm
# from dotenv import load_dotenv

# from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
# from skimage.measure import regionprops, label
# from skimage.filters import threshold_multiotsu


# class FeatureExtraction:

#     def __init__(self):

#         self.train_dir = Path("data/preprocessed/train")
#         self.test_dir = Path("data/preprocessed/test")

#         self.output_dir = Path("artifacts/features")
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )

#         # DenseNet backbone
#         model = models.densenet121(
#             weights=models.DenseNet121_Weights.IMAGENET1K_V1
#         )
#         self.backbone = model.features.to(self.device).eval()

#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x.repeat(3,1,1)),
#             transforms.Normalize(
#                 mean=[0.485,0.456,0.406],
#                 std=[0.229,0.224,0.225]
#             )
#         ])

#     # ================= DEEP =================
#     def extract_deep(self, img):

#         x = self.transform(img).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             fmap = self.backbone(x)
#             fmap = torch.relu(fmap)
#             avg_pool = torch.mean(fmap, dim=(2,3))

#         return avg_pool.squeeze().cpu().numpy()

#     # ================= GLCM =================
#     def extract_glcm(self, img):

#         img = cv2.resize(img,(128,128))

#         glcm = graycomatrix(
#             img,
#             distances=[1],
#             angles=[0],
#             levels=256,
#             symmetric=True,
#             normed=True
#         )

#         feats = []

#         props = [
#             "contrast","correlation","energy",
#             "homogeneity","ASM","dissimilarity"
#         ]

#         for p in props:
#             feats.append(graycoprops(glcm,p)[0,0])

#         P = glcm[:,:,0,0]
#         P = P/(P.sum()+1e-6)

#         i,j = np.indices(P.shape)

#         mean = np.sum(i*P)
#         var = np.sum(((i-mean)**2)*P)
#         entropy = -np.sum(P*np.log(P+1e-6))
#         max_prob = np.max(P)

#         cluster_shade = np.sum(((i+j-2*mean)**3)*P)
#         cluster_prom = np.sum(((i+j-2*mean)**4)*P)

#         feats += [
#             mean,var,entropy,
#             cluster_shade,cluster_prom,max_prob
#         ]

#         return np.array(feats)

#     # ================= LBP =================
#     def extract_lbp(self, img):

#         lbp = local_binary_pattern(
#             img,
#             P=8,
#             R=1,
#             method="uniform"
#         )

#         hist,_ = np.histogram(
#             lbp.ravel(),
#             bins=59,
#             range=(0,59),
#             density=True
#         )

#         return hist

#     # ================= GFCC =================
#     def extract_gfcc(self, img):

#         try:
#             thresholds = threshold_multiotsu(img, classes=3)
#             segmented = np.digitize(img, bins=thresholds)

#             cc = (segmented==2).astype(np.uint8)

#             lbl = label(cc)
#             props = regionprops(lbl)

#             if len(props)==0:
#                 return np.zeros(11)

#             r = max(props, key=lambda x:x.area)

#             area = r.area
#             perim = r.perimeter
#             major = r.major_axis_length
#             minor = r.minor_axis_length
#             solidity = r.solidity
#             extent = r.extent

#             ecc = r.eccentricity
#             circ = 4*np.pi*area/(perim**2+1e-6)
#             axis_ratio = minor/(major+1e-6)
#             convex_ratio = area/(r.convex_area+1e-6)

#             minr,minc,maxr,maxc = r.bbox
#             bbox_area = (maxr-minr)*(maxc-minc)
#             bbox_ratio = area/(bbox_area+1e-6)

#             return np.array([
#                 area,perim,major,minor,
#                 solidity,extent,
#                 ecc,circ,axis_ratio,
#                 convex_ratio,bbox_ratio
#             ])

#         except:
#             return np.zeros(11)

#     # ================= DATASET =================
#     def process_dataset(self, base_dir):

#         rows = []

#         for class_dir in base_dir.iterdir():

#             if not class_dir.is_dir():
#                 continue

#             label = class_dir.name
#             files = list(class_dir.glob("*.png"))

#             subject_dict = {}

#             for file in tqdm(files, desc=f"{label}"):

#                 subject_id = file.stem.split("_slice")[0]
#                 slice_idx = int(file.stem.split("_slice")[1])

#                 img = cv2.imread(str(file),0)

#                 if img is None:
#                     continue

#                 deep = self.extract_deep(img)
#                 glcm = self.extract_glcm(img)
#                 lbp = self.extract_lbp(img)

#                 # GFCC only mid slices
#                 if 40 <= slice_idx <= 60:
#                     gfcc = self.extract_gfcc(img)
#                 else:
#                     gfcc = np.zeros(11)

#                 feat = np.concatenate([deep,glcm,lbp,gfcc])

#                 subject_dict.setdefault(subject_id, []).append(feat)

#             # SUBJECT LEVEL AGGREGATION
#             for sid,flist in subject_dict.items():

#                 stack = np.vstack(flist)
#                 subject_feat = np.mean(stack, axis=0)

#                 rows.append(list(subject_feat)+[label])

#         # ===== FEATURE NAMES =====
#         deep_cols = [f"deep_{i}" for i in range(1024)]

#         glcm_cols = [
#             "glcm_contrast","glcm_correlation","glcm_energy",
#             "glcm_homogeneity","glcm_ASM","glcm_dissimilarity",
#             "glcm_mean","glcm_variance","glcm_entropy",
#             "glcm_cluster_shade","glcm_cluster_prominence",
#             "glcm_max_probability"
#         ]

#         lbp_cols = [f"lbp_bin_{i}" for i in range(59)]

#         gfcc_cols = [
#             "gfcc_area","gfcc_perimeter","gfcc_major_axis",
#             "gfcc_minor_axis","gfcc_solidity","gfcc_extent",
#             "gfcc_eccentricity","gfcc_circularity",
#             "gfcc_axis_ratio","gfcc_convex_ratio",
#             "gfcc_bbox_ratio"
#         ]

#         cols = deep_cols + glcm_cols + lbp_cols + gfcc_cols + ["label"]

#         df = pd.DataFrame(rows, columns=cols)

#         return df, len(cols)-1

#     # ================= RUN =================
#     def run(self):

#         load_dotenv()
#         os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
        

#         mlflow.set_tracking_uri(
#             "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
#         )

#         mlflow.set_experiment("ASD_HYBRID_FEATURES")

#         with mlflow.start_run():

#             train_df,dim = self.process_dataset(self.train_dir)
#             test_df,_ = self.process_dataset(self.test_dir)

#             train_path = self.output_dir/"train_features.csv"
#             test_path = self.output_dir/"test_features.csv"

#             train_df.to_csv(train_path,index=False)
#             test_df.to_csv(test_path,index=False)

#             mlflow.log_param("feature_type","hybrid(DENSENET121+GLCM+LBP+GFCC)")
#             mlflow.log_param("feature_dim",dim)
#             mlflow.log_artifact(str(train_path))
#             mlflow.log_artifact(str(test_path))
#             mlflow.log_param("train_samples", len(train_df))
#             mlflow.log_param("test_samples", len(test_df))

#             print("✅ Hybrid Feature Extraction Completed")