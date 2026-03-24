import numpy as np
import cv2
import nibabel as nib
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import joblib
import tempfile
import os
from dotenv import load_dotenv
import mlflow


from pathlib import Path
from mlflow.tracking import MlflowClient


class ASD_Prediction:

    def __init__(self):
        load_dotenv()

        username = os.getenv("DAGSHUB_USERNAME")
        token = os.getenv("DAGSHUB_TOKEN")

        if username and token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        mlflow.set_tracking_uri("https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow")
        # ===== DEVICE =====
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ===== LOAD DENSENET BACKBONE =====
        model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )

        self.backbone = model.features.to(self.device)
        self.backbone.eval()

        # ===== TRANSFORM (SAME AS TRAINING) =====
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

        # ===== LOAD BEST MODEL =====
        self.model = self.load_best_model()

    # ==========================================
    # LOAD MODEL FROM MLFLOW REGISTRY (STAGING)
    # ==========================================

    def load_best_model(self):

        model_uri = "models:/ASD_BEST_MODEL@staging"
        model = mlflow.sklearn.load_model(model_uri)

        return model

    # ==========================================
    # MRI PREPROCESSING (SAME AS YOUR TRAINING)
    # ==========================================

    def preprocess_subject(self, mri_path):

        img = nib.load(mri_path)
        img = nib.as_closest_canonical(img)
        volume = img.get_fdata()

        mid = volume.shape[0] // 2

        # ⭐ SAME 11 SLICE STRATEGY
        offsets = list(range(-5,6))

        slices = []

        for off in offsets:

            idx = mid + off

            if idx < 0 or idx >= volume.shape[0]:
                continue

            slice_2d = volume[idx,:,:]

            # orientation fix
            slice_2d = np.rot90(slice_2d)

            # resize SAME as preprocessing
            slice_2d = cv2.resize(slice_2d,(128,128))
            min_val = slice_2d.min()
            max_val = slice_2d.max()

            slice_2d = (slice_2d - min_val) / (max_val - min_val + 1e-8)
            slice_2d = (slice_2d * 255).astype(np.uint8)

            slices.append(slice_2d)

        return slices

    # ==========================================
    # FEATURE EXTRACTION (SAME AS TRAINING)
    # ==========================================

    def extract_feature(self, img):

        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():

            feat_map = self.backbone(img)

            feat_map = torch.relu(feat_map)

            avg_pool = torch.mean(feat_map, dim=(2,3))
            max_pool = torch.amax(feat_map, dim=(2,3))
            std_pool = torch.std(feat_map, dim=(2,3))

            feat = torch.cat([avg_pool, max_pool, std_pool], dim=1)

        return feat.squeeze().cpu().numpy()
        

    # ==========================================
    # FINAL PREDICTION FUNCTION
    # ==========================================

    def predict(self, mgz_path):

        slices = self.preprocess_subject(mgz_path)

        feats = []

        for s in slices:
            # feats.append(self.extract_feature(s))
            fd,temp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cv2.imwrite(temp_path, s)
            img = cv2.imread(temp_path,0)
            os.remove(temp_path)
            feats.append(self.extract_feature(img))

        feat_stack = np.vstack(feats)

        # ⭐ SAME SUBJECT AGGREGATION
        subject_feat = np.mean(feat_stack, axis=0)

        pred = self.model.predict([subject_feat])[0]

        prob = self.model.predict_proba([subject_feat])[0]

        return pred, prob, slices