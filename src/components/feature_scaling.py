import pandas as pd
import numpy as np
import joblib
import mlflow

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os


class FeatureScaling:

    def __init__(self):

        load_dotenv()

        username = os.getenv("DAGSHUB_USERNAME")
        token = os.getenv("DAGSHUB_TOKEN")

        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        mlflow.set_tracking_uri(
            "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
        )

        mlflow.set_experiment("ASD_CLASSICAL_ML")

        self.feature_dir = Path("artifacts/selected_features")

        self.output_dir = Path("artifacts/scaled")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):

        with mlflow.start_run(run_name="feature_scaling_stage"):

            print("Loading selected feature arrays...")

            X_train = np.load(self.feature_dir / "X_train.npy")
            y_train = np.load(self.feature_dir / "y_train.npy", allow_pickle=True)

            X_test = np.load(self.feature_dir / "X_test.npy")
            y_test = np.load(self.feature_dir / "y_test.npy", allow_pickle=True)

            print("Fitting StandardScaler on TRAIN data...")

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # ================= SAVE =================

            np.save(self.output_dir / "X_train.npy", X_train_scaled)
            np.save(self.output_dir / "y_train.npy", y_train)

            np.save(self.output_dir / "X_test.npy", X_test_scaled)
            np.save(self.output_dir / "y_test.npy", y_test)

            scaler_path = self.output_dir / "scaler.joblib"
            joblib.dump(scaler, scaler_path)

            # ================= MLFLOW LOGGING =================

            mlflow.log_param("scaler_type", "StandardScaler")
            mlflow.log_param("num_features_after_selection", X_train.shape[1])

            mlflow.log_artifact(str(scaler_path))

            print("Feature Scaling Completed ✅")