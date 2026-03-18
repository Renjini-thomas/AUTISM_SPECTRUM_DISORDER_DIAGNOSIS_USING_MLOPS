# import pandas as pd
# import numpy as np
# import joblib
# import mlflow

# from pathlib import Path
# from dotenv import load_dotenv
# import os

# from sklearn.feature_selection import RFECV
# from sklearn.model_selection import StratifiedKFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import make_scorer, recall_score


# class FeatureSelection:

#     def __init__(self):

#         load_dotenv()

#         os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

#         mlflow.set_tracking_uri(
#             "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
#         )

#         mlflow.set_experiment("ASD_CLASSICAL_ML")

#         self.feature_dir = Path("artifacts/features")
#         self.output_dir = Path("artifacts/selected_features")
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#     def load_data(self):

#         train_df = pd.read_csv(self.feature_dir / "train_features.csv")
#         test_df = pd.read_csv(self.feature_dir / "test_features.csv")

#         X_train = train_df.drop("label", axis=1)
#         y_train = train_df["label"]

#         X_test = test_df.drop("label", axis=1)
#         y_test = test_df["label"]

#         return X_train, y_train, X_test, y_test

#     def run(self):

#         X_train, y_train, X_test, y_test = self.load_data()

#         recall_scorer = make_scorer(
#             recall_score,
#             pos_label="autistic"
#         )

#         with mlflow.start_run(run_name="feature_selection_rfecv"):

#             estimator = LogisticRegression(max_iter=2000)

#             selector = RFECV(
#                 estimator=estimator,
#                 step=1,
#                 cv=StratifiedKFold(5),
#                 scoring=recall_scorer,
#                 n_jobs=-1
#             )

#             selector.fit(X_train, y_train)

#             X_train_sel = selector.transform(X_train)
#             X_test_sel = selector.transform(X_test)

#             # ⭐ SAVE ARRAYS
#             np.save(self.output_dir / "X_train.npy", X_train_sel)
#             np.save(self.output_dir / "X_test.npy", X_test_sel)
#             np.save(self.output_dir / "y_train.npy", y_train.values)
#             np.save(self.output_dir / "y_test.npy", y_test.values)

#             joblib.dump(selector, self.output_dir / "selector.joblib")

#             # ⭐ FEATURE NAMES
#             selected_cols = X_train.columns[selector.support_]

#             pd.DataFrame({
#                 "selected_features": selected_cols
#             }).to_csv(
#                 self.output_dir / "selected_feature_names.csv",
#                 index=False
#             )

#             # ⭐ FEATURE RANKING
#             ranking_df = pd.DataFrame({
#                 "feature": X_train.columns,
#                 "ranking": selector.ranking_
#             })

#             ranking_df.to_csv(
#                 self.output_dir / "feature_ranking.csv",
#                 index=False
#             )

#             # ⭐ LOG TO MLFLOW
#             mlflow.log_param(
#                 "num_selected_features",
#                 selector.n_features_
#             )

#             mlflow.log_param(
#                 "selection_method",
#                 "RFECV_logistic_recall"
#             )

#             mlflow.log_artifact(
#                 str(self.output_dir / "selected_feature_names.csv")
#             )

#             mlflow.log_artifact(
#                 str(self.output_dir / "feature_ranking.csv")
#             )

#             print("RFECV Feature Selection Completed ✅")
import pandas as pd
import numpy as np
import joblib
import mlflow
import os

from pathlib import Path
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score


class FeatureSelection:

    def __init__(self):

        load_dotenv()

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

        mlflow.set_tracking_uri(
            "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
        )

        mlflow.set_experiment("ASD_CLASSICAL_ML")

        self.feature_dir = Path("artifacts/features")
        self.output_dir = Path("artifacts/selected_features")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ================= LOAD =================
    def load_data(self):

        train_df = pd.read_csv(self.feature_dir / "train_features.csv")
        test_df = pd.read_csv(self.feature_dir / "test_features.csv")

        X_train = train_df.drop("label", axis=1)
        y_train = train_df["label"]

        X_test = test_df.drop("label", axis=1)
        y_test = test_df["label"]

        return X_train, y_train, X_test, y_test

    # ================= RUN =================
    def run(self):

        X_train, y_train, X_test, y_test = self.load_data()

        with mlflow.start_run(run_name="feature_selection_stable_rfecv"):

            print("Scaling Features...")

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            print("Applying PCA Stabilisation...")

            pca = PCA(n_components=100, random_state=42)

            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            print("Running RFECV...")

            recall_scorer = make_scorer(
                recall_score,
                pos_label="autism"
            )

            estimator = RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )

            selector = RFECV(
                estimator=estimator,
                step=10,
                cv=StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=42
                ),
                scoring=recall_scorer,
                n_jobs=-1
            )

            selector.fit(X_train, y_train)

            X_train_sel = selector.transform(X_train)
            X_test_sel = selector.transform(X_test)

            print("Selected Features:", selector.n_features_)

            # SAVE ARRAYS
            np.save(self.output_dir / "X_train.npy", X_train_sel)
            np.save(self.output_dir / "X_test.npy", X_test_sel)
            np.save(self.output_dir / "y_train.npy", y_train.values)
            np.save(self.output_dir / "y_test.npy", y_test.values)

            # SAVE OBJECTS
            joblib.dump(scaler, self.output_dir / "scaler.joblib")
            joblib.dump(pca, self.output_dir / "pca.joblib")
            joblib.dump(selector, self.output_dir / "selector.joblib")

            # LOG
            mlflow.log_param("pca_components", 150)
            mlflow.log_param("rfecv_step", 10)
            mlflow.log_param("rf_estimators", 400)
            mlflow.log_param("num_selected_features", selector.n_features_)

            print("Feature Selection Completed ✅")