# import numpy as np
# import joblib
# from pathlib import Path

# import mlflow
# import mlflow.sklearn
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC, LinearSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from dotenv import load_dotenv
# import os
# from sklearn.metrics import make_scorer, recall_score

# class ModelTrainer:

#     def __init__(self):
#         load_dotenv()
#         username = os.getenv("DAGSHUB_USERNAME")
#         token = os.getenv("DAGSHUB_TOKEN")

#         os.environ["MLFLOW_TRACKING_USERNAME"] = username
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = token
#         self.scaled_dir = Path("artifacts/selected_features")
#         self.model_dir = Path("artifacts/models")
#         self.model_dir.mkdir(parents=True, exist_ok=True)

#         mlflow.set_tracking_uri("https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow")
#         mlflow.set_experiment("ASD_CLASSICAL_ML")

#     def load_data(self):

#         X_train = np.load(self.scaled_dir / "X_train.npy")
#         y_train = np.load(self.scaled_dir / "y_train.npy", allow_pickle=True)

#         X_test = np.load(self.scaled_dir / "X_test.npy")
#         y_test = np.load(self.scaled_dir / "y_test.npy", allow_pickle=True)

#         return X_train, y_train, X_test, y_test

#     def get_models(self):

#         return {

#             "knn": (
#                 KNeighborsClassifier(),
#                 {"n_neighbors": [5,7]}
#             ),

#             "svm": (   # ⭐ FAST LINEAR SVM
#                 LinearSVC(
#                     C=1,
#                     class_weight="balanced",
#                     max_iter=5000
#                 ),
#                 {}
#             ),

#             "decision Tree": (
#                 DecisionTreeClassifier(),
#                 {"max_depth":[3,5,10]}
#             ),

#             "random forest": (
#                 RandomForestClassifier(),
#                 {"n_estimators":[100,200], "max_depth":[5,10]}
#             ),

#             "gradient boosting": (
#                 GradientBoostingClassifier(),
#                 {"learning_rate":[0.01,0.1], "n_estimators":[100]}
#             ),

#             "logistic regression": (
#                 Pipeline([
                    
#                     ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
#                 ]),
#                 {"model__C":[0.1,1,10]}
#             )
#         }
#     def run(self):

#         X_train, y_train, _, _ = self.load_data()   # ⭐ DO NOT USE TEST

#         best_model = None
#         best_name = ""

#         best_recall = 0
#         best_auc = 0
#         best_f1 = 0

#         cv = StratifiedKFold(
#             n_splits=5,
#             shuffle=True,
#             random_state=42
#         )

#         scoring = {
#             "recall": make_scorer(recall_score, pos_label="autism"),
#             "auc": "roc_auc",
#             "f1": make_scorer(f1_score, pos_label="autism"),
#             "accuracy": "accuracy"
#         }

#         for name, (model, params) in self.get_models().items():

#             with mlflow.start_run(run_name=name):

#                 print(f"Training {name}")

#                 grid = GridSearchCV(
#                     model,
#                     params,
#                     cv=cv,
#                     scoring=scoring,
#                     refit="recall",      # ⭐ model refitted on best recall
#                     n_jobs=-1,
#                     verbose=1
#                 )

#                 grid.fit(X_train, y_train)

#                 best_estimator = grid.best_estimator_

#                 idx = grid.best_index_

#                 cv_recall = grid.cv_results_["mean_test_recall"][idx]
#                 cv_auc = grid.cv_results_["mean_test_auc"][idx]
#                 cv_f1 = grid.cv_results_["mean_test_f1"][idx]
#                 cv_acc = grid.cv_results_["mean_test_accuracy"][idx]

#                 mlflow.log_params(grid.best_params_)
#                 mlflow.log_metric("cv_recall", cv_recall)
#                 mlflow.log_metric("cv_auc", cv_auc)
#                 mlflow.log_metric("cv_f1", cv_f1)
#                 mlflow.log_metric("cv_accuracy", cv_acc)

#                 print(f"{name} → Recall:{cv_recall:.4f} AUC:{cv_auc:.4f} F1:{cv_f1:.4f}")

#                 # ⭐ MODEL SELECTION RULE
#                 if cv_recall > best_recall:

#                     best_model = best_estimator
#                     best_name = name
#                     best_recall = cv_recall
#                     best_auc = cv_auc
#                     best_f1 = cv_f1

#                 elif cv_recall == best_recall:

#                     if cv_auc > best_auc:

#                         best_model = best_estimator
#                         best_name = name
#                         best_auc = cv_auc
#                         best_f1 = cv_f1

#                     elif cv_auc == best_auc:

#                         if cv_f1 > best_f1:

#                             best_model = best_estimator
#                             best_name = name
#                             best_f1 = cv_f1

#         joblib.dump(best_model, self.model_dir / "best_model.joblib")

#         print("⭐ BEST MODEL:", best_name)
#         print("CV Recall:", best_recall)
#         print("CV AUC:", best_auc)
#         print("CV F1:", best_f1)
#         print("CV Accuracy:", cv_acc)

#         print("Model Saved ✅")
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import hashlib
import joblib
from sklearn.linear_model import LogisticRegression

from pathlib import Path
from dotenv import load_dotenv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,PowerTransformer
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, f1_score, balanced_accuracy_score,auc, accuracy_score, precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import tempfile
import optuna
from optuna.integration import OptunaSearchCV


class ModelTrainer:

    def __init__(self):

        load_dotenv()

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

        mlflow.set_tracking_uri(
            "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
        )

        mlflow.set_experiment("ASD_MLOPS_PIPELINE")

        self.feature_dir = Path("artifacts/features")
        # self.model_dir = Path("artifacts/models")
        # self.model_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- DATA HASH ----------------

    def get_file_hash(self, path):

        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    # ---------------- LOAD DATA ----------------

    def load_data(self):

        train_df = pd.read_csv(self.feature_dir / "train_features.csv")

        X_train = train_df.drop("label", axis=1)
        y_train = train_df["label"].values

        return X_train, y_train

    # ---------------- MODELS ----------------

    def get_models(self):

        return {

            "logistic_regression": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA()),
                    ("model", LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced"
                    ))
                ]),
                {   
                    "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler(), PowerTransformer()],
                    "pca__n_components": [0.90,0.95, 0.99],
                    "model__C": [0.01, 0.1, 1, 10],
                    "model__penalty": ["l1", "l2"],
                    "model__solver": ["liblinear", "saga"]
                }
            ),

            "random_forest": (
                Pipeline([
                    
                    ("pca", PCA()),
                    ("model", RandomForestClassifier(
                        class_weight="balanced",
                        random_state=42
                    ))
                ]),
                {
                    "pca__n_components": [0.90, 0.95, 0.99],
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [5, 10, 15]
                }
            ),

            "svm": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA()),
                    ("model", SVC(
                        probability=True,
                        class_weight="balanced"
                    ))
                ]),
                {
                    "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler(), PowerTransformer()],
                    "pca__n_components": [0.90, 0.95, 0.99],
                    "model__kernel": ["rbf"],
                    "model__C": [0.5, 1, 5],
                    "model__gamma": ["scale"]
                }
            ),

            "knn": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA()),
                    ("model", KNeighborsClassifier())
                ]),
                {
                    "scaler": [StandardScaler(), RobustScaler(), MinMaxScaler(), PowerTransformer()],
                    "pca__n_components": [0.90, 0.95, 0.99],
                    "model__n_neighbors": [5, 7, 9]
                }
            )
        }

    # ---------------- RUN ----------------

    def run(self):

        X_train, y_train = self.load_data()

        data_hash = self.get_file_hash(
            self.feature_dir / "train_features.csv"
        )

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        scoring = {
            "recall": make_scorer(recall_score, pos_label="autism"),
            "f1": make_scorer(f1_score, pos_label="autism"),
            "bal_acc": make_scorer(balanced_accuracy_score),
            "auc": "roc_auc",
            "accuracy": "accuracy"
        }

        for name, (pipe, params) in self.get_models().items():

            with mlflow.start_run(run_name=name):

                print(f"Training {name}")

                grid = GridSearchCV(
                    pipe,
                    params,
                    cv=cv,
                    scoring=scoring,
                    refit="bal_acc",
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                idx = grid.best_index_

                cv_recall = grid.cv_results_["mean_test_recall"][idx]
                cv_f1 = grid.cv_results_["mean_test_f1"][idx]
                cv_bal_acc = grid.cv_results_["mean_test_bal_acc"][idx]
                cv_auc = grid.cv_results_["mean_test_auc"][idx]
                cv_acc = grid.cv_results_["mean_test_accuracy"][idx]

                mlflow.log_param("candidate_model", name)
                mlflow.log_param("data_version", data_hash)
                mlflow.log_params(grid.best_params_)

                mlflow.log_metric("cv_recall", cv_recall)
                mlflow.log_metric("cv_f1", cv_f1)
                mlflow.log_metric("cv_balanced_accuracy", cv_bal_acc)
                mlflow.log_metric("cv_auc", cv_auc)
                mlflow.log_metric("cv_accuracy", cv_acc)

                # # ⭐ MOST IMPORTANT → log candidate model
                # mlflow.sklearn.log_model(
                #     sk_model=grid.best_estimator_,
                #     name="model"
                # )
                with tempfile.TemporaryDirectory() as tmp_dir:

                    model_path = os.path.join(tmp_dir, "model.joblib")
                    joblib.dump(grid.best_estimator_, model_path)

                    mlflow.log_artifact(model_path, artifact_path="model")

                print(f"{name} CV Bal Acc: {cv_bal_acc:.3f} Recall: {cv_recall:.3f} F1: {cv_f1:.3f} AUC: {cv_auc:.3f} Accuracy: {cv_acc:.3f}")