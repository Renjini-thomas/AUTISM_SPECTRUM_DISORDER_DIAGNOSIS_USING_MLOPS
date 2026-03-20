import numpy as np
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import os
from sklearn.metrics import make_scorer, recall_score

class ModelTrainer:

    def __init__(self):
        load_dotenv()
        username = os.getenv("DAGSHUB_USERNAME")
        token = os.getenv("DAGSHUB_TOKEN")

        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        self.scaled_dir = Path("artifacts/selected_features")
        self.model_dir = Path("artifacts/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri("https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow")
        mlflow.set_experiment("ASD_CLASSICAL_ML")

    def load_data(self):

        X_train = np.load(self.scaled_dir / "X_train.npy")
        y_train = np.load(self.scaled_dir / "y_train.npy", allow_pickle=True)

        X_test = np.load(self.scaled_dir / "X_test.npy")
        y_test = np.load(self.scaled_dir / "y_test.npy", allow_pickle=True)

        return X_train, y_train, X_test, y_test

    def get_models(self):

        return {

            "knn": (
                KNeighborsClassifier(),
                {"n_neighbors": [5,7]}
            ),

            "svm": (   # ⭐ FAST LINEAR SVM
                LinearSVC(
                    C=1,
                    class_weight="balanced",
                    max_iter=5000
                ),
                {}
            ),

            "decision Tree": (
                DecisionTreeClassifier(),
                {"max_depth":[3,5,10]}
            ),

            "random forest": (
                RandomForestClassifier(),
                {"n_estimators":[100,200], "max_depth":[5,10]}
            ),

            "gradient boosting": (
                GradientBoostingClassifier(),
                {"learning_rate":[0.01,0.1], "n_estimators":[100]}
            ),

            "logistic regression": (
                Pipeline([
                    
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
                ]),
                {"model__C":[0.1,1,10]}
            )
        }
    def run(self):

        X_train, y_train, X_test, y_test = self.load_data()

        best_model = None
        best_recall = 0
        best_f1 = 0
        best_auc = 0
        best_name = ""

        model_complexity_rank = {
            "logistic regression": 1,
            "knn": 2,
            "decision Tree": 3,
            "svm": 4,
            "random forest": 5,
            "gradient boosting": 6
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        for name, (model, params) in self.get_models().items():

            with mlflow.start_run(run_name=name):

                print(f"Training {name}")

                grid = GridSearchCV(
                    model,
                    params,
                    cv=cv,
                    scoring="roc_auc",   # ⭐ VERY IMPORTANT CHANGE
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                best_estimator = grid.best_estimator_

                y_pred = best_estimator.predict(X_test)

                recall = recall_score(y_test, y_pred, pos_label="autism")
                f1 = f1_score(y_test, y_pred, pos_label="autism")
                acc = accuracy_score(y_test, y_pred)

                try:
                    proba = best_estimator.predict_proba(X_test)
                    autistic_index = list(best_estimator.classes_).index("autism")
                    y_prob = proba[:, autistic_index]

                    auc = roc_auc_score(
                        (y_test == "autism").astype(int),
                        y_prob
                    )
                except:
                    auc = 0

                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("auc", auc)

                mlflow.sklearn.log_model(best_estimator, name="model")

                print(f"{name} Recall:", recall)

                # ⭐ BEST MODEL SELECTION RULE
                if recall > best_recall:

                    best_model = best_estimator
                    best_recall = recall
                    best_f1 = f1
                    best_auc = auc
                    best_name = name

                elif recall == best_recall:

                    if f1 > best_f1:

                        best_model = best_estimator
                        best_f1 = f1
                        best_auc = auc
                        best_name = name

                    elif f1 == best_f1:

                        if auc > best_auc:

                            best_model = best_estimator
                            best_auc = auc
                            best_name = name

                        elif auc == best_auc:

                            if model_complexity_rank[name] < model_complexity_rank[best_name]:

                                best_model = best_estimator
                                best_name = name

        joblib.dump(best_model, self.model_dir / "best_model.joblib")

        print("BEST MODEL:", best_name)
        print("Recall:", best_recall)
        print("F1:", best_f1)

        print("Model saved ✅")