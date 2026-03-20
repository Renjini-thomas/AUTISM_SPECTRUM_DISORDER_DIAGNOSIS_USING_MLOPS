import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pathlib import Path
from dotenv import load_dotenv

from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)


class ModelEvaluation:

    def __init__(self):

        load_dotenv()

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

        mlflow.set_tracking_uri(
            "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
        )

        mlflow.set_experiment("ASD_CLASSICAL_ML")

        # ⭐ IMPORTANT → RAW SELECTED FEATURES (NOT SCALED)
        self.feature_dir = Path("artifacts/selected_features")

        self.model_dir = Path("artifacts/models")
        self.eval_dir = Path("artifacts/evaluation")
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):

        X_test = np.load(self.feature_dir / "X_test.npy")
        y_test = np.load(self.feature_dir / "y_test.npy", allow_pickle=True)

        model = joblib.load(self.model_dir / "best_model.joblib")

        return X_test, y_test, model

    def evaluate(self):

        X_test, y_test, model = self.load_data()

        with mlflow.start_run(run_name="evaluation_stage"):

            # ================= MODEL INFO LOGGING =================

            model_name = model.__class__.__name__

            if hasattr(model, "named_steps"):

                final_estimator = model.named_steps["model"]
                model_name = final_estimator.__class__.__name__

                if "scaler" in model.named_steps:
                    mlflow.log_param("scaler_used", "StandardScaler")

                params = final_estimator.get_params()

            else:
                params = model.get_params()

            mlflow.log_param("model_name", model_name)

            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(f"model__{k}", v)

            # ================= PREDICTION =================

            y_pred = model.predict(X_test)

            recall = recall_score(y_test, y_pred, pos_label="autism")
            f1 = f1_score(y_test, y_pred, pos_label="autism")
            acc = accuracy_score(y_test, y_pred)

            if hasattr(model, "predict_proba"):

                proba = model.predict_proba(X_test)

                autistic_index = list(model.classes_).index("autism")

                y_prob = proba[:, autistic_index]

            elif hasattr(model, "decision_function"):

                scores = model.decision_function(X_test)

                # scale scores to 0-1
                y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

            else:

                print("Model has no probability or decision score")
                y_prob = np.zeros(len(X_test))

            auc = roc_auc_score(
                (y_test == "autism").astype(int),
                y_prob
            )

            mlflow.log_metric("eval_recall", recall)
            mlflow.log_metric("eval_f1", f1)
            mlflow.log_metric("eval_accuracy", acc)
            mlflow.log_metric("eval_auc", auc)

            # ================= CONFUSION MATRIX =================

            class_order = ["autism", "control"]
            cm = confusion_matrix(y_test, y_pred, labels=class_order)

            display_labels = ["autism", "control"]

            plt.figure(figsize=(6,5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=display_labels,
                yticklabels=display_labels
            )

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")

            cm_path = self.eval_dir / "confusion_matrix.png"

            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()

            mlflow.log_artifact(str(cm_path))

            # ================= ROC CURVE =================

            plt.figure()

            RocCurveDisplay.from_predictions(
                (y_test == "autism").astype(int),
                y_prob
            )

            roc_path = self.eval_dir / "roc_curve.png"

            plt.tight_layout()
            plt.savefig(roc_path)
            plt.close()

            mlflow.log_artifact(str(roc_path))

            # ================= CLASSIFICATION REPORT =================

            report = classification_report(y_test, y_pred)

            report_path = self.eval_dir / "classification_report.txt"

            with open(report_path, "w") as f:
                f.write(report)

            mlflow.log_artifact(report_path)

            print("Evaluation Metrics Logged")

            # ================= GOVERNANCE =================

            client = mlflow.tracking.MlflowClient()

            staging_recall = 0

            try:
                alias_info = client.get_model_version_by_alias(
                    name="ASD_BEST_MODEL",
                    alias="staging"
                )

                staging_run = client.get_run(alias_info.run_id)

                staging_recall = staging_run.data.metrics["eval_recall"]

                print("Previous staging Recall:", staging_recall)

            except Exception:
                print("No staging model found")

            # ================= PROMOTION =================

            if recall > staging_recall:

                print("New model is BETTER → registering & promoting ✅")

                result = mlflow.sklearn.log_model(
                    model,
                    name="best_model",
                    registered_model_name="ASD_BEST_MODEL"
                )

                version = result.registered_model_version

                client.set_registered_model_alias(
                    name="ASD_BEST_MODEL",
                    alias="staging",
                    version=version
                )

            else:

                print("New model WORSE → NOT registering ❌")