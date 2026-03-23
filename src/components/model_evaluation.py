# import numpy as np
# import joblib
# import mlflow
# import mlflow.sklearn
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# from pathlib import Path
# from dotenv import load_dotenv

# from sklearn.metrics import (
#     recall_score,
#     f1_score,
#     accuracy_score,
#     roc_auc_score,
#     confusion_matrix,
#     classification_report,
#     RocCurveDisplay
# )


# class ModelEvaluation:

#     def __init__(self):

#         load_dotenv()

#         os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

#         mlflow.set_tracking_uri(
#             "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
#         )

#         mlflow.set_experiment("ASD_CLASSICAL_ML")

#         # ⭐ IMPORTANT → RAW SELECTED FEATURES (NOT SCALED)
#         self.feature_dir = Path("artifacts/selected_features")

#         self.model_dir = Path("artifacts/models")
#         self.eval_dir = Path("artifacts/evaluation")
#         self.eval_dir.mkdir(parents=True, exist_ok=True)

#     def load_data(self):

#         X_test = np.load(self.feature_dir / "X_test.npy")
#         y_test = np.load(self.feature_dir / "y_test.npy", allow_pickle=True)

#         model = joblib.load(self.model_dir / "best_model.joblib")

#         return X_test, y_test, model

#     def evaluate(self):

#         X_test, y_test, model = self.load_data()

#         with mlflow.start_run(run_name="evaluation_stage"):

#             # ================= MODEL INFO LOGGING =================

#             model_name = model.__class__.__name__

#             if hasattr(model, "named_steps"):

#                 final_estimator = model.named_steps["model"]
#                 model_name = final_estimator.__class__.__name__

#                 if "scaler" in model.named_steps:
#                     mlflow.log_param("scaler_used", "StandardScaler")

#                 params = final_estimator.get_params()

#             else:
#                 params = model.get_params()

#             mlflow.log_param("model_name", model_name)

#             for k, v in params.items():
#                 if isinstance(v, (int, float, str, bool)):
#                     mlflow.log_param(f"model__{k}", v)

#             # ================= PREDICTION =================

#             y_pred = model.predict(X_test)

#             recall = recall_score(y_test, y_pred, pos_label="autism")
#             f1 = f1_score(y_test, y_pred, pos_label="autism")
#             acc = accuracy_score(y_test, y_pred)

#             if hasattr(model, "predict_proba"):

#                 proba = model.predict_proba(X_test)

#                 autistic_index = list(model.classes_).index("autism")

#                 y_prob = proba[:, autistic_index]

#             elif hasattr(model, "decision_function"):

#                 scores = model.decision_function(X_test)

#                 # scale scores to 0-1
#                 y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

#             else:

#                 print("Model has no probability or decision score")
#                 y_prob = np.zeros(len(X_test))

#             auc = roc_auc_score(
#                 (y_test == "autism").astype(int),
#                 y_prob
#             )

#             mlflow.log_metric("eval_recall", recall)
#             mlflow.log_metric("eval_f1", f1)
#             mlflow.log_metric("eval_accuracy", acc)
#             mlflow.log_metric("eval_auc", auc)

#             # ================= CONFUSION MATRIX =================

#             class_order = ["autism", "control"]
#             cm = confusion_matrix(y_test, y_pred, labels=class_order)

#             display_labels = ["autism", "control"]

#             plt.figure(figsize=(6,5))

#             sns.heatmap(
#                 cm,
#                 annot=True,
#                 fmt="d",
#                 cmap="Blues",
#                 xticklabels=display_labels,
#                 yticklabels=display_labels
#             )

#             plt.xlabel("Predicted Label")
#             plt.ylabel("True Label")
#             plt.title("Confusion Matrix")

#             cm_path = self.eval_dir / "confusion_matrix.png"

#             plt.tight_layout()
#             plt.savefig(cm_path)
#             plt.close()

#             mlflow.log_artifact(str(cm_path))

#             # ================= ROC CURVE =================

#             plt.figure()

#             RocCurveDisplay.from_predictions(
#                 (y_test == "autism").astype(int),
#                 y_prob
#             )

#             roc_path = self.eval_dir / "roc_curve.png"

#             plt.tight_layout()
#             plt.savefig(roc_path)
#             plt.close()

#             mlflow.log_artifact(str(roc_path))

#             # ================= CLASSIFICATION REPORT =================

#             report = classification_report(y_test, y_pred)

#             report_path = self.eval_dir / "classification_report.txt"

#             with open(report_path, "w") as f:
#                 f.write(report)

#             mlflow.log_artifact(report_path)

#             print("Evaluation Metrics Logged")

#             # ================= GOVERNANCE =================

#             client = mlflow.tracking.MlflowClient()

#             staging_recall = 0

#             try:
#                 alias_info = client.get_model_version_by_alias(
#                     name="ASD_BEST_MODEL",
#                     alias="staging"
#                 )

#                 staging_run = client.get_run(alias_info.run_id)

#                 staging_recall = staging_run.data.metrics["eval_recall"]

#                 print("Previous staging Recall:", staging_recall)

#             except Exception:
#                 print("No staging model found")

#             # ================= PROMOTION =================

#             if recall > staging_recall:

#                 print("New model is BETTER → registering & promoting ✅")

#                 result = mlflow.sklearn.log_model(
#                     model,
#                     name="best_model",
#                     registered_model_name="ASD_BEST_MODEL"
#                 )

#                 version = result.registered_model_version

#                 client.set_registered_model_alias(
#                     name="ASD_BEST_MODEL",
#                     alias="staging",
#                     version=version
#                 )

#             else:

#                 print("New model WORSE → NOT registering ❌")
# import numpy as np
# import pandas as pd
# import joblib
# import mlflow
# import mlflow.sklearn
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# from pathlib import Path
# from dotenv import load_dotenv

# from sklearn.metrics import (
#     recall_score,
#     f1_score,
#     accuracy_score,
#     roc_auc_score,
#     confusion_matrix,
#     classification_report,
#     RocCurveDisplay,
#     balanced_accuracy_score,
#     precision_score
# )


# class ModelEvaluation:

#     def __init__(self):

#         load_dotenv()

#         os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

#         mlflow.set_tracking_uri(
#             "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
#         )

#         mlflow.set_experiment("ASD_HYBRID_BLOCK_PCA_ALL_MODELS")

#         # ⭐ NOW FEATURES ARE FINAL
#         self.feature_dir = Path("artifacts/features")

#         self.model_dir = Path("artifacts/models")
#         self.eval_dir = Path("artifacts/evaluation")
#         self.eval_dir.mkdir(parents=True, exist_ok=True)

#     # ---------------- LOAD ----------------

#     def load_data(self):

#         test_df = pd.read_csv(self.feature_dir / "test_features.csv")

#         X_test = test_df.drop("label", axis=1)
#         y_test = test_df["label"].values

#         model = joblib.load(self.model_dir / "best_model.joblib")

#         return X_test, y_test, model

#     # ---------------- EVALUATE ----------------

#     def evaluate(self):

#         X_test, y_test, model = self.load_data()

#         with mlflow.start_run(run_name="evaluation_stage"):

#             # ===== MODEL INFO =====

#             final_estimator = model.named_steps["model"]
#             model_name = final_estimator.__class__.__name__

#             mlflow.log_param("model_name", model_name)

#             if "pca" in model.named_steps:
#                 mlflow.log_param(
#                     "pca_n_components",
#                     model.named_steps["pca"].n_components
#                 )

#             if "scaler" in model.named_steps:
#                 mlflow.log_param("scaler", "StandardScaler")
            

#             # ===== PREDICTION =====

#             y_pred = model.predict(X_test)
            

#             recall = recall_score(y_test, y_pred, pos_label="autism")
#             f1 = f1_score(y_test, y_pred, pos_label="autism")
#             acc = accuracy_score(y_test, y_pred)
#             bal_acc = balanced_accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred, pos_label="autism")
            
#             # ===== PROBABILITY =====

#             y_prob = model.predict_proba(X_test)[
#                 :, list(model.classes_).index("autism")
#             ]

#             auc = roc_auc_score(
#                 (y_test == "autism").astype(int),
#                 y_prob
#             )

#             # ===== LOG METRICS =====

#             mlflow.log_metric("eval_recall", recall)
#             mlflow.log_metric("eval_f1", f1)
#             mlflow.log_metric("eval_accuracy", acc)
#             mlflow.log_metric("eval_auc", auc)
#             mlflow.log_metric("eval_balanced_accuracy", bal_acc)
#             mlflow.log_metric("eval_precision", precision)

#             # ===== CONFUSION MATRIX =====

#             class_order = ["autism", "control"]
#             cm = confusion_matrix(y_test, y_pred, labels=class_order)

#             plt.figure(figsize=(6,5))

#             sns.heatmap(
#                 cm,
#                 annot=True,
#                 fmt="d",
#                 cmap="Blues",
#                 xticklabels=class_order,
#                 yticklabels=class_order
#             )

#             plt.xlabel("Predicted")
#             plt.ylabel("True")
#             plt.title("Confusion Matrix")

#             cm_path = self.eval_dir / "confusion_matrix.png"

#             plt.tight_layout()
#             plt.savefig(cm_path)
#             plt.close()

#             mlflow.log_artifact(str(cm_path))

#             # ===== ROC =====

#             plt.figure()

#             RocCurveDisplay.from_predictions(
#                 (y_test == "autism").astype(int),
#                 y_prob
#             )

#             roc_path = self.eval_dir / "roc_curve.png"

#             plt.tight_layout()
#             plt.savefig(roc_path)
#             plt.close()

#             mlflow.log_artifact(str(roc_path))

#             # ===== REPORT =====

#             report = classification_report(y_test, y_pred)

#             report_path = self.eval_dir / "classification_report.txt"

#             with open(report_path, "w") as f:
#                 f.write(report)

#             mlflow.log_artifact(report_path)
#             print("Evaluation Metrics Logged")

#             # ================= CLASSIFICATION REPORT =================

#             report = classification_report(y_test, y_pred)

#             report_path = self.eval_dir / "classification_report.txt"

#             with open(report_path, "w") as f:
#                 f.write(report)

#             mlflow.log_artifact(str(report_path))

#             print("Evaluation Metrics Logged")

#             # ================= GOVERNANCE & PROMOTION =================    
#             # ================= GOVERNANCE =================

#             client = mlflow.tracking.MlflowClient()

#             staging_bal_acc = 0

#             try:
#                 alias_info = client.get_model_version_by_alias(
#                     name="ASD_BEST_MODEL",
#                     alias="staging"
#                 )

#                 staging_run = client.get_run(alias_info.run_id)

#                 staging_bal_acc = staging_run.data.metrics["eval_balanced_accuracy"]

#                 print("Previous staging Balanced Accuracy:", staging_bal_acc)

#             except Exception:
#                 print("No staging model found")

#             # ================= PROMOTION =================

#             if bal_acc > staging_bal_acc:

#                 print("New model is BETTER → registering & promoting ✅")

#                 result = mlflow.sklearn.log_model(
#                     model,
#                     name="best_model",
#                     registered_model_name="ASD_BEST_MODEL"
#                 )

#                 version = result.registered_model_version

#                 client.set_registered_model_alias(
#                     name="ASD_BEST_MODEL",
#                     alias="staging",
#                     version=version
#                 )

#             else:

#                 print("New model WORSE → NOT registering ❌")
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pathlib import Path
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    balanced_accuracy_score,
    precision_score
)


class ModelEvaluation:

    def __init__(self):

        load_dotenv()

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

        mlflow.set_tracking_uri(
            "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
        )

        mlflow.set_experiment("ASD_MLOPS_PIPELINE")

        self.feature_dir = Path("artifacts/features")
        self.eval_dir = Path("artifacts/evaluation")
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    # ⭐ GET BEST CANDIDATE RUN FROM TRAINING

    def get_best_candidate_run(self):

        client = MlflowClient()

        exp = client.get_experiment_by_name("ASD_MLOPS_PIPELINE")

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.cv_balanced_accuracy DESC"]
        )

        best_run = runs[0]

        print("⭐ Best Candidate Run:", best_run.info.run_id)

        return best_run.info.run_id

    # ⭐ LOAD TEST DATA + LOAD MODEL FROM MLFLOW RUN

    def load_data(self):

        # test_df = pd.read_csv(self.feature_dir / "test_features.csv")

        # X_test = test_df.drop("label", axis=1)
        # y_test = test_df["label"].values

        # run_id = self.get_best_candidate_run()

        # model_uri = f"runs:/{run_id}/model"

        # model = mlflow.pyfunc.load_model(model_uri)

        # return X_test, y_test, model, run_id
        import joblib
        import tempfile
        test_df = pd.read_csv(self.feature_dir / "test_features.csv")
        X_test = test_df.drop("label", axis=1)
        y_test = test_df["label"].values

        run_id = self.get_best_candidate_run()
        client = MlflowClient()
        with tempfile.TemporaryDirectory() as tmp_dir:
            load_dir = client.download_artifacts(run_id, "model", dst_path=tmp_dir)
            model_path = os.path.join(load_dir, "model.joblib")
            model = joblib.load(model_path)
        return X_test, y_test, model, run_id

    # ⭐ EVALUATION

    def evaluate(self):

        X_test, y_test, model, run_id = self.load_data()

        with mlflow.start_run(run_name="evaluation_stage"):

            mlflow.log_param("candidate_run_id", run_id)

            # ===== PREDICTIONS =====

            y_pred = model.predict(X_test)

            recall = recall_score(y_test, y_pred, pos_label="autism")
            f1 = f1_score(y_test, y_pred, pos_label="autism")
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label="autism")

            y_prob = model.predict_proba(X_test)[
                :, list(model.classes_).index("autism")
            ]

            auc = roc_auc_score(
                (y_test == "autism").astype(int),
                y_prob
            )
            mlflow.log_param("model_name", model.__class__.__name__)
            mlflow.log_param("model_params", model.get_params())
            mlflow.log_param("pca_n_components", model.named_steps["pca"].n_components)
            mlflow.log_param("scaler_used", "StandardScaler")
            # ===== LOG METRICS =====

            mlflow.log_metric("eval_recall", recall)
            mlflow.log_metric("eval_f1", f1)
            mlflow.log_metric("eval_accuracy", acc)
            mlflow.log_metric("eval_auc", auc)
            mlflow.log_metric("eval_balanced_accuracy", bal_acc)
            mlflow.log_metric("eval_precision", precision)

            # ===== CONFUSION MATRIX =====

            class_order = ["autism", "control"]

            cm = confusion_matrix(y_test, y_pred, labels=class_order)

            plt.figure(figsize=(6, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_order,
                yticklabels=class_order
            )

            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")

            cm_path = self.eval_dir / "confusion_matrix.png"

            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()

            mlflow.log_artifact(str(cm_path))

            # ===== ROC =====

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

            # ===== CLASSIFICATION REPORT =====

            report = classification_report(y_test, y_pred)

            report_path = self.eval_dir / "classification_report.txt"

            with open(report_path, "w") as f:
                f.write(report)

            mlflow.log_artifact(str(report_path))

            print("✅ Evaluation Metrics Logged")

            # ⭐ GOVERNANCE

            client = MlflowClient()

            staging_bal_acc = 0

            try:

                alias_info = client.get_model_version_by_alias(
                    name="ASD_BEST_MODEL",
                    alias="staging"
                )

                staging_run = client.get_run(alias_info.run_id)

                staging_bal_acc = staging_run.data.metrics[
                    "eval_balanced_accuracy"
                ]

                print("Previous Staging Balanced Accuracy:", staging_bal_acc)

            except Exception:

                print("No staging model found")

            # ⭐ PROMOTION

            if bal_acc > staging_bal_acc:

                print("⭐ NEW MODEL BETTER → Registering")

                result = mlflow.register_model(
                    f"runs:/{run_id}/model",
                    "ASD_BEST_MODEL"
                )

                client.set_registered_model_alias(
                    name="ASD_BEST_MODEL",
                    alias="staging",
                    version=result.version
                )

                print("✅ Promoted to STAGING")

            else:

                print("❌ New Model WORSE → Not Promoted")