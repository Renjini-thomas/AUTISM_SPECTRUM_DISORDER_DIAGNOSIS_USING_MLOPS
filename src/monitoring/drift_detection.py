# src/monitoring/drift_detection.py

import pandas as pd
import json
import os
import shutil
import mlflow
from pathlib import Path
from dotenv import load_dotenv

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric


class DriftDetection:

    def __init__(self):

        load_dotenv()

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

        mlflow.set_tracking_uri(
            "https://dagshub.com/renjini2539thomas/AUTISM_SPECTRUM_DISORDER_DIAGNOSIS_USING_MLOPS.mlflow"
        )
        mlflow.set_experiment("ASD_MLOPS_PIPELINE")

        self.feature_dir  = Path("artifacts/features")
        self.report_dir   = Path("artifacts/drift")
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Retrain if >30% of raw DenseNet features have drifted
        self.drift_threshold = 0.30

    # ── LOAD DATA ──────────────────────────────────────────────
    def load_data(self):

        reference_path = self.feature_dir / "reference_features.csv"
        current_path   = self.feature_dir / "train_features.csv"

        # ⭐ First ever run — no reference exists yet.
        # Copy current as reference and force pipeline to run
        # so a model gets registered against a known baseline.
        if not reference_path.exists():
            shutil.copy(current_path, reference_path)
            print("No reference found — first run detected.")
            print("Reference saved. Forcing full pipeline run.")
            self._write_output(drift_detected=True, score=1.0)
            return None, None

        # ⭐ Load RAW DenseNet features — no scaler, no PCA.
        #
        # Why raw features?
        # Your sklearn Pipeline's StandardScaler is fitted on training data.
        # Applying it to new incoming data subtracts the training mean —
        # which is exactly what drift looks like. A shifted mean in the
        # new batch gets forced back to zero, hiding the drift from Evidently.
        # PCA has the same problem: it projects new data onto training-era
        # components, hiding variation in directions the training PCA
        # didn't capture.
        #
        # Drift must be detected before any transformation is applied.
        # The scaler/PCA inside model_training.py are for the model only.

        reference_df = pd.read_csv(reference_path).drop("label", axis=1)
        current_df   = pd.read_csv(current_path).drop("label", axis=1)

        print(f"Reference shape : {reference_df.shape}")
        print(f"Current shape   : {current_df.shape}")

        if reference_df.shape[1] != current_df.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: reference has {reference_df.shape[1]} "
                f"columns but current has {current_df.shape[1]}. "
                "This usually means feature_extraction.py changed. "
                "Update the reference manually after verifying the new features."
            )

        return reference_df, current_df

    # ── MAIN RUN ───────────────────────────────────────────────
    def run(self):

        reference_df, current_df = self.load_data()

        # First-run case — output already written in load_data()
        if reference_df is None:
            return True

        # ── EVIDENTLY REPORT ───────────────────────────────────
        # Evidently runs its own statistical tests per feature
        # (KS test for numerical columns by default).
        # Since our features are raw L2-normalised DenseNet embeddings,
        # all columns are numerical — KS test is the right choice.
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(drift_share=self.drift_threshold)
        ])

        report.run(
            reference_data=reference_df,
            current_data=current_df
        )

        # ── SAVE HTML REPORT ───────────────────────────────────
        report_path = self.report_dir / "drift_report.html"
        report.save_html(str(report_path))

        # ── EXTRACT METRICS ────────────────────────────────────
        report_dict    = report.as_dict()
        dataset_result = report_dict["metrics"][1]["result"]

        drift_detected = dataset_result.get("dataset_drift", False)

        share_drifted_cols = dataset_result.get(
            "share_drifted_columns",
            dataset_result.get("drift_share", 0.0)
        )

        n_drifted = dataset_result.get(
            "number_of_drifted_columns",
            dataset_result.get("n_drifted_columns", 0)
        )

        n_total = dataset_result.get(
            "number_of_columns",
            dataset_result.get("n_columns", 0)
        )

        print(f"Features drifted : {n_drifted} / {n_total}")
        print(f"Share drifted    : {share_drifted_cols:.2%}")
        print(f"Drift detected   : {drift_detected}")

        # ── SAVE JSON SUMMARY ──────────────────────────────────
        metrics_summary = {
            "drift_detected"     : drift_detected,
            "share_drifted_cols" : round(share_drifted_cols, 4),
            "n_drifted_columns"  : n_drifted,
            "total_columns"      : n_total,
            "threshold_used"     : self.drift_threshold,
            "note"               : "Drift computed on raw DenseNet features before scaler/PCA"
        }

        metrics_path = self.report_dir / "drift_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)

        # ── LOG TO MLFLOW ──────────────────────────────────────
        with mlflow.start_run(run_name="drift_detection"):

            mlflow.log_metric("share_drifted_columns", share_drifted_cols)
            mlflow.log_metric("n_drifted_columns",     n_drifted)
            mlflow.log_param("drift_detected",         drift_detected)
            mlflow.log_param("drift_threshold",        self.drift_threshold)
            mlflow.log_param("drift_on",               "raw_densenet_features")
            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(str(metrics_path))

        # ── WRITE GITHUB ACTIONS OUTPUT ────────────────────────
        self._write_output(drift_detected, share_drifted_cols)

        return drift_detected

    def _write_output(self, drift_detected, score):
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"DRIFT_DETECTED={'true' if drift_detected else 'false'}\n")
                f.write(f"DRIFT_SCORE={score:.4f}\n")