def test_app_import():
    import app
    assert app.app is not None


def test_prediction_import():
    from src.prediction.ASD_prediction import ASD_Prediction
    assert ASD_Prediction is not None


def test_pipeline_import():
    from src.pipeline.stage_07_model_training import main
    assert main is not None


def test_drift_import():
    from src.monitoring.drift_detection import DriftDetection
    assert DriftDetection is not None