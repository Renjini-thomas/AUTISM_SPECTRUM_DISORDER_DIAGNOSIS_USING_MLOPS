from src.monitoring.drift_detection import DriftDetection
from src.utils.logger import logging
class DataDriftDetectionPipeline:


    def main(self):
        logging.info("Stage 06: Data Drift Detection Started")

        obj = DriftDetection()
        obj.run()   
        logging.info("Stage 06: Data Drift Detection Completed")
if __name__ == "__main__":
    pipeline = DataDriftDetectionPipeline()
    pipeline.main()
