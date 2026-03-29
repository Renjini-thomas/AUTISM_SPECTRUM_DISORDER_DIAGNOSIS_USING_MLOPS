from src.components.feature_extraction import FeatureExtraction

class FeatureExtractionPipeline:

    def main(self):
        logging.info("Stage 05: Feature Extraction Started")
        obj = FeatureExtraction()
        obj.run()
        logging.info("Stage 05: Feature Extraction Completed")

if __name__ == "__main__":
    pipeline = FeatureExtractionPipeline()
    pipeline.main()