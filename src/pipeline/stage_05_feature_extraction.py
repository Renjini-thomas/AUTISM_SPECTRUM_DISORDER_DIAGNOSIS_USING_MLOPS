from src.components.feature_extraction import FeatureExtraction

class FeatureExtractionPipeline:

    def main(self):

        obj = FeatureExtraction()
        obj.run()

if __name__ == "__main__":
    pipeline = FeatureExtractionPipeline()
    pipeline.main()