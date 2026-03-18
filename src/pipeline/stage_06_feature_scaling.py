from src.components.feature_scaling import FeatureScaling


class FeatureScalingPipeline:

    def main(self):

        obj = FeatureScaling()
        obj.run()


if __name__ == "__main__":

    pipeline = FeatureScalingPipeline()
    pipeline.main()