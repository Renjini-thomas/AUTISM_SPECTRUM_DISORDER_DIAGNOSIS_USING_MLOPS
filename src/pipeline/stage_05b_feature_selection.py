from src.components.feature_selection import FeatureSelection


class FeatureSelectionPipeline:

    def main(self):

        obj = FeatureSelection()
        obj.run()


if __name__ == "__main__":
    pipeline = FeatureSelectionPipeline()
    pipeline.main()