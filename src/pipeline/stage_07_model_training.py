from src.components.model_training import ModelTrainer


class ModelTrainingPipeline:

    def main(self):

        obj = ModelTrainer()
        obj.run()


if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.main()