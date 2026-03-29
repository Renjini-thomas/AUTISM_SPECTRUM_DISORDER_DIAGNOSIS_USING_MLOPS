from src.components.model_training import ModelTrainer
from src.utils.logger import logging

class ModelTrainingPipeline:

    def main(self):
        logging.info("Stage 07: Model Training Started")

        obj = ModelTrainer()
        obj.run()

        logging.info("Stage 07: Model Training Completed")

if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.main()