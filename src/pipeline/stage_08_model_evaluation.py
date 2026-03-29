from src.components.model_evaluation import ModelEvaluation
from src.utils.logger import logging    
class ModelEvaluationPipeline:

    def main(self):
        logging.info("Stage 08: Model Evaluation Started")
        obj = ModelEvaluation()
        obj.evaluate()
        logging.info("Stage 08: Model Evaluation Completed")
if __name__ == "__main__":
    pipeline = ModelEvaluationPipeline()
    pipeline.main()