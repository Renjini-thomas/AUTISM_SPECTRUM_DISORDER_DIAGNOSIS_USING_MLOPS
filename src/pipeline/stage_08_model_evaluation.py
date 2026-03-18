from src.components.model_evaluation import ModelEvaluation

class ModelEvaluationPipeline:

    def main(self):

        obj = ModelEvaluation()
        obj.evaluate()

if __name__ == "__main__":
    pipeline = ModelEvaluationPipeline()
    pipeline.main()