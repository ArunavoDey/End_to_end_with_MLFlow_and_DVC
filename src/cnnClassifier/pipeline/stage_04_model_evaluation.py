from cnnClassifier.utils.common import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.components.model_trainer import Training
from cnnClassifier.components.model_evaluation_with_mlflow import Evaluation

STAGE_NAME = "Model_evaluation"

class ModelEvaluationPipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        model_evaluation = Evaluation(config=evaluation_config)
        model_evaluation.evaluation()
        model_evaluation.log_intto_mlflow()

if __name__=="__main__":
    try:
        logger.info(f"*************************")
        logger.info(">>>>> stage name {STAGE_NAME} started <<<<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e
