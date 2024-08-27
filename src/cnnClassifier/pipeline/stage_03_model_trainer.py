from cnnClassifier.utils.common import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.components.model_trainer import Training

STAGE_NAME = "Model_training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        model_trainer = Training(config=training_config)
        model_trainer.get_base_model()
        model_trainer.train_valid_generator()
        model_trainer.train()


if __name__ =="__main__":
    try:
        logger.info(f"*************************")
        logger.info(">>>>> stage name {STAGE_NAME} started <<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e