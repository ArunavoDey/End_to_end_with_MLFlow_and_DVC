from cnnClassifier.utils.common import logger
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

STAGE_NAME = "Prepare_base_model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ =="__main__":
    try:
        logger.info(f"*************************")
        logger.info(">>>>> stage name {STAGE_NAME} started <<<<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e