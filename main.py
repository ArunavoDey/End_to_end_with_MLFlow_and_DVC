from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline

STAGE_NAME_1 = "data_ingestion_stage"

STAGE_NAME_2 = "Prepare_base_model"

STAGE_NAME_3= "Model_training"

STAGE_NAME_4= "Model_Evaluation"


if __name__=="__main__":
    try:
        logger.info("Welcome to cnnClassifier")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME_1} completed <<<<<\n\nx========x\n")
    except Exception as e:
        logger.exception(e)
        raise e
    try:
        logger.info(f"*************************")
        logger.info(">>>>> stage name {STAGE_NAME_2} started <<<<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME_2} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        logger.info(f"*************************")
        logger.info(">>>>> stage name {STAGE_NAME_3} started <<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME_3} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e


    try:
        logger.info(f"*************************")
        logger.info(">>>>> stage name {STAGE_NAME_4} started <<<<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME_4} completed <<<<<<\n\nx============x")
    except Exception as e:
        logger.exception(e)
        raise e
