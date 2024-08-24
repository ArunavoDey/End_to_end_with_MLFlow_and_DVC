from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "data_ingestion_stage"
if __name__=="__main__":
    try:
        logger.info("Welcome to cnnClassifier")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<<\n\nx========x\n")
    except Exception as e:
        logger.exception(e)
        raise e
