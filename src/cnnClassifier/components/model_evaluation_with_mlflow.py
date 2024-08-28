import os
from urllib.request import Request
from zipfile import ZipFile
import tensorflow as tf
import time
import dagshub
import mlflow
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from pathlib import Path
from cnnClassifier.utils.common import save_json
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.30
        )

        dataflow_kwargs = dict(
            target_size= self.config.params_image_size[:-1],
            batch_size= self.config.params_batch_size,
            interpolation = "bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator= valid_datagenerator.flow_from_directory(
            directory= self.config.training_data,
            subset= "validation",
            shuffle= False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model= self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        scores= {"loss": self.score[0], "accuracy":self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_intto_mlflow(self):
        dagshub.init(repo_owner='ArunavoDey', repo_name='End_to_end_with_MLFlow_and_DVC', mlflow=True)
        mlflow.set_registry_uri(self.config.mlflow_url)
        tracking_url_type_store= urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            #k = dict(self.config.all_params)
            #l = list(k.keys())
            #v = list(k.values())
            #print(f"l {l}")
            #l = [0.01, 0.001, 0.0001, 0.1]
            #for i in range(len(l)):
            mlflow.log_param('LEARNING_RATE',0.001)
            mlflow.log_metric("accuracy",1)
            #mlflow.log_param(self.config.all_params)
            #mlflow.log_metric({"loss": self.score[0], "accuracy":self.score[1]})
        if tracking_url_type_store != "file":
            mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            mlflow.keras.log_model(self.model, "model")
        else:
            mlflow.keras.log_model(self.model, "model")


