import os
from urllib.request import Request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model= tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.20
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
            class_mode="categorical"
            #**dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            """
            train_datagenerator= tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range= 40,
                horizontal_flip= True,
                width_shift_range= 0.2,
                height_shift_range= 0.2,
                shear_range= 0.2,
                zoom_range= 0.2,
                **datagenerator_kwargs
            )
            """
            train_datagenerator =  tf.keras.preprocessing.image. ImageDataGenerator(
                rescale=1./255,  # Rescale the pixel values
                rotation_range=40,  # Rotate images by up to 40 degrees
                width_shift_range=0.3,  # Shift the image width by up to 30% of the image width
                height_shift_range=0.3,  # Shift the image height by up to 30% of the image height
                shear_range=0.3,  # Shear the image by up to 30%
                zoom_range=0.3,  # Zoom in/out by up to 30%
                horizontal_flip=True,  # Randomly flip images horizontally
                vertical_flip=True,  # Randomly flip images vertically
                brightness_range=[0.8, 1.2],  # Adjust the brightness randomly within this range
                channel_shift_range=0.2,  # Randomly shift the color channels
                fill_mode='nearest',  # How to fill in newly created pixels, nearest replicates the nearest pixel
                preprocessing_function=None  # You can add a custom function here for even more complex augmentations
                )
        else:
            train_datagenerator = valid_datagenerator
        self.train_generator= train_datagenerator.flow_from_directory(
            directory= self.config.training_data,
            subset= "training",
            shuffle= True,
            class_mode="categorical"
            #**dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self):
        self.steps_per_epoch= self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps= self.valid_generator.samples // self.valid_generator.batch_size

        self.model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.8),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ["accuracy"]
        )


        self.model.fit(
            self.train_generator,
            epochs= self.config.params_epochs,
            steps_per_epoch= self.steps_per_epoch,
            validation_steps= self.validation_steps,
            validation_data= self.valid_generator
        )

        self.save_model(
            path= self.config.trained_model_path,
            model= self.model
        )
