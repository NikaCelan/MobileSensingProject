import flwr as fl
import tensorflow as tf

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from model import KeywordRecognitionModel

DATASET_PATH = '/Users/michalsh/Documents/[03]Nauka/project_topical_research/06-01-24/spectJPG/user03'
data_dir = pathlib.Path(DATASET_PATH)

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.2,
    seed=0,
    image_size=(496, 369),
    subset='both',
    labels='inferred',
    label_mode="categorical")

len_of_x = int(tf.data.experimental.cardinality(train_ds).numpy())

model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(496, 369, 3)),
        # Downsample the input.
        tf.keras.layers.Resizing(32, 32),
        # Normalize.
        tf.keras.layers.Normalization(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation="softmax"),
        ])


model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.F1Score()])

class Client3(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_ds, epochs=20)
        return model.get_weights(), len_of_x, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy, precision, recall, f1_score = model.evaluate(val_ds)
        return loss, len_of_x, {"accuracy": float(accuracy)}
    

fl.client.start_numpy_client(server_address="[::]:8080", client=Client3())