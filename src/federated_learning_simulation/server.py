"""Start a Flower server.

Derived from Flower Android example.
"""
import pathlib
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvgAndroid

from typing import Callable, Dict, List, Optional, Tuple, Union, cast
from flwr.common import Metrics , EvaluateRes, NDArrays, Scalar, FitRes, Parameters, EvaluateIns, FitIns, NDArray
import flwr.common
from flwr.server.client_proxy import ClientProxy
import flwr.server.strategy
import numpy as np
import time
import tensorflow as tf

DATASET_PATH = '/Users/michalsh/Library/CloudStorage/OneDrive-Personal/Dokumenty/Uczelnia/SEM23_24_ZIM/Topic-research/Project/database/test_data_spectograms'
data_dir = pathlib.Path(DATASET_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    seed=0,
    image_size=(496, 369),
    labels='inferred',
    label_mode="categorical")

def fit_config(server_round: int):
    config = {
        "local_epochs": 30,
    }
    return config

def get_evaluate_fn(model, test_ds):
    """Return an evaluation function for server-side evaluation."""


    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        model.save('/Users/michalsh/Library/CloudStorage/OneDrive-Personal/Dokumenty/Uczelnia/SEM23_24_ZIM/Topic-research/Project/sim_srv_cli/Models/model_sc1.keras')
        loss, accuracy, p, r, f1 = model.evaluate(test_ds)
        return loss, {"accuracy": accuracy, "precision": p, "recall": r, "f1_score": f1}

    return evaluate
len_of_x = int(tf.data.experimental.cardinality(test_ds).numpy())

norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(data=test_ds.map(map_func=lambda spec, label: spec))
# Load and compile model for server-side parameter evaluation
model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(496, 369, 3)),
                    # Downsample the input.
                    tf.keras.layers.Resizing(32, 32),
                    # Normalize.
                    # tf.keras.layers.Normalization(),
                    norm_layer,
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(4, activation="softmax"),
                    ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy",
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.F1Score()
                       ])


# Create strategy
strategy = flwr.server.strategy.FedAvg(
    # ... other FedAvg arguments
    evaluate_fn=get_evaluate_fn(model, test_ds),
    on_fit_config_fn=fit_config,
    min_fit_clients=9,
    min_evaluate_clients=9,
    min_available_clients=9,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)

start_server(config=ServerConfig(num_rounds=10), strategy=strategy)
