import tensorflow as tf
from tensorflow.keras import layers
from keras import Sequential
from load_dataset import LoadData
import flwr as fl
import matplotlib.pyplot as plt
import numpy as np

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
# norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))
data_process = LoadData()
model = Sequential([
    layers.Input(shape=(124, 129, 1)),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

train_ds, test_ds, val_ds = data_process.load_data("/Users/michalsh/Documents/[03]Nauka/project_topical_research/MobileSensingProject/data/users_data/user01")
train_spectrogram_ds = data_process.convert_to_spectrogram(train_ds)
val_spectrogram_ds = data_process.convert_to_spectrogram(val_ds)
test_spectrogram_ds = data_process.convert_to_spectrogram(test_ds)
x_train, y_train = train_spectrogram_ds
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break
rows = 3
cols = 3
n = rows*cols

fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)

plt.show()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_spectrogram_ds,
        epochs=10
        )
        return model.get_weights(), len(x_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_spectrogram_ds)
        return loss, len(x_train), {"accuracy": accuracy}
    
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
    