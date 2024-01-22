import tensorflow as tf

class KeywordRecognitionModel():

    def __init__(self, lr=0.0001):
        self.model = tf.keras.Sequential([
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


        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy",
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.F1Score()])
        
    def get_weights(self):
        return self.model.get_weights()
    
    def get_weights(self):
        return self.model.get_weights()