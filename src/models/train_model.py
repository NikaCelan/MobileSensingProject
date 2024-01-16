# 2023, Hubert Michalski, Nika Celan
# # https://github.com/NikaCelan/MobileSensingProject
#  _   _       _                    _ _                 
# | | | |_ __ (_)_   _____ _ __ ___(_) |_ _   _         
# | | | | '_ \| \ \ / / _ \ '__/ __| | __| | | |        
# | |_| | | | | |\ V /  __/ |  \__ \ | |_| |_| |        
#  \___/|_|_|_|_| \_/ \___|_| _|___/_|\__|\__, |        
#   ___  / _| | |    (_)_   _| |__ (_) __ |___/_   __ _ 
#  / _ \| |_  | |    | | | | | '_ \| |/ _` | '_ \ / _` |
# | (_) |  _| | |___ | | |_| | |_) | | (_| | | | | (_| |
#  \___/|_|   |_____|/ |\__,_|_.__// |\__,_|_| |_|\__,_|
#                  |__/          |__/              
# -*- coding: utf-8 -*-
import tensorflow as tf
import models.mobilenet as mobilenet
from keras.layers import Layer

class ModelTraining():
    def __init__(self, train_ds, val_ds, test_ds) -> None:
        self.train_ds = train_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
        self.val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        self.test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
        # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.info)
    
    def prepare_model_settings(self, label_count, sample_rate = 16000, clip_duration_ms = 1000,
                               window_size_ms = 30, window_stride_ms = 10,
                               dct_coefficient_count = 40):
        """Calculates common settings needed for all models.

        Args:
            label_count: How many classes are to be recognized.
            sample_rate: Number of audio samples per second.
            clip_duration_ms: Length of each audio clip to be analyzed.
            window_size_ms: Duration of frequency analysis window.
            window_stride_ms: How far to move in time between frequency windows.
            dct_coefficient_count: Number of frequency bins to use for analysis.

        Returns:
            Dictionary containing common settings.
        """
        desired_samples = int(sample_rate * clip_duration_ms / 1000)
        window_size_samples = int(sample_rate * window_size_ms / 1000)
        window_stride_samples = int(sample_rate * window_stride_ms / 1000)
        length_minus_window = (desired_samples - window_size_samples)
        if length_minus_window < 0:
            spectrogram_length = 0
        else:
            spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
        fingerprint_size = dct_coefficient_count * spectrogram_length
        return {
            'desired_samples': desired_samples,
            'window_size_samples': window_size_samples,
            'window_stride_samples': window_stride_samples,
            'spectrogram_length': spectrogram_length,
            'dct_coefficient_count': dct_coefficient_count,
            'fingerprint_size': fingerprint_size,
            'label_count': label_count,
            'sample_rate': sample_rate,
        }
    
    def train_model(self, model_settings):
        fingerprint_size = model_settings['fingerprint_size']
        label_count = model_settings['label_count']
        fingerprint_input = tf.keras.layers.Input(
            shape=(fingerprint_size,), dtype=tf.float32, name='fingerprint_input')
        logits, dropout_prob = self._create_model(
            fingerprint_input, model_settings, is_training=True)
        #define loss and optimizer
        ground_truth_input = tf.keras.layers.Input(
            dtype=tf.float32, shape=(label_count,), name='groundtruth_input')
          # Create the back propagation and training evaluation machinery in the graph.
        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=logits))
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        with tf.name_scope('train'):
            learning_rate_input = tf.keras.layers.Input(dtype=tf.float32, shape=[], name='learning_rate_input')
            momentum =  tf.keras.layers.Input(dtype=tf.float32, shape=[], name='momentum')
            # optimizer
            # train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(cross_entropy_mean)
            # train_step = tf.train.MomentumOptimizer(learning_rate_input, momentum, use_nesterov=True).minimize(cross_entropy_mean)
            # train_step = tf.train.AdamOptimizer(learning_rate_input).minimize(cross_entropy_mean)
            train_step = tf.train.RMSPropOptimizer(learning_rate_input, momentum).minimize(cross_entropy_mean)

    def _create_model(self, fingerprint_input, model_settings, is_training=True):
        input_frequency_size = model_settings['dct_coefficient_count']
        input_time_size = model_settings['spectrogram_length']
        fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_time_size,
                                                        input_frequency_size, 1])
        # modify_fingerprint_4d = tf.image.resize_bilinear(fingerprint_4d, [224, 224])
        resize_layer = ResizeLayer(target_size=[224, 224])
        modify_fingerprint_4d = resize_layer(fingerprint_4d)
        # modify_fingerprint_4d =tf.image.resize(fingerprint_4d, size=[224, 224], method=tf.image.ResizeMethod.BILINEAR)

        logits, end_points = mobilenet.mobilenet(modify_fingerprint_4d, model_settings['label_count'])
        return logits


class ResizeLayer(Layer):
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs, **kwargs):
        return tf.image.resize(inputs, size=self.target_size, method=tf.image.ResizeMethod.BILINEAR)