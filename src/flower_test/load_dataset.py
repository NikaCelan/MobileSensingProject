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

import logging
import tensorflow as tf
import numpy as np
import librosa


class LoadData():
    def __init__(self) -> None:
        self.seed = 42
        self.dataset_directory = "data/speech_commands_v0.01"
        self.commands = ['up', 'down', 'left', 'right']
        self.label_count = 0

    def load_data(self, directory):
        """
        Loads audio data from a specified directory and split it for training, validation, and testing datasets.

        Args:
            directory (str): The directory containing audio data.

        Returns:
            tuple: A tuple containing three datasets (train_ds, test_ds, val_ds):
                - train_ds (tf.data.Dataset): The training dataset.
                - test_ds (tf.data.Dataset): The testing dataset.
                - val_ds (tf.data.Dataset): The validation dataset.

        This function loads audio data from the specified directory, splits it into training, validation, and testing subsets,
        and returns them as TensorFlow datasets. It also logs label names and the element specification of the training dataset.

        Example:
            To load data from a custom directory:
            >>> train_ds, test_ds, val_ds = load_data(directory="data/speech_commands_v0.01")
        """

        train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=directory,
            batch_size = 64,
            validation_split=0.2,
            seed=self.seed,
            output_sequence_length=16000,
            subset='both'
        )
        label_names = np.array(train_ds.class_names)
        self.label_count = len(label_names)
        
        print("Label names:", label_names)
        print(train_ds.element_spec)

        train_ds = train_ds.map(self._squeeze, tf.data.AUTOTUNE)
        val_ds = val_ds.map(self._squeeze, tf.data.AUTOTUNE)

        test_ds = val_ds.shard(num_shards=2, index=0)
        val_ds = val_ds.shard(num_shards=2, index=1)
        
        return  train_ds, test_ds, val_ds



    def _squeeze(self, audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels
    
    def _get_spectrogram(self, waveform):
        spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram
    
    def convert_to_spectrogram(self, data_set):
        """
        Converts audio data to spectrograms.

        Args:
        data_set (tf.data.Dataset): The input dataset containing audio data.

        Returns:
        tf.data.Dataset: A new dataset containing spectrogram representations of audio data.

        This function takes an input dataset of audio data and converts it to spectrograms. It uses the
        `_get_spectrogram` method to perform the conversion.

        Example:
        ```
        audio_dataset = ...
        spectrogram_dataset = self.convert_to_spectrogram(audio_dataset)
        ```

        """
        return data_set.map(
            map_func=lambda audio,label: (self._get_spectrogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE)
