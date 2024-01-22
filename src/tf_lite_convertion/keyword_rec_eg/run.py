from os import path

import numpy as np
import pandas as pd
import tensorflow as tf

from .. import *
from .model import KeywordRecognitionModel


TFLITE_FILE = f"KeyWordRecognition_v1.tflite"


def main():
    model = KeywordRecognitionModel()
    save_model(model, SAVED_MODEL_DIR)
    tflite_model = convert_saved_model(SAVED_MODEL_DIR)
    save_tflite_model(tflite_model, TFLITE_FILE)


main() if __name__ == "__main__" else None