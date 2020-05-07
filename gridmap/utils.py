import io
import sys
import os
import numpy as np
import pandas as pd
import logging

import functools

from itertools import *
from datetime import datetime

from contextlib import redirect_stdout

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category = FutureWarning)
    from gridmap.keras_custom_patch import *

    import tensorflow as tf

    from keras.models import *
    from keras.layers import *
    from keras.initializers import *
    from keras.optimizers import *
    from keras.regularizers import *
    from keras.objectives import *
    from keras.callbacks import * 
    from keras.losses import * 
    from keras.preprocessing.image import *

    from keras import backend as K
    from keras.utils import generic_utils

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

HOURS_PER_DAY = 24.
MINUTES_PER_HOUR = 60.
SECONDS_PER_MINUTE = 60.
MILLISECONDS_PER_SECOND = 1000.
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY
SECONDS_PER_DAY = SECONDS_PER_MINUTE * MINUTES_PER_DAY # 60 * 1,440 = 86,400 SI seconds
MILLISECONDS_PER_DAY = MILLISECONDS_PER_SECOND * SECONDS_PER_DAY # 1000 * 86400 = 86,400,000

def steps_from_gen(generator):
    steps = generator.n // generator.batch_size
    assert steps > 0
    return steps

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def capture_stdout(f):
    with io.StringIO() as buf, redirect_stdout(buf):
        f()
        return buf.getvalue()

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, "training")
        super().__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, "validation")

    def set_model(self, model):
        # Setup writer for validation metrics
        # self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super().set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        # print (logs)
        val_logs = {k.replace("val_", ""): v for k, v in logs.items() if k.startswith("val_")}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.tag = name
            summary_value.simple_value = value
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
        super().on_epoch_end(epoch, logs)



    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.val_writer.close()

class PredictionVisualizer(Callback):
    def __init__(self, prefix, generator, step, log_dir = "./logs", **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        self.prefix = prefix
        self.log_dir = os.path.join(log_dir, self.prefix)
        self.writer = tf.summary.FileWriter(self.log_dir)
        super().__init__(**kwargs)

        self.X, self.ground_truth = generator
        self.step = step

    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        
        if epoch % self.step == 0:
            prediction = self.model.predict_generator(self.X)
            self.write_images({
                "images": self.X,
                "ground_truth": self.ground_truth,
                "prediction": prediction
            })

    def write_images(self, dict):
        for (key, value) in dict.items():
            self.writer.add_image(key, value)

