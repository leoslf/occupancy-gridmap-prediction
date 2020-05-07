import os
import numpy as np
import pandas as pd
import logging

import functools

from itertools import *
from datetime import datetime

# from gridmap.keras_custom_patch import *
import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    # import keras_preprocessing.image
    # keras_preprocessing.image.iterator.BatchFromFilesMixin.set_processing_attrs = set_processing_attrs

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

def random_center_crop(batch, p = 0.75):
    img_x, img_y = batch
    ratio = random.uniform(p, 1.0)
    size_x = int(img_x.shape[0, 0] * ratio)
    size_y = int(img_y.shape[0, 0] * ratio)

    # return img_x[:, x


def augment_data(pair):
    pass


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, "training")
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, "validation")

    def set_model(self, model):
        # Setup writer for validation metrics
        # self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace("val_", ""): v for k, v in logs.items() if k.startswith("val_")}
        for name, value in val_logs.items():
            tf.summary.scalar(name, value, epoch)
            # summary = tf.Summary()
            # summary_value = summary.value.add()
            # summary_value.simple_value = value.item()
            # summary_value.tag = name
            # self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

mse = MeanSquaredError()
coord_weight = 0.5
time_weight = 0.5
def custom_loss(y_true, y_predict):
    coord_true, coord_predict = y_true[:, :2], y_predict[:, :2]
    time_true, time_predict = y_true[:, 2], y_predict[:, 2]

    def custom(y_true, y_predict):
        abs_diff = K.abs(y_true - y_predict)
        diff_int = tf.floor(abs_diff)
        diff = abs_diff - diff_int - 0.5
        lessthan_0_5 = K.cast(K.less_equal(diff, -0.5), tf.float32)

        error = lessthan_0_5 * (diff + 1) + (1 - lessthan_0_5) * diff

        return K.mean(K.abs(error))

    # MSE for coordinates, custom for time
    return coord_weight * K.mean(K.square(coord_true - coord_predict)) + time_weight * custom(time_true, time_predict)
