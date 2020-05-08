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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio

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

def grid_length(n):
    """ Find the length for the square grid to contain all n items """
    return np.ceil(np.sqrt(n)).astype(int)

def make_grid(images):
    # logger.info("images.shape: %r", images.shape)
    batch_size, height, width, intensity = images.shape
    n = grid_length(batch_size)
    # logger.info("n: %r", n)
    nrows = ncols = n
    # assert batch_size == nrows * ncols
    result = np.concatenate((images, np.zeros((np.square(n) - batch_size, height, width, intensity))), axis = 0)
    # result = result.reshape((ncols, nrows, height, width, intensity))
    # result = np.transpose(result, (1, 0, 2, 3, 4))
    result = result.reshape((ncols, nrows, height, width, intensity))
    result = np.transpose(result, (0, 2, 1, 3, 4))
    result = result.reshape((height * nrows, width * ncols, intensity))
    # logger.info("result.shape: %r", result.shape)
    # logger.info("result.min: %r, result.max: %r", np.min(result), np.max(result))
    return np.uint8(result * 255)

def img_diff(A, B):
    diff = (A.astype(float) - B.astype(float)) #  / 255
    pos = diff.copy()
    neg = diff.copy()
    pos[pos < 0] = 0
    neg[neg > 0] = 0
    neg *= -1

    results = np.zeros((*A.shape[:-1], 3))
    results[:, :, 0] = pos[:, :, 0]
    results[:, :, 1] = neg[:, :, 0]

    return np.uint8(results * 255)

    

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    logger.info("tensor: %r", tensor)
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class PredictionVisualizer(Callback):
    def __init__(self, prefix, generator, step, log_dir = "./logs", **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        self.prefix = prefix
        self.log_dir = os.path.join(log_dir, self.prefix)
        self.writer = tf.summary.FileWriter(self.log_dir)
        super().__init__(**kwargs)

        # logger.info("vars(generator): %r", vars(generator))

        # self.X, self.ground_truth = generator
        self.generator = generator
        self.step = step

    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        
        if epoch % self.step == 0:
            # prediction = self.model.predict_generator(self.X)
            prediction = self.model.predict_generator(self.generator)

            self.write_images(prediction, epoch)
            # {
            #     "input_groundtruth": self.generator,
            #     # "ground_truth": self.ground_truth,
            #     "prediction": prediction
            # }, epoch)

    def make_grid(self, img_batch, ncols = 8):
        batch_size, height, width, intensity = K.int_shape(img_batch)
        nrows = batch_size // ncols
        assert batch_size == nrows * ncols
        result = K.reshape(img_batch, (nrows, ncols, height, width, intensity))
        result = K.permute_dimensions(result, (1, 0, 2, 3, 4))
        result = K.reshape(result, (1, height * nrows, width * ncols, intensity))

        return result


    def write_image(self, key, img, epoch):
        # image = make_image(img)
        with tf.Session() as sess:
            summary = tf.summary.image(key, self.make_grid(img))
            self.writer.add_summary(summary.eval(), epoch)

    def write_images(self, prediction, epoch):
        for (X_batch, ground_truth_batch) in self.generator:
            self.write_image("input", X_batch, epoch)
            self.write_image("ground_truth", ground_truth_batch, epoch)

        for batch in prediction:
            self.write_image("prediction", batch, epoch)

