import os

from datetime import datetime
from gridmap.utils import *
import utils

class CustomEarlyStopping(EarlyStopping):

    def __init__(self, target=None, **kwargs):
        self.target = target
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        current = self.get_monitor_value(logs)
        if not self.target or self.monitor_op(self.target, self.best):
            super().on_epoch_end(epoch, logs)

class BaseModel:
    def __init__(self,
                 # input_shape = (3, 3),
                 # output_shape = (3, ),
                 epochs = 1000,
                 verbose = 2,
                 # validation_split = 0.3,
                 # testing_split = 0.3,
                 use_multiprocessing = False,
                 compiled = False,
                 *argv, **kwargs):
        # self.input_shape = input_shape
        # self.output_shape = output_shape
        self.compiled = compiled
        self.epochs = epochs
        self.verbose = verbose
        # self.validation_split = validation_split
        # self.testing_split = testing_split
        self.use_multiprocessing = use_multiprocessing
        self.__dict__.update(kwargs)

        self.config = self.prepare_config()

        df = pd.read_csv(self.config.txt_file, header = None, names = ["image_id"])

        trainval_df, self.testing_df = train_test_split(df, test_size = self.testing_split, random_state = 6487001)
        self.train_df, self.validation_df = train_test_split(trainval_df, test_size = self.validation_split * (1 - self.testing_split), random_state = 6487002)

        self.data_generator = ImageDataGenerator(
                                # validation_split=self.validation_split,
                                # testing_split=self.testing_split,
                                preprocessing_function = self.preprocessing_function,
                                **self.data_generator_kwargs)

        self.init()
        self.input_layer = Input(self.input_shape)
        self.output_layer = self.prepare_model(self.input_layer)
        self.model = Model(self.input_layer, self.output_layer, name = self.name)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        try:
            self.load_weights()
        except:
            raise ImportError("Could not load pretrained model weights")

        if not self.compiled:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            print ("compiled: %s" % self.__class__.__name__)

        self.model.summary()

    @property
    def config_filename(self):
        return "config.yml"

    def prepare_config(self):
        return utils.Config(self.config_filename)
    
    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def data_generator_kwargs(self):
        return {}

    @property
    def validation_split(self):
        return 0.1

    @property
    def testing_split(self):
        return 0.1

    @property
    def preprocessing_function(self):
        return None

    @property
    def name(self):
        return self.__class__.__name__

    def init(self):
        pass

    def flow_from_dataframe(self, dataframe, directory, subset = None, class_mode = None, **kwargs):
        return self.data_generator.flow_from_dataframe(dataframe = dataframe,
                                                       subset = subset,
                                                       directory = directory,
                                                       x_col = "image_id",
                                                       # y_col = ["healthy", "multiple_diseases", "rust", "scab"],
                                                       # has_ext = False,
                                                       class_mode = class_mode,
                                                       target_size = self.input_shape[:2],
                                                       color_mode = "grayscale",
                                                       validate_filenames = True,
                                                       **kwargs)
    @property
    def x_directory(self):
        return os.path.join(self.config.root_dir, "submaps")

    @property
    def y_directory(self):
        return os.path.join(self.config.root_dir, "submaps_gt")

    def get_generator(self, df):
        x_generator = self.flow_from_dataframe(df, self.x_directory)
        y_generator = self.flow_from_dataframe(df, self.y_directory)
        return x_generator, y_generator

    @property
    def optimizer(self):
        return Adadelta()

    @property
    def loss(self):
        return "binary_crossentropy"

    @property
    def weight_filename(self):
        return "%s.h5" % self.name

    def load_weights(self, filename = None):
        if filename is None:
            filename = self.weight_filename

        if os.path.exists(filename):
            self.model.load_weights(filename, by_name=True, skip_mismatch=True)

    def save_weights(self):
        self.model.save_weights(self.weight_filename)

    @property
    def metrics(self):
        return [] # "mean_squared_error"]

    @property
    def use_earlystopping(self):
        return False

    @property
    def main_metric(self):
        return "loss"

    @property
    def main_metric_mode(self):
        """
        In min mode, training will stop when the quantity monitored has stopped decreasing;
        in max mode it will stop when the quantity monitored has stopped increasing;
        in auto mode, the direction is automatically inferred from the name of the monitored quantity.
        """
        return "min"

    @property
    def monitor_metric(self):
        return "val_%s" % self.main_metric

    @property
    def earlystopping(self):
       # return CustomEarlyStopping(monitor="val_loss", # use validation accuracy for stopping
       return EarlyStopping(monitor = self.monitor_metric, 
                            min_delta = 0.0001,
                            patience = 20, 
                            verbose = self.verbose,
                            # target = 
                            mode = self.main_metric_mode)

    @property
    def checkpoint_filename(self):
        return os.path.join(self.logdir, "epoch{epoch:03d}-%(metric)s{%(metric)s:.3f}-val_%(metric)s{val_%(metric)s:.3f}.h5" % dict(metric = self.main_metric))

    @property
    def modelcheckpoint(self):
        return ModelCheckpoint(self.checkpoint_filename,
                               save_weights_only = True,
                               save_best_only = True,
                               mode = self.main_metric_mode)

    @property
    def write_images(self):
        return False

    @property
    def tensorboard_kwargs(self):
        return dict(
            write_graph = True,
            write_images = self.write_images,
            update_freq = "epoch",
            histogram_freq = 1,
        )

    @property
    def tensorboard_class(self):
        return TrainValTensorBoard # instead of the vanilla Tensorboard

    @property
    def callbacks(self):
        callbacks = [
            self.modelcheckpoint,
            self.tensorboard_class(log_dir=self.logdir, **self.tensorboard_kwargs),
            TerminateOnNaN(),
        ]
        if self.use_earlystopping:
            callbacks.append(self.earlystopping)

        return callbacks

    @property
    def logdir(self):
        return "logs/%s/%s" % (self.__class__.__name__, datetime.now().strftime("%Y%m%d-%H%M%S"))

    def prepare_model(self):
        return None

    def fit(self, train_X, train_Y, validation_X, validation_Y):
        history = self.model.fit(train_X, train_Y,
                                 validation_data = (validation_X, validation_Y),
                                 batch_size = self.config.batch_size,
                                 epochs = self.epochs,
                                 callbacks = self.callbacks,
                                 verbose = self.verbose,
                                 use_multiprocessing = self.use_multiprocessing)
        self.save_weights()
        return history

    def transform_gen(self, gen):
        pass

    def fit_df(self, **kwargs): 
        train_generator = self.get_generator(self.train_df, **kwargs) # augmentation = True, 
        validation_generator = self.get_generator(self.validation_df, **kwargs)
        history = self.model.fit_generator(generator = zip(*train_generator),
                                           steps_per_epoch = steps_from_gen(train_generator[0]),
                                           validation_data = zip(*validation_generator),
                                           validation_steps = steps_from_gen(validation_generator[0]),
                                           epochs = self.epochs,
                                           callbacks = self.callbacks,
                                           verbose = self.verbose)
        self.save_weights()
        return history
    
    def evaluate(self, test_X, test_Y):
        return self.model.evaluate(test_X, test_Y,
                                   batch_size = self.batch_size,
                                   verbose = self.verbose,
                                   use_multiprocessing = self.use_multiprocessing)

    def evaluate_df(self, **kwargs):
        test_generator = self.flow_from_dataframe(self.testing_df, batcH_size = 1, **kwargs) # , "testing", **kwargs)
        return self.model.evaluate_generator(generator = zip(*test_generator),
                                             steps = steps_from_gen(test_generator[0]),
                                             # callbacks = self.callbacks,
                                             verbose = self.verbose)

    def predict(self, X, *argv, **kwargs):
        return self.model.predict(X, *argv, **kwargs)
