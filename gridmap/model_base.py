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
                 verbose = 2,
                 use_multiprocessing = False,
                 compiled = False,
                 *argv, **kwargs):

        self.config = self.prepare_config()
        self.logdir = self.prepare_logdir()
        self.logger = logging.getLogger(self.name)

        self.compiled = compiled
        self.epochs = self.config.epochs
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing
        self.__dict__.update(kwargs)

        df = pd.read_csv(self.config.txt_file, header = None, names = ["image_id"])

        trainval_df, self.testing_df = train_test_split(df, test_size = self.testing_split, random_state = 6487001)
        self.train_df, self.validation_df = train_test_split(trainval_df, test_size = self.validation_split * (1 - self.testing_split), random_state = 6487002)

        self.train_img_generator = ImageDataGenerator(
                                    preprocessing_function = self.preprocessing_function,
                                    vertical_flip = self.vertical_flip,
                                    **self.data_generator_kwargs)

        self.test_img_generator = ImageDataGenerator(
                                    preprocessing_function = self.preprocessing_function,
                                    **self.data_generator_kwargs)


        self.train_generator = self.get_generator(self.train_img_generator, self.train_df, batch_size = self.config.batch_size, interpolation = self.train_interpolation, **self.generator_kwargs)
        self.validation_generator = self.get_generator(self.test_img_generator, self.validation_df, batch_size = self.config.batch_size, **self.generator_kwargs)
        self.test_generator = self.get_generator(self.test_img_generator, self.testing_df, batch_size = 1, **self.generator_kwargs)

        self.construct_model()

        self.train_generator_single_batch = self.get_generator(self.train_img_generator, self.train_df, batch_size = len(self.train_df), interpolation = "lanczos:random_center", **self.generator_kwargs)


        self.init()

        # self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        try:
            if self.loading_weights:
                self.load_weights()
        except:
            raise ImportError("Could not load pretrained model weights")

        if not self.compiled:
            self.compile() # self.model)
            self.logger.debug("compiled: %s" % self.name)

        self.logger.info(capture_stdout(self.model.summary))
        # self.model.summary()

    @property
    def train_interpolation(self):
        return "lanczos:random_center"

    @property
    def loading_weights(self):
        return True

    @property
    def vertical_flip(self):
        return True

    def compile(self):
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)

    def construct_model(self):
        self.input_layer = Input(self.input_shape)
        self.output_layer = self.prepare_model(self.input_layer)
        self.model = Model(self.input_layer, self.output_layer, name = self.name)

    @property
    def config_filename(self):
        return "config.yml"

    @property
    def class_mode(self):
        raise NotImplementedError

    @property
    def generator_kwargs(self):
        return {
            "class_mode": self.class_mode
        }

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
        return {
            "rescale": 1. / 255.0
        }

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

    def flow_from_dataframe(self, data_generator, dataframe, directory, subset = None, class_mode = None, **kwargs):
        return data_generator.flow_from_dataframe(dataframe = dataframe,
                                                  subset = subset,
                                                  directory = directory,
                                                  x_col = "image_id",
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

    def get_generator(self, data_generator, df, **kwargs):
        return self.flow_from_dataframe(data_generator, df, directory = self.x_directory, directory_y = self.y_directory, **kwargs)

    @property
    def optimizer(self):
        return Adam(lr = self.config.learning_rate) # Adadelta()

    @property
    def loss(self):
        return "binary_crossentropy"

    @property
    def weight_filename(self):
        return "%s.h5" % self.name

    def load_weights(self, filename = None, model = None):
        if model is None:
            model = self.model

        if filename is None:
            filename = self.weight_filename

        if os.path.exists(filename):
            model.load_weights(filename, by_name=True, skip_mismatch=True)

    def save_weights(self):
        self.model.save_weights(self.weight_filename)

    @property
    def metrics(self):
        return ["accuracy"]

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
                            patience = self.earlystopping_patience, 
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
    def earlystopping_patience(self):
        return 10

    @property
    def tensorboard_kwargs(self):
        return dict(
            write_graph = True,
            update_freq = "epoch",
        )

    @property
    def tensorboard_class(self):
        return TrainValTensorBoard # instead of the vanilla Tensorboard

    @property
    def callbacks(self):
        callbacks = [
            self.modelcheckpoint,
            self.tensorboard_class(log_dir = self.logdir, **self.tensorboard_kwargs),
            TerminateOnNaN(),
            # PredictionVisualizer(
            #     log_dir = self.logdir,
            #     prefix = "training",
            #     generator = self.train_generator,
            #     step = self.config.vis_step
            # ),
            # PredictionVisualizer(
            #     log_dir = self.logdir,
            #     prefix = "validation",
            #     generator = self.validation_generator,
            #     step = self.config.vis_step
            # )
        ]
        if self.use_earlystopping:
            callbacks.append(self.earlystopping)

        return callbacks

    def prepare_logdir(self):
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
        history = self.model.fit_generator(generator = self.train_generator, # zip(*self.train_generator),
                                           # steps_per_epoch = steps_from_gen(self.train_generator[0]),
                                           steps_per_epoch = steps_from_gen(self.train_generator),
                                           # validation_data = zip(*self.validation_generator),
                                           validation_data = self.validation_generator,
                                           # validation_steps = steps_from_gen(self.validation_generator[0]),
                                           validation_steps = steps_from_gen(self.validation_generator),
                                           epochs = self.epochs,
                                           callbacks = self.callbacks,
                                           verbose = self.verbose)
        self.save_weights()
        return history
    
    def evaluate(self, test_X, test_Y):
        return self.model.evaluate(test_X, test_Y,
                                   batch_size = self.config.batch_size,
                                   verbose = self.verbose,
                                   use_multiprocessing = self.use_multiprocessing)

    def evaluate_df(self, **kwargs):
        # return self.model.evaluate_generator(generator = zip(*self.test_generator),
        return self.model.evaluate_generator(generator = self.test_generator,
                                             # steps = steps_from_gen(self.test_generator[0]),
                                             steps = steps_from_gen(self.test_generator),
                                             # callbacks = self.callbacks,
                                             verbose = self.verbose)

    def predict(self, X, *argv, **kwargs):
        return self.model.predict(X, *argv, **kwargs)

    def filter_series(self, num_filters_init, growth_factor, repeats):
        return [num_filters_init * int(growth_factor ** i) for i in range(repeats)]

    def save_images(self):
        train_X, train_gt = self.train_generator_single_batch.next()
        train_X_grid = make_grid(train_X)
        train_gt_grid = make_grid(train_gt)
        train_pred_grid = make_grid(self.model.predict(train_X))

        imageio.imwrite("train_X.png", train_X_grid)
        imageio.imwrite("train_gt.png", train_gt_grid)
        imageio.imwrite("train_pred.png", train_pred_grid)

        X = train_X_grid / 255
        gt = train_gt_grid / 255
        pred = train_pred_grid / 255

        gt_pred = img_diff(gt, pred)
        x_pred = img_diff(X, pred)
        x_gt = img_diff(X, gt)
        
        imageio.imwrite("gt_pred.png", gt_pred)
        imageio.imwrite("x_pred.png", x_pred)
        imageio.imwrite("x_gt.png", x_gt)



