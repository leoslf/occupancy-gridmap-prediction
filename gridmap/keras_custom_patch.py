import os
import random
import keras_preprocessing.image
# from keras_preprocessing.image.utils import load_img
import numpy as np

def set_processing_attrs(self,
                         image_data_generator,
                         target_size,
                         color_mode,
                         data_format,
                         save_to_dir,
                         save_prefix,
                         save_format,
                         subset,
                         interpolation):
    """Sets attributes to use later for processing files into a batch.
    # Arguments
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """
    self.image_data_generator = image_data_generator
    self.target_size = tuple(target_size)
    if color_mode not in {"rgb", "rgba", "grayscale"}:
        raise ValueError("Invalid color mode:", color_mode,
                         "; expected \"rgb\", \"rgba\", or \"grayscale\".")
    self.color_mode = color_mode
    self.data_format = data_format
    if self.color_mode == "rgba":
        if self.data_format == "channels_last":
            self.image_shape = self.target_size + (4,)
        else:
            self.image_shape = (4,) + self.target_size
    elif self.color_mode == "rgb":
        if self.data_format == "channels_last":
            self.image_shape = self.target_size + (3,)
        else:
            self.image_shape = (3,) + self.target_size
    else:
        if self.data_format == "channels_last":
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size
    self.save_to_dir = save_to_dir
    self.save_prefix = save_prefix
    self.save_format = save_format
    self.interpolation = interpolation
    if subset is not None:
        if subset not in {"training", "validation", "testing"}:
            raise ValueError(
                "Invalid subset name: %s;"
                "expected \"training\", \"validation\" or \"testing\"" % (subset,))

        validation_split = self.image_data_generator._validation_split
        testing_split = self.image_data_generator._testing_split
        if subset == "validation":
            split = (0, validation_split)
        elif subset == "testing":
            split = (validation_split, testing_split)
        else:
            split = (validation_split + testing_split, 1)
    else:
        split = None
    self.split = split
    self.subset = subset

# keras_preprocessing.image.iterator.BatchFromFilesMixin.set_processing_attrs = set_processing_attrs

def load_and_crop_img(path,
                      grayscale = False,
                      color_mode = "rgb",
                      target_size = None,
                      interpolation = "nearest"):
    """Wraps keras_preprocessing.image.utils.loag_img() and adds cropping.
    Cropping method enumarated in interpolation
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ":" where first part is interpolation and second is crop
            e.g. "lanczos:random".
            Supported interpolation methods are "nearest", "bilinear", "bicubic", "lanczos",
            "box", "hamming" By default, "nearest" is used.
            Supported crop methods are "none", "center", "random".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    # Decode interpolation string. Allowed Crop methods: none, center, random
    interpolation, crop = interpolation.split(":") if ":" in interpolation else (interpolation, "none")  

    if crop == "none":
        return keras_preprocessing.image.utils.load_img(path, 
                                                        grayscale = grayscale, 
                                                        color_mode = color_mode, 
                                                        target_size = target_size,
                                                        interpolation = interpolation)

    # Load original size image using Keras
    img = keras_preprocessing.image.utils.load_img(path, 
                                                   grayscale = grayscale, 
                                                   color_mode = color_mode, 
                                                   target_size = None, 
                                                   interpolation = interpolation)

    # Crop fraction of total image
    crop_fraction = 0.875
    target_width = target_size[1]
    target_height = target_size[0]

    if target_size is not None:        
        if img.size != (target_width, target_height):

            if crop not in ["center", "random", "random_center"]:
                raise ValueError("Invalid crop method {} specified.", crop)

            if interpolation not in keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    "Invalid interpolation method {} specified. Supported "
                    "methods are {}".format(interpolation,
                        ", ".join(keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS.keys())))
            
            resample = keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

            width, height = img.size

            # Resize keeping aspect ratio
            # result shold be no smaller than the targer size, include crop fraction overhead
            target_size_before_crop = (target_width/crop_fraction, target_height/crop_fraction)
            ratio = max(target_size_before_crop[0] / width, target_size_before_crop[1] / height)
            target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
            img = img.resize(target_size_before_crop_keep_ratio, resample = resample)

            width, height = img.size

            if crop == "center":
                left_corner = width // 2 - target_width // 2
                # int(round(width/2)) - int(round(target_width/2))
                top_corner = height // 2 - target_height // 2
                # int(round(height/2)) - int(round(target_height/2))
                return img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
            elif crop == "random":
                left_shift = random.randint(0, int((width - target_width)))
                down_shift = random.randint(0, int((height - target_height)))
                return img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))
            elif crop == "random_center":
                ratio = random.uniform(random_center_p, 1.0)
                # randomly cropped (width, height)
                _w, _h = map(int, [x * ratio for x in (width, height)])
                left_corner = width // 2 - _w // 2
                top_corner = height // 2 - _h // 2

                return img.crop((left_corner, top_corner, left_corner + _w, top_corner + _h)).resize((target_width, target_height), resample = resample)

    return img

def _get_batches_of_transformed_samples(self, index_array):
    """Gets a batch of transformed samples.
    # Arguments
        index_array: Array of sample indices to include in batch.
    # Returns
        A batch of transformed samples.
    """
    batch_x = np.zeros((len(index_array),) + self.image_shape, dtype = self.dtype)
    batch_y_ = np.zeros((len(index_array),) + self.image_shape, dtype = self.dtype)

    def _load_img(filepath):
        img = load_img(filepath,
                       color_mode = self.color_mode,
                       target_size = self.target_size,
                       interpolation = self.interpolation)

        x = img_to_array(img, data_format = self.data_format)
        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, "close"):
            img.close()
        if self.image_data_generator:
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)

        return x

    def replace_path(path, filepath):
        basename = os.path.basename(filepath)
        # print ("filepath: %s, path: %s, basename: %s" % (filepath, path, basename))
        return os.path.join(path, basename)

    # build batch of image data
    # self.filepaths is dynamic, is better to call it once outside the loop
    filepaths = self.filepaths
    for i, j in enumerate(index_array):
        batch_x[i] = _load_img(filepaths[j])
        if self.class_mode == "image":
            batch_y_[i] = _load_img(replace_path(self.directory_y, filepaths[j]))

    # optionally save augmented images to disk for debugging purposes
    if self.save_to_dir:
        for i, j in enumerate(index_array):
            img = array_to_img(batch_x[i], self.data_format, scale = True)
            fname = "{prefix}_{index}_{hash}.{format}".format(
                prefix = self.save_prefix,
                index = j,
                hash = np.random.randint(1e7),
                format = self.save_format)
            img.save(os.path.join(self.save_to_dir, fname))
    # build batch of labels
    if self.class_mode == "input":
        batch_y = batch_x.copy()
    elif self.class_mode == "image":
        batch_y = batch_y_
    elif self.class_mode in {"binary", "sparse"}:
        batch_y = np.empty(len(batch_x), dtype = self.dtype)
        for i, n_observation in enumerate(index_array):
            batch_y[i] = self.classes[n_observation]
    elif self.class_mode == "categorical":
        batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                           dtype = self.dtype)
        for i, n_observation in enumerate(index_array):
            batch_y[i, self.classes[n_observation]] = 1.
    elif self.class_mode == "multi_output":
        batch_y = [output[index_array] for output in self.labels]
    elif self.class_mode == "raw":
        batch_y = self.labels[index_array]
    else:
        return batch_x
    if self.sample_weight is None:
        return batch_x, batch_y
    else:
        return batch_x, batch_y, self.sample_weight[index_array]

def DataFrameIterator_constructor(self, dataframe, *argv, directory_y = None, **kwargs):
    DataFrameIterator_constructor_backup(self, dataframe, *argv, **kwargs)
    self.directory_y = directory_y
    # print ("DataFrameIterator_constructor: self.directory_y: %r" % self.directory_y)

def DataFrameIterator_constructor2(self, dataframe, *argv, directory_y = None, **kwargs):
    is_image = "class_mode" in kwargs and kwargs["class_mode"] == "image"
    # Manually set it after constructor
    if is_image:
        kwargs["class_mode"] = None

    DataFrameIterator_constructor_backup2(self, dataframe, *argv, **kwargs)

    if is_image:
        self.class_mode = "image"
    self.directory_y = directory_y

    # print ("DataFrameIterator_constructor2: self.directory_y: %r" % self.directory_y)

# Monkey patch
# load_img = keras_preprocessing.image.utils.load_img
keras_preprocessing.image.iterator.load_img = load_and_crop_img
random_center_p = 0.75
load_img = keras_preprocessing.image.iterator.load_img # keras_preprocessing.image.utils.load_img
img_to_array = keras_preprocessing.image.iterator.img_to_array
array_to_img = keras_preprocessing.image.iterator.array_to_img
keras_preprocessing.image.iterator.BatchFromFilesMixin._get_batches_of_transformed_samples = _get_batches_of_transformed_samples
keras_preprocessing.image.dataframe_iterator.DataFrameIterator.allowed_class_modes.add("image")
DataFrameIterator_constructor_backup = keras_preprocessing.image.dataframe_iterator.DataFrameIterator.__init__
keras_preprocessing.image.dataframe_iterator.DataFrameIterator.__init__ = DataFrameIterator_constructor

import keras
DataFrameIterator_constructor_backup2 = keras.preprocessing.image.DataFrameIterator.__init__
keras.preprocessing.image.DataFrameIterator.__init__ = DataFrameIterator_constructor2
