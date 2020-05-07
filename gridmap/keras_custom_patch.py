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
    if color_mode not in {'rgb', 'rgba', 'grayscale'}:
        raise ValueError('Invalid color mode:', color_mode,
                         '; expected "rgb", "rgba", or "grayscale".')
    self.color_mode = color_mode
    self.data_format = data_format
    if self.color_mode == 'rgba':
        if self.data_format == 'channels_last':
            self.image_shape = self.target_size + (4,)
        else:
            self.image_shape = (4,) + self.target_size
    elif self.color_mode == 'rgb':
        if self.data_format == 'channels_last':
            self.image_shape = self.target_size + (3,)
        else:
            self.image_shape = (3,) + self.target_size
    else:
        if self.data_format == 'channels_last':
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
                'Invalid subset name: %s;'
                'expected "training", "validation" or "testing"' % (subset,))

        validation_split = self.image_data_generator._validation_split
        testing_split = self.image_data_generator._testing_split
        if subset == 'validation':
            split = (0, validation_split)
        elif subset == "testing":
            split = (validation_split, testing_split)
        else:
            split = (validation_split + testing_split, 1)
    else:
        split = None
    self.split = split
    self.subset = subset

