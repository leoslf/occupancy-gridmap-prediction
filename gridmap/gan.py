from gridmap.model_base import *

class GAN(BaseModel):
    @property
    def input_shape(self):
        return (64, 64, 1)

    @property
    def leaky_relu_alpha(self):
        return 0.2

    @property
    def optimizer(self):
        return Adam(0.0002, beta_1 = 0.5)

    @property
    def bn_momentum(self):
        return 0.8

    @property
    def latent_shape(self):
        return (100,)

    @property
    def class_mode(self):
        return "image"

    def init(self):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.writer = tf.summary.FileWriter(self.logdir)

    def conv_block(self, units, inputs, block_number, kernel_size, strides, bn = False, dropout = False):
        with K.name_scope("ConvBlock_%d" % block_number):
            conv = Conv2D(units,
                          kernel_size = kernel_size,
                          strides = strides,
                          padding = "same",
                          kernel_initializer = self.kernel_init)(inputs)
            conv = LeakyReLU(alpha = self.leaky_relu_alpha)(conv)
            if dropout:
                conv = Dropout(self.dropout_rate)(conv)
            if bn:
                conv = BatchNormalization(momentum = self.bn_momentum)(conv)
            return conv

    def deconv_block(self, filters, inputs, block_number, kernel_size, strides, short_circuit_layer, bn = False, dropout = False):
        with K.name_scope("DeConvBlock_%d" % block_number):
            conv = inputs
            # conv = UpSampling2D(2)(inputs)
            conv = Conv2DTranspose(filters,
                                   kernel_size = kernel_size,
                                   strides = strides,
                                   padding = "same",
                                   kernel_initializer = self.kernel_init)(conv)
            if dropout:
                conv = Dropout(self.dropout_rate)(conv)

            if bn:
                conv = BatchNormalization(momentum = self.bn_momentum)(conv)

            output = Concatenate()([conv, short_circuit_layer])
            return output


    def enumerated_filter_series(self, *argv, **kwargs):
        return list(enumerate(self.filter_series(*argv, **kwargs), 1))

    @property
    def conv_blocks(self):
        return 4

    @property
    def generator_filter_init(self):
        return 64

    @property
    def discriminator_filter_init(self):
        return 64

    @property
    def input_name(self):
        return "noisy_img_input"


    def prepare_generator(self, noisy_img_input):

        conv = noisy_img_input
        downsampling_layers = [conv]
        filters = self.filter_series(self.generator_filter_init, 2, self.conv_blocks)
        with K.name_scope("Downsampling"):
            for (i, units) in enumerate(filters, 1): # , last_repeat = 4):
                conv = self.conv_block(units, conv, i, kernel_size = (3, 3), strides = (2, 2), bn = (i > 1))
                # Save layers
                downsampling_layers.append(conv)

        self.logger.info("downsampling_layers: %r", "\n".join(map(str, enumerate(map(K.int_shape, downsampling_layers)))))

        # upsampling_filters = [1] + filters
        upsampling_filters = filters
        with K.name_scope("Upsampling"):
            for (i, units) in zip(reversed(range(len(upsampling_filters))), reversed(upsampling_filters)):
                self.logger.info("upsampling: i: %d", i)
                conv = self.deconv_block(units, conv, i + 1, kernel_size = (3, 3), strides = (2, 2), short_circuit_layer = downsampling_layers[i], bn = True)
            
            # conv = UpSampling2D(2)(conv)
            output = Conv2DTranspose(self.input_shape[-1],
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     padding = "same",
                                     activation = "tanh",
                                     kernel_initializer = self.kernel_init)(conv)

            return output

    @property
    def disc_patch(self):
        """ Shape of PatchGAN """
        return tuple(np.array(self.input_shape[:2]) // int(2 ** self.conv_blocks)) + (1, )

    def prepare_discriminator(self, sensor_image_input, output_image_input):

        conv = Concatenate(axis = -1)([sensor_image_input, output_image_input])

        for (i, units) in self.enumerated_filter_series(self.discriminator_filter_init, 2, self.conv_blocks):
            conv = self.conv_block(units, conv, i, kernel_size = (3, 3), strides = (2, 2), bn = (i > 1))

        output = Conv2D(self.input_shape[-1],
                        kernel_size = (3, 3),
                        strides = (1, 1),
                        padding = "same",
                        kernel_initializer = self.kernel_init)(conv)

        return output


    @property
    def loading_weights(self):
        return False 

    @property
    def discriminator_loss(self):
        return "mse"

    @property
    def loss(self):
        return ["mse", "mae"]

    @property
    def loss_weights(self):
        return [1, 100]

    def construct_model(self):
        self.sensor_input_layer = Input(self.input_shape, name = "noisy_img_input")
        self.output_image_input_layer = Input(self.input_shape, name = "gt_img_input")

        self.discriminator_output = self.prepare_discriminator(self.sensor_input_layer, self.output_image_input_layer)
        self.discriminator = Model([self.sensor_input_layer, self.output_image_input_layer], self.discriminator_output, name = "%s_Discriminator" % self.name)
        self.discriminator.summary()

        self.generator = Model(self.sensor_input_layer, self.prepare_generator(self.sensor_input_layer), name = "%s_Generator" % self.name)
        self.generator.summary()

    def compile(self):
        try:
            self.load_weights(model = self.discriminator)
        except:
            raise ImportError("Could not load pretrained model weights")
        self.discriminator.compile(loss = self.discriminator_loss, optimizer = self.optimizer, metrics = ["accuracy"])


        try:
            self.load_weights(model = self.generator)
        except:
            raise ImportError("Could not load pretrained model weights")
        # self.generator.compile(loss = self.loss, optimizer = self.optimizer)

        self.generated_image = self.generator(self.sensor_input_layer)
        self.validity = self.discriminator([self.sensor_input_layer, self.generated_image])

        # Disable discrminiator training in combined model
        self.discriminator.trainable = False

        self.model = Model([self.sensor_input_layer, self.output_image_input_layer], [self.validity, self.generated_image], name = self.name)
        self.model.compile(loss = self.loss, loss_weights = self.loss_weights, optimizer = self.optimizer)

    def write_log(self, logs, epoch):
        for (name, value) in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.tag = name
            summary_value.simple_value = value
            self.writer.add_summary(summary, epoch)

    @property
    def gan_checkpoint_filename(self):
        return os.path.join(self.logdir, "epoch{epoch:03d}-d_loss{d_loss:.3f}-d_accuracy{d_accuracy:.3f}-g_loss{g_loss:.3f}.h5" % dict(metric = self.main_metric))

    @property
    def x_directory(self):
        return os.path.join(self.config.root_dir, "submaps")

    @property
    def train_interpolation(self):
        # return "bilinear" # "lanczos:random_center"
        return "lanczos:random_center"

    @property
    def use_earlystopping(self):
        return True

    def fit_df(self):

        d_losses = []
        d_accuracies = []
        g_losses = []
        g_accuracies = []
    
        batches = steps_from_gen(self.train_generator)

        self.save_images("input", self.validation_X, 0, revert = False)
        self.save_images("ground_truth", self.validation_gt, 0, revert = False)

        for epoch in range(0, self.epochs):
            d_losses_epoch = []
            d_accuracies_epoch = []
            g_losses_epoch = []
            g_accuracies_epoch = []
            for batch_num, (X_batch, gt_batch) in zip(np.arange(batches) + 1, self.train_generator):
                batch_size = len(X_batch)

                # Adversarial ground truths
                trues = np.ones((batch_size, *self.disc_patch))
                falses = np.zeros((batch_size, *self.disc_patch))

                X_batch 

                # Generate images
                generated_images = self.generator.predict(X_batch)

                # Training the discriminator
                loss_real, accuracy_real = self.discriminator.train_on_batch([X_batch, gt_batch], trues)
                loss_fake, accuracy_fake = self.discriminator.train_on_batch([X_batch, generated_images], falses)

                d_loss = 0.5 * (loss_real + loss_fake)
                d_losses_epoch.append(d_loss)
                d_accuracy = 0.5 * (accuracy_real + accuracy_fake)
                d_accuracies_epoch.append(d_accuracy)

                # Training the generator
                # noise = np.random.normal(0, 1, (batch_size, *self.latent_shape))
                g_loss = self.model.train_on_batch([gt_batch, X_batch], [trues, gt_batch])
                g_accuracy = np.average(g_loss[1:], weights = self.loss_weights)
                g_accuracies_epoch.append(g_accuracy)
                g_loss = g_loss[0]
                g_losses_epoch.append(g_loss)

                self.logger.info("[%3d/%3d] [%3d/%3d] [D: loss: %f, acc: %.2f%%][G: loss: %f]", epoch, self.epochs, batch_num, batches, d_loss, d_accuracy * 100, g_loss)


            d_loss = np.mean(d_losses_epoch)
            d_accuracy = np.mean(d_accuracies_epoch)
            g_loss = np.mean(g_losses_epoch)
            g_accuracy = np.mean(g_accuracies_epoch)

            logs = dict(d_loss = d_loss, d_accuracy = d_accuracy, g_loss = g_loss)
            self.write_log(logs, epoch)

            if epoch % self.config.vis_step == 0:
                self.save_images("prediction", self.generator.predict(self.validation_X), epoch)

            
            if len(g_losses) == 0 or g_loss < np.min(g_losses) or epoch % self.config.checkpoint_step == 0:
                self.model.save_weights(self.gan_checkpoint_filename.format(epoch = epoch, **logs))

            d_losses.append(d_loss)
            d_accuracies.append(d_accuracy)
            g_losses.append(g_loss)
            g_accuracies.append(g_accuracy)

        
        return dict(d_loss = d_losses, d_accuracy = d_accuracies, g_loss = g_losses) # , g_accuracy = g_accuracies)

    @property
    def postprocessing(self):
        return revert_standardize

    def save_images(self, key, generated_images, epoch, revert = True):
        if revert:
            generated_images = revert_standardize(generated_images)
        grid = make_grid(generated_images)
        imageio.imwrite(os.path.join(self.logdir, "%s_epoch_%d_grid.png" % (key, epoch)), grid)

        image = make_image(grid)
        self.writer.add_summary(tf.Summary(value = [tf.Summary.Value(tag = key, image = image)]), epoch)
        self.writer.flush()

