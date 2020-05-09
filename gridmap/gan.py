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
        return Adam(0.0002, 0.5)

    @property
    def bn_momentum(self):
        return 0.8

    @property
    def latent_shape(self):
        return (100,)

    @property
    def class_mode(self):
        return None

    def init(self):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.writer = tf.summary.FileWriter(self.logdir)

    def conv_block(self, units, inputs, block_number, activation, bn = False, reverse = False):
        with K.name_scope("ConvBlock_%d" % block_number):
            if not reverse:
                conv = Conv2D(units,
                              kernel_size = (5, 5),
                              strides = (3, 3),
                              padding = "same")(inputs)
            else:
                conv = Conv2DTranspose(units,
                                       kernel_size = (5, 5),
                                       strides = (2, 2),
                                       padding = "same")(inputs)
            conv = activation(conv)
            if bn:
                conv = BatchNormalization(momentum = self.bn_momentum)(conv)
        return conv

    def enumerated_filter_series(self, *argv, **kwargs):
        return list(enumerate(self.filter_series(*argv, **kwargs), 1))

    @property
    def conv_blocks(self):
        return 3

    def prepare_generator(self, noise_input):

        shape = (16, 16, 256)

        dense = Dense(np.prod(shape))(noise_input)
        conv = Reshape(shape)(dense)

        for (i, units) in self.enumerated_filter_series(256, 2, self.conv_blocks - 1):
            conv = self.conv_block(units, conv, i, ReLU(), bn = True, reverse = True)

        output = Conv2DTranspose(1,
                                 kernel_size = (5, 5),
                                 padding = "same",
                                 activation = "tanh")(conv)

        return output

    def prepare_discriminator(self, image_input):

        # conv = Flatten()(image_input)
        conv = image_input
        for (i, units) in reversed(self.enumerated_filter_series(256, 2, self.conv_blocks - 1)):
            conv = self.conv_block(units, conv, i, LeakyReLU(alpha = self.leaky_relu_alpha), bn = True) # False)

        flatten = Flatten()(conv)

        output = Dense(1, activation = "sigmoid")(flatten)

        return output


    def construct_model(self):
        self.image_input = Input(self.input_shape)
        self.noise_input = Input(self.latent_shape)

        self.validity_layer = self.prepare_discriminator(self.image_input)
        self.discriminator = Model(self.image_input, self.validity_layer, name = "%s_discriminator" % self.name)
        self.discriminator.summary()

        self.image_layer = self.prepare_generator(self.noise_input)
        self.generator = Model(self.noise_input, self.image_layer, name = "%s_generator" % self.name)
        self.generator.summary()

    @property
    def loading_weights(self):
        return False 


    def compile(self):
        try:
            self.load_weights(model = self.discriminator)
        except:
            raise ImportError("Could not load pretrained model weights")
        self.discriminator.compile(loss = self.loss, optimizer = self.optimizer, metrics = ["accuracy"])

        try:
            self.load_weights(model = self.generator)
        except:
            raise ImportError("Could not load pretrained model weights")
        self.generator.compile(loss = self.loss, optimizer = self.optimizer)

        # Disable discrminiator training in combined model
        self.discriminator.trainable = False
        self.model = Model(self.noise_input, self.discriminator(self.generator(self.noise_input)), name = self.name)
        self.model.compile(loss = self.loss, optimizer = self.optimizer)

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
        return os.path.join(self.config.root_dir, "submaps_gt")

    @property
    def train_interpolation(self):
        return "bilinear" # "lanczos:random_center"

    @property
    def use_earlystopping(self):
        return True

    def fit_df(self):

        d_losses = []
        d_accuracies = []
        g_losses = []
    
        batches = steps_from_gen(self.train_generator)

        if not os.path.exists(self.weight_filename):
            repeats = 2

            noise = np.random.normal(0, 1, (repeats * batches * self.config.batch_size, *self.latent_shape))
            generated_images = self.generator.predict(noise)

            train_X = []
            for _, X in zip(range(batches), self.train_generator):
                train_X.append(X)

            train_X = np.row_stack(train_X)
            train_X = np.repeat(train_X, repeats, axis = 0)
            # self.discriminator.trainable = True
            self.discriminator.fit(np.row_stack((train_X, generated_images)), np.row_stack((np.ones((len(train_X), 1)), np.zeros((len(generated_images), 1)))),
                                   callbacks = self.callbacks,
                                   epochs = self.epochs,
                                   verbose = self.verbose,
                                   batch_size = self.config.batch_size,
                                   validation_split = self.validation_split)


        # sub_epoch = 1
        # # Epoch 0: Train Discriminator
        # while True:
        #     d_losses_epoch = []
        #     d_accuracies_epoch = []
        #     g_losses_epoch = []
        # 
        #     for batch_num, X_batch in zip(np.arange(batches) + 1, self.train_generator):
        #         batch_size = len(X_batch)

        #         # Adversarial ground truths
        #         trues = np.ones((batch_size, 1))
        #         falses = np.zeros((batch_size, 1))

        #         noise = np.random.normal(0, 1, (batch_size, *self.latent_shape))
        #         # Generate images
        #         generated_images = self.generator.predict(noise)

        #         # Training the discriminator
        #         loss_real, accuracy_real = self.discriminator.train_on_batch(X_batch * 2 - 1, trues)
        #         loss_fake, accuracy_fake = self.discriminator.train_on_batch(generated_images, falses)

        #         d_loss = 0.5 * (loss_real + loss_fake)
        #         d_losses_epoch.append(d_loss)
        #         d_accuracy = 0.5 * (accuracy_real + accuracy_fake)
        #         d_accuracies_epoch.append(d_accuracy)

        #         self.logger.info("[0.%3d/%3d] [%3d/%3d] [D: loss: %f, acc: %.2f%%][G: loss: -]", sub_epoch, self.epochs, batch_num, batches, d_loss, d_accuracy * 100)

        #     d_loss = np.mean(d_losses_epoch)
        #     d_accuracy = np.mean(d_accuracies_epoch)

        #     self.logger.info("[0.%3d/%3d] [D: loss: %f, acc: %.2f%%][G: loss: -]", sub_epoch, self.epochs, d_loss, d_accuracy * 100)

        #     if d_accuracy > 0.9:
        #         break

        #     sub_epoch += 1

        self.save_images(0)


        for epoch in range(1, self.epochs):
            d_losses_epoch = []
            d_accuracies_epoch = []
            g_losses_epoch = []
            for batch_num, X_batch in zip(np.arange(batches) + 1, self.train_generator):
                batch_size = len(X_batch)

                # Adversarial ground truths
                trues = np.ones((batch_size, 1))
                falses = np.zeros((batch_size, 1))

                noise = np.random.normal(0, 1, (batch_size, *self.latent_shape))
                # Generate images
                generated_images = self.generator.predict(noise)

                # Training the discriminator
                loss_real, accuracy_real = self.discriminator.train_on_batch(X_batch * 2 - 1, trues)
                loss_fake, accuracy_fake = self.discriminator.train_on_batch(generated_images, falses)

                d_loss = 0.5 * (loss_real + loss_fake)
                d_losses_epoch.append(d_loss)
                d_accuracy = 0.5 * (accuracy_real + accuracy_fake)
                d_accuracies_epoch.append(d_accuracy)

                # Training the generator
                noise = np.random.normal(0, 1, (batch_size, *self.latent_shape))
                g_loss = self.model.train_on_batch(noise, trues)
                g_losses_epoch.append(g_loss)

                self.logger.info("[%3d/%3d] [%3d/%3d] [D: loss: %f, acc: %.2f%%][G: loss: %f]", epoch, self.epochs, batch_num, batches, d_loss, d_accuracy * 100, g_loss)


            d_loss = np.mean(d_losses_epoch)
            d_accuracy = np.mean(d_accuracies_epoch)
            g_loss = np.mean(g_losses_epoch)

            logs = dict(d_loss = d_loss, d_accuracy = d_accuracy, g_loss = g_loss)
            self.write_log(logs, epoch)

            if epoch % self.config.vis_step == 0:
                self.save_images(epoch)

            
            if len(g_losses) == 0 or g_loss < np.min(g_losses) or epoch % self.config.checkpoint_step == 0:
                self.model.save_weights(self.gan_checkpoint_filename.format(epoch = epoch, **logs))

            d_losses.append(d_loss)
            d_accuracies.append(d_accuracy)
            g_losses.append(g_loss)

        
        return dict(d_loss = d_losses, d_accuracy = d_accuracies, g_loss = g_losses)



    def save_images(self, epoch):
        rows = cols = 5
        noise = np.random.normal(0, 1, (rows * cols, *self.latent_shape))
        generated_images = (self.generator.predict(noise) + 1) / 2
        
        grid = make_grid(generated_images)
        imageio.imwrite(os.path.join(self.logdir, "epoch_%d_grid.png" % epoch), grid)

        image = make_image(grid)
        self.writer.add_summary(tf.Summary(value = [tf.Summary.Value(tag = "generated_image", image = image)]), epoch)
        self.writer.flush()

