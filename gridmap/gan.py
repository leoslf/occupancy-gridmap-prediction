from gridmap.model_base import *

class GAN(BaseModel):
    @property
    def input_shape(self):
        return (64, 64, 1)

    @property
    def leaky_relu_alpha(self):
        return 0.2

    @property
    def bn_momentum(self):
        return 0.8

    @property
    def latent_shape(self):
        return (256,)

    @property
    def class_mode(self):
        return None

    def init(self):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.writer = tf.summary.FileWriter(self.logdir)

    def dense_block(self, units, inputs, block_number, bn = False):
        with K.name_scope("DenseBlock_%d" % block_number):
            dense = Dense(units)(inputs)
            dense = LeakyReLU(alpha = self.leaky_relu_alpha)(dense)
            if bn:
                dense = BatchNormalization(momentum = self.bn_momentum)(dense)
        return dense

    def enumerated_filter_series(self, *argv, **kwargs):
        return list(enumerate(self.filter_series(*argv, **kwargs), 1))

    def prepare_generator(self, noise_input):

        dense = noise_input
        for (i, units) in self.enumerated_filter_series(256, 2, 3):
            dense = self.dense_block(units, dense, i, bn = True)

        dense = Dense(np.prod(self.input_shape), activation = "tanh")(dense)
        output = Reshape(self.input_shape)(dense)

        return output

    def prepare_discriminator(self, image_input):

        dense = Flatten()(image_input)
        for (i, units) in reversed(self.enumerated_filter_series(256, 2, 2)):
            dense = self.dense_block(units, dense, i, bn = False)

        output = Dense(1, activation = "sigmoid")(dense)

        return output


    def construct_model(self):
        self.image_input = Input(self.input_shape)
        self.noise_input = Input(self.latent_shape)

        self.validity_layer = self.prepare_discriminator(self.image_input)
        self.discriminator = Model(self.image_input, self.validity_layer, name = "%s_discriminator" % self.name)

        self.image_layer = self.prepare_generator(self.noise_input)
        self.generator = Model(self.noise_input, self.image_layer, name = "%s_generator" % self.name)

        # Disable discrminiator training in combined model
        self.discriminator.trainable = False
        self.model = Model(self.noise_input, self.discriminator(self.generator(self.noise_input)), name = self.name)

    def compile(self):
        self.discriminator.compile(loss = self.loss, optimizer = self.optimizer, metrics = ["accuracy"])
        self.generator.compile(loss = self.loss, optimizer = self.optimizer)
        self.model.compile(loss = self.loss, optimizer = self.optimizer)

    def write_log(self, logs, epoch):
        for (name, value) in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.tag = name
            summary_value.simple_value = value
            self.writer.add_summary(summary, epoch)

    @property
    def checkpoint_filename(self):
        return os.path.join(self.logdir, "epoch{epoch:03d}-d_loss{d_loss:.3f}-d_accuracy{d_accuracy:.3f}-g_loss{g_loss:.3f}.h5" % dict(metric = self.main_metric))

    def fit_df(self):

        d_losses = []
        d_accuracies = []
        g_losses = []
    
        batches = steps_from_gen(self.train_generator)

        for epoch in range(self.epochs):
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

            d_losses.append(d_loss)
            d_accuracies.append(d_accuracy)
            g_losses.append(g_loss)

            logs = dict(d_loss = d_loss, d_accuracy = d_accuracy, g_loss = g_loss)
            self.write_log(logs, epoch)

            if epoch % self.config.vis_step == 0:
                self.save_images(epoch)

            self.model.save_weights(self.checkpoint_filename.format(epoch = epoch, **logs))
        
        return dict(d_loss = d_losses, d_accuracy = d_accuracies, g_loss = g_losses)



    def save_images(self, epoch):
        rows = cols = 5
        noise = np.random.normal(0, 1, (rows * cols, *self.latent_shape))
        generated_images = (self.generator.predict(noise) + 1) / 2
        
        grid = make_grid(generated_images)
        imageio.imwrite(os.path.join(self.logdir, "epoch_%d_grid.png" % epoch), grid)


