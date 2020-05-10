from gridmap.model_base import *

class ResidualFullyConvVAE(BaseModel):
    @property
    def input_shape(self):
        return (64, 64, 1)

    def init(self):
        self.encoder = Model(self.input_layer, [self.latent_mu_layer, self.latent_logvariance_layer, self.latent_representation_layer], name = "%s_encoder" % self.name)
        self.decoder_input = Input(shape = (self.latent_encoding_channels, ), name = "decoder_input")
        self.decoder = Model(self.decoder_input, self.model.layers[-1](self.decoder_input), name = "%s_decoder" % self.name)

        # self.model = Model(self.input_layer, self.decoder(self.encoder(self.input_layer)[2]), name = self.name)

    def custom_loss(self, ground_truth, generated_images):
        dimensions = np.prod(self.input_shape)
        reconstruction_loss = binary_crossentropy(ground_truth, generated_images) 
        reconstruction_loss = K.sum(reconstruction_loss)
        # D_KL(N(\mu, \Sigma) || N(0, 1)) = -\frac{1}{2} \sum_{k} (1 + \Sigma - (\mu)^2 - \exp{\Sigma})
        kl_divergence = 1 + self.latent_logvariance_layer - K.square(self.latent_mu_layer) - K.exp(self.latent_logvariance_layer)
        # self.logger.info("kl_divergence: %r", K.int_shape(kl_divergence))
        kl_divergence = -0.5 * K.sum(kl_divergence)
        return reconstruction_loss + kl_divergence

    @property
    def loss(self):
        return self.custom_loss

    @property
    def optimizer(self):
        return "adam" # Adadelta() # Nadam() # "adam"

    @property
    def input_name(self):
        return "encoder_input"

    @property
    def shortcircuit_connection_type(self):
        return "concat"

    # @property
    # def class_mode(self):
    #     return "input" 

    @property
    def class_mode(self):
         return "image" 

    @property
    def use_predictionvisualizer(self):
        return True

    @property
    def shortcircuit_connection_channel_expansion(self):
        return 2 if self.shortcircuit_connection_type == "concat" else 1

    def conv_bn_block(self, inputs, num_filters, block_number):
        with K.name_scope("ConvBNBlock_%d" % block_number):
            conv = Conv2D(num_filters,
                          kernel_size = (3, 3),
                          strides = (2, 2),
                          padding = "same",
                          kernel_initializer = self.kernel_init,
                          kernel_regularizer = self.regularizer)(inputs)
            bn = BatchNormalization()(conv)
            activation = LeakyReLU()(bn)

        return activation

    def deconv_bn_block(self, inputs, num_filters, block_number, forward_layers):
        with K.name_scope("DeConvBNBlock_%d" % block_number):
            deconv = Conv2DTranspose(num_filters,
                                     kernel_size = (3, 3),
                                     strides = (2, 2),
                                     padding = "same",
                                     output_padding = 1,
                                     kernel_initializer = self.kernel_init,
                                     kernel_regularizer = self.regularizer)(inputs)
            deconv = BatchNormalization()(deconv)

            # self.logger.info("deconv: %r, forward_layers[%d]: %r", K.int_shape(deconv), block_number - 1, K.int_shape(forward_layers[block_number - 1]))
            if self.shortcircuit_connection_type == "concat":
                deconv = concatenate([deconv, forward_layers[block_number - 1]], axis = -1)
            else:
                deconv = add([deconv, forward_layers[block_number - 1]])

            activation = LeakyReLU()(deconv)

        return activation

    @property
    def latent_encoding_channels(self):
        return 16

    def prepare_model(self, inputs):
        conv = inputs
        filters = self.filter_series(8, 2, 3)
        forward_layers = [inputs]

        with K.name_scope("Encoder"):
            for i, num_filters in enumerate(filters, 1):
                conv = self.conv_bn_block(conv, num_filters, i)
                forward_layers.append(conv)
        
            self.latent_encoding_input_layer = conv

            # Latent Encoding
            latent_mean_conv = Conv2D(self.latent_encoding_channels,
                                      kernel_size = (3, 3),
                                      strides = (2, 2),
                                      padding = "same",
                                      kernel_initializer = self.kernel_init,
                                      kernel_regularizer = self.regularizer)(self.latent_encoding_input_layer)

            latent_mean_bn = BatchNormalization()(latent_mean_conv)
            latent_mean_activation = LeakyReLU(name = "latent_mean")(latent_mean_bn)
            self.latent_mu_layer = latent_mean_activation

            latent_logvariance_conv = Conv2D(self.latent_encoding_channels,
                                             kernel_size = (3, 3),
                                             strides = (2, 2),
                                             padding = "same",
                                             kernel_initializer = self.kernel_init,
                                             kernel_regularizer = self.regularizer)(self.latent_encoding_input_layer)
            latent_logvariance_bn = BatchNormalization()(latent_logvariance_conv)
            latent_logvariance_activation = LeakyReLU(name = "latent_logvariance")(latent_logvariance_bn)
            self.latent_logvariance_layer = latent_logvariance_activation

        # Reparametrize
        reparametrized_latent = Lambda(self.reparametrize, name="latent_representation")([latent_mean_activation, latent_logvariance_activation])
        self.latent_representation_layer = reparametrized_latent

        # self.logger.info("forward_layers: %r" % list(map(lambda layer: K.int_shape(layer), forward_layers)))

        # Decoder
        deconv = reparametrized_latent
        decode_filters = [1] + filters # [:-1]

        with K.name_scope("Decoder"):
            for i, num_filters in zip(reversed(range(len(decode_filters))), reversed(decode_filters)):
                deconv = self.deconv_bn_block(deconv, num_filters, i + 1, forward_layers)

            deconv = Conv2DTranspose(1,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     padding = "same",
                                     output_padding = 0,
                                     kernel_initializer = self.kernel_init,
                                     kernel_regularizer = self.regularizer)(deconv)
            deconv = BatchNormalization()(deconv)
            deconv = Activation("sigmoid", name = "decoder_output")(deconv)
            return deconv

    def reparametrize(self, argv):
        latent_mean, latent_logvariance = argv
        batch = K.shape(latent_mean)[0]
        dim = K.int_shape(latent_mean)[1:]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape = [batch] + list(dim))
        return latent_mean + K.exp(0.5 * latent_logvariance) * epsilon

