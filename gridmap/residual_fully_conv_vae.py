from gridmap.model_base import *

class ResidualFullyConvVAE(BaseModel):

    def init(self):
        self.encoder = Model(self.input_layer, self.latent_encoding_input_layer, name = "%s_encoder" % self.name)
        self.decoder = Model(Input(shape = (self.latent_encoding_channels, )), self.output_layer, name = "%s_decoder" % self.name)


    @property
    def shortcircuit_connection_type(self):
        return "concat"

    @property
    def shortcircuit_connection_channel_expansion(self):
        return 2 if self.shortcircuit_connection_type == "concat" else 1

    def conv_bn_block(self, inputs, num_filters, block_number):
        with K.name_scope("ConvBNBlock_%d" % block_number):
            conv = Conv2D(num_filters,
                          kernel_size = (3, 3),
                          strides = (2, 2),
                          padding = "same",
                          name = "conv")
            bn = BatchNormalization()(conv)
            activation = LeakyReLU()(bn)

        return activation

    def deconv_bn_block(self, inputs, num_filters, block_number):
        with K.name_scope("DeConvBNBlock_%d" % block_number):
            deconv = Conv2dTranspose(num_filters,
                                     kernel_size = (3, 3),
                                     strides = (2, 2),
                                     padding = "same",
                                     output_padding = 1)(inputs)
            bn = BatchNormalization()(deconv)
            activation = LeakyReLU()(bn)

        return activation

    @property
    def latent_encoding_channels(self):
        return 16

    def prepare_model(self, inputs):
        conv = inputs
        filters = self.filter_series(8, 2, 3)
        forward_layers = [inputs]
        for i, num_filters in enumerate(filters, 1):
            conv = self.conv_bn_block(conv, num_filters, i)
            forward_layers.append(conv)
        
        self.latent_encoding_input_layer = conv

        # Latent Encoding
        latent_mean_conv = Conv2D(self.latent_encoding_channels,
                                  kernel_size = (3, 3),
                                  strides = (2, 2),
                                  padding = "same")(conv)

        latent_mean_bn = BatchNormalization()(latent_mean_conv)
        latent_mean_activation = LeakyReLU(name = "latent_mean")(latent_mean_bn)

        latent_logvariance_conv = Conv2D(self.latent_encoding_channels,
                                         kernel_size = (3, 3),
                                         strides = (2, 2),
                                         padding = "same")
        latent_logvariance_bn = BatchNormalization()(latent_logvariance_conv)
        latent_logvariance_activation = LeakyReLU(name = "latent_logvariance")(latent_logvariance_bn)

        # Reparametrize
        reparametrized_latent = self.reparametrize(latent_mean_activation, latent_logvariance_activation)

        # Decoder
        deconv = reparametrized_latent
        for i, num_filters in reversed(enumerate(filters, 1)):
            if self.shortcircuit_connection_type = "concat":
                deconv = Concatenate(axis = -1)([deconv, forward_layers[i - 1]])
            else:
                deconv = Add()([deconv, forward_layers[i - 1]])

            deconv = self.deconv_bn_block(deconv, num_filters, i)

        deconv = Conv2DTranspose(1,
                                 kernel_size = (3, 3),
                                 strides = (1, 1),
                                 padding = "same",
                                 output_padding = 0)(deconv)
        deconv = BatchNormalization()(deconv)
        deconv = Activation("sigmoid")(deconv)
        return deconv



