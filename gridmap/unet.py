from gridmap.model_base import *

class UNet(BaseModel):
   @property
   def input_shape(self):
       return (64, 64, 1)

   def conv_block(self, inputs, block_number, middle_channels, out_channels = None, is_reversing = False):
       """ Conv + ReLU 2 times """
       if out_channels is None:
           out_channels = middle_channels

       with K.name_scope("ConvBlock_%d%s" % (block_number, "_decode" if is_reversing else "")):
           conv_1 = Conv2D(middle_channels,
                           kernel_size = (3, 3),
                           strides = (1, 1),
                           padding = "same",
                           activation = "relu")(inputs)
           conv_2 = Conv2D(out_channels,
                           kernel_size = (3, 3),
                           strides = (1, 1),
                           padding = "same",
                           activation = "relu")(conv_1)

           return conv_2

   def stack_conv_blocks(self, inputs, filters, pooling_layer = None, apply_last = False):
       conv_block = inputs

       blocks = []
       intermediates = []
       for i, num_filters in enumerate(filters, 1):
           conv_block = self.conv_block(conv_block, i, num_filters)
           blocks.append(conv_block)
           if pooling_layer and (i < len(filters) or apply_last):
               conv_block = pooling_layer(conv_block)
               intermediates.append(conv_block)

       return conv_block, blocks, intermediates

           
   @property
   def left_conv_filters(self):
       return self.filter_series(num_filters_init = 64,
                                 growth_factor = 2,
                                 repeats = 5)

   def decoding_blocks(self, left_conv, left_blocks, left_filters):
       blocks = []
       # Dropping the first ones
       left_blocks_reversed = left_blocks[:-1][::-1]
       left_filters_reversed = left_filters[:-1][::-1]

       de_feature = left_conv
       for i, (num_filters, left_conv) in enumerate(zip(left_filters_reversed, left_blocks_reversed), 1):
           de_feature = Conv2DTranspose(num_filters,
                                        kernel_size = (3, 3),
                                        strides = (2, 2),
                                        padding = "same",
                                        output_padding = 1)(de_feature)
           de_feature = Concatenate(axis = -1)([left_conv, de_feature])
           de_feature = self.conv_block(de_feature, i, num_filters, is_reversing = True)

       # Finally a 1x1 Convolution
       de_feature = Conv2D(1,
                           kernel_size = (1, 1),
                           strides = (1, 1),
                           padding = "valid")(de_feature)

       de_feature = Activation("sigmoid", name = "ouput")(de_feature)

       return de_feature

   def prepare_model(self, inputs):
       # First define the left-half of the network
       left_conv, left_blocks, _ = self.stack_conv_blocks(inputs,
                                                          self.left_conv_filters,
                                                          pooling_layer = MaxPool2D((2, 2)),
                                                          apply_last = False)

       # Decoding 
       right_conv = self.decoding_blocks(left_conv, left_blocks, self.left_conv_filters)

       return right_conv
   
   @property
   def loss(self):
       return "binary_crossentropy"

