import keras.backend as K
from keras.layers import Conv2DTranspose as Transpose
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Add


def handle_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def ConvRelu(filters,
             kernel_size,
             use_batchnorm=False,
             conv_name='conv',
             bn_name='bn',
             relu_name='relu'):

    def layer(x):

        x = Conv2D(filters,
                   kernel_size,
                   padding="same",
                   name=conv_name,
                   use_bias=not(use_batchnorm))(x)

        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)

        x = Activation('relu', name=relu_name)(x)

        return x
    return layer


def Conv2DUpsample(filters,
                   upsample_rate,
                   kernel_size=(3,3),
                   up_name='up',
                   conv_name='conv',
                   **kwargs):

    def layer(input_tensor):
        x = UpSampling2D(upsample_rate, name=up_name)(input_tensor)
        x = Conv2D(filters,
                   kernel_size,
                   padding='same',
                   name=conv_name,
                   **kwargs)(x)
        return x
    return layer


def Conv2DTranspose(filters,
                    upsample_rate,
                    kernel_size=(4,4),
                    up_name='up',
                    **kwargs):

    if not tuple(upsample_rate) == (2,2):
        raise NotImplementedError(
            f'Conv2DTranspose support only upsample_rate=(2, 2), got {upsample_rate}')

    def layer(input_tensor):
        x = Transpose(filters,
                      kernel_size=kernel_size,
                      strides=upsample_rate,
                      padding='same',
                      name=up_name)(input_tensor)
        return x
    return layer


def UpsampleBlock(filters,
                  upsample_rate,
                  kernel_size,
                  use_batchnorm=False,
                  upsample_layer='upsampling',
                  conv_name='conv',
                  bn_name='bn',
                  relu_name='relu',
                  up_name='up',
                  **kwargs):

    if upsample_layer == 'upsampling':
        UpBlock = Conv2DUpsample

    elif upsample_layer == 'transpose':
        UpBlock = Conv2DTranspose

    else:
        raise ValueError(f'Not supported up layer type {upsample_layer}')

    def layer(input_tensor):

        x = UpBlock(filters,
                    upsample_rate=upsample_rate,
                    kernel_size=kernel_size,
                    use_bias=not(use_batchnorm),
                    conv_name=conv_name,
                    up_name=up_name,
                    **kwargs)(input_tensor)

        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)

        x = Activation('relu', name=relu_name)(x)

        return x
    return layer


def DecoderBlock(stage,
                 filters=None,
                 kernel_size=(3,3),
                 upsample_rate=(2,2),
                 use_batchnorm=False,
                 skip=None,
                 upsample_layer='upsampling'):

    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)
        input_filters = K.int_shape(input_tensor)[-1]

        if skip is not None:
            output_filters = K.int_shape(skip)[-1]
        else:
            output_filters = filters

        x = ConvRelu(input_filters // 4,
                     kernel_size=(1, 1),
                     use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1',
                     bn_name=bn_name + '1',
                     relu_name=relu_name + '1')(input_tensor)

        x = UpsampleBlock(filters=input_filters // 4,
                          kernel_size=kernel_size,
                          upsample_layer=upsample_layer,
                          upsample_rate=upsample_rate,
                          use_batchnorm=use_batchnorm,
                          conv_name=conv_name + '2',
                          bn_name=bn_name + '2',
                          up_name=up_name + '2',
                          relu_name=relu_name + '2')(x)

        x = ConvRelu(output_filters,
                     kernel_size=(1, 1),
                     use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '3',
                     bn_name=bn_name + '3',
                     relu_name=relu_name + '3')(x)

        if skip is not None:
            x = Add()([x, skip])

        return x
    return layer
