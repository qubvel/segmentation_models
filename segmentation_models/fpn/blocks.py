from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Add
from keras.layers import Activation
from keras.layers import BatchNormalization


def Conv(n_filters, kernel_size, activation='relu', batchnorm=False, **kwargs):
    """Extension of Conv2D layer with batchnorm"""
    def layer(input_tensor):

        x = Conv2D(n_filters, kernel_size, **kwargs)(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)

        return x
    return layer


def pyramid_block(pyramid_filters=256, segmentation_filters=128, upsample_rate=2):

    def layer(c, m=None):

        x = Conv2D(pyramid_filters, (1, 1))(c)

        if m is not None:
            up = UpSampling2D((upsample_rate, upsample_rate))(m)
            x = Add()([x, up])

        # segmentation head
        p = Conv(segmentation_filters, (3, 3), padding='same')(x)
        p = Conv(segmentation_filters, (3, 3), padding='same')(p)
        m = x

        return m, p
    return layer
