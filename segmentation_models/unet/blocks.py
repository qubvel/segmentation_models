from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate


def Upsample2D_block(filters, kernel_size=(3,3), upsample_rate=(2,2),
                     batchnorm=False, skip=None):

    def layer(input_tensor):

        x = UpSampling2D(size=upsample_rate)(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv2D(filters, kernel_size, padding='same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x
    return layer


def Transpose2D_block(filters, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), batchnorm=False, skip=None):

    def layer(input_tensor):

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = Conv2D(filters, kernel_size, padding='same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x
    return layer