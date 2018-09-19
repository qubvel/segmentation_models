import numpy as np
from keras.layers import MaxPool2D
from keras.layers import AveragePooling2D
from keras.layers import Concatenate
from keras.layers import Permute
from keras.layers import Reshape
from keras.backend import int_shape

from ..common import Conv2DBlock
from ..common import ResizeImage


def InterpBlock(level, feature_map_shape,
                conv_filters=512,
                conv_kernel_size=(1,1),
                conv_padding='same',
                pooling_type='avg',
                pool_padding='same',
                use_batchnorm=True,
                activation='relu',
                interpolation='bilinear'):

    if pooling_type == 'max':
        Pool2D = MaxPool2D
    elif pooling_type == 'avg':
        Pool2D = AveragePooling2D
    else:
        raise ValueError('Unsupported pooling type - `{}`.'.format(pooling_type) +
                         'Use `avg` or `max`.')

    def layer(input_tensor):
        # Compute the kernel and stride sizes according to how large the final feature map will be
        # When the kernel factor and strides are equal, then we can compute the final feature map factor
        # by simply dividing the current factor by the kernel or stride factor
        # The final feature map sizes are 1x1, 2x2, 3x3, and 6x6. We round to the closest integer
        pool_size = [int(np.round(feature_map_shape[0] / level)),
                       int(np.round(feature_map_shape[1] / level))]
        strides = pool_size

        x = Pool2D(pool_size, strides=strides, padding=pool_padding)(input_tensor)
        x = Conv2DBlock(conv_filters,
                        kernel_size=conv_kernel_size,
                        padding=conv_padding,
                        use_batchnorm=use_batchnorm,
                        activation=activation,
                        name='level{}'.format(level))(x)
        x = ResizeImage(strides, interpolation=interpolation)(x)
        return x
    return layer


def DUC(factor=(8, 8)):

    if factor[0] != factor[1]:
        raise ValueError('DUC upconvolution support only equal factors, '
                         'got {}'.format(factor))
    factor = factor[0]

    def layer(input_tensor):

        h, w, c = int_shape(input_tensor)[1:]
        H = h * factor
        W = w * factor

        x = Conv2DBlock(c*factor**2, (1,1),
                        padding='same',
                        name='duc_{}'.format(factor))(input_tensor)
        x = Permute((3, 1, 2))(x)
        x = Reshape((c, factor, factor, h, w))(x)
        x = Permute((1, 4, 2, 5, 3))(x)
        x = Reshape((c, H, W))(x)
        x = Permute((2, 3, 1))(x)
        return x
    return layer


def PyramidPoolingModule(**params):
    """
    Build the Pyramid Pooling Module.
    """

    _params = {
        'conv_filters': 512,
        'conv_kernel_size': (1, 1),
        'conv_padding': 'same',
        'pooling_type': 'avg',
        'pool_padding': 'same',
        'use_batchnorm': True,
        'activation': 'relu',
        'interpolation': 'bilinear',
    }

    _params.update(params)

    def module(input_tensor):

        feature_map_shape = int_shape(input_tensor)[1:3]

        x1 = InterpBlock(1, feature_map_shape, **_params)(input_tensor)
        x2 = InterpBlock(2, feature_map_shape, **_params)(input_tensor)
        x3 = InterpBlock(3, feature_map_shape, **_params)(input_tensor)
        x6 = InterpBlock(6, feature_map_shape, **_params)(input_tensor)

        x = Concatenate()([input_tensor, x1, x2, x3, x6])
        return x
    return module