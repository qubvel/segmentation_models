"""
Code is constructed based on following repositories:
    https://github.com/ykamikawa/PSPNet/
    https://github.com/hujh14/PSPNet-Keras/
    https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow/

And original paper of PSPNet:
    https://arxiv.org/pdf/1612.01105.pdf
"""

from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import SpatialDropout2D
from keras.models import Model

from .blocks import PyramidPoolingModule, DUC
from ..common import Conv2DBlock
from ..common import ResizeImage
from ..utils import extract_outputs
from ..utils import to_tuple


def build_psp(backbone,
              psp_layer,
              last_upsampling_factor,
              classes=21,
              activation='softmax',
              conv_filters=512,
              pooling_type='avg',
              dropout=None,
              final_interpolation='bilinear',
              use_batchnorm=True):

    input = backbone.input

    x = extract_outputs(backbone, [psp_layer])[0]

    x = PyramidPoolingModule(
        conv_filters=conv_filters,
        pooling_type=pooling_type,
        use_batchnorm=use_batchnorm)(x)

    x = Conv2DBlock(512, (1, 1), activation='relu', padding='same',
                    use_batchnorm=use_batchnorm)(x)

    if dropout is not None:
        x = SpatialDropout2D(dropout)(x)

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)

    if final_interpolation == 'bilinear':
        x = ResizeImage(to_tuple(last_upsampling_factor))(x)
    elif final_interpolation == 'duc':
        x = DUC(to_tuple(last_upsampling_factor))(x)
    else:
        raise ValueError('Unsupported interpolation type {}. '.format(final_interpolation) +
                         'Use `duc` or `bilinear`.')

    x = Activation(activation, name=activation)(x)

    model = Model(input, x)

    return model
