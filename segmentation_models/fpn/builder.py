from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.models import Model

from .blocks import pyramid_block
from ..utils import extract_outputs


def build_fpn(backbone, layers, classes=21, activation='softmax', upsample_rates=(2,2,2),
              pyramid_filters=256, segmentation_filters=128, last_upsample=4):
    """
    Implementation of FPN head for segmentation models according to:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    Args:
        backbone: Keras `Model`, some classification model without top
        layers: list of layer names or indexes, used for pyramid building
        classes: int, number of output feature maps
        activation: activation in last layer, e.g. 'sigmoid' or 'softmax'
        upsample_rates: tuple of integers, scaling rates between pyramid blocks
        pyramid_filters: int, number of filters in `M` blocks of top-down FPN branch
        segmentation_filters: int, number of filters in `P` blocks of FPN
        last_upsample: rate for upsumpling concatenated pyramid predictions to
            match spatial resolution of input data

    Returns:
        model: Keras `Model`
    """

    if len(upsample_rates) != len(layers):
        raise ValueError('Number of intermediate feature maps and upsample steps should match')

    # extract model layer outputs
    outputs = extract_outputs(backbone, layers, include_top=True)

    # add upsample rate `1` for first block
    upsample_rates =  list(upsample_rates)
    upsample_rates.insert(0, 1)

    # top - down path, build pyramid
    m = None
    pyramid = []
    for i, c in enumerate(outputs):
        m, p = pyramid_block(pyramid_filters=pyramid_filters,
                            segmentation_filters=segmentation_filters,
                            upsample_rate=upsample_rates[i])(c, m)
        pyramid.append(p)


    # upsample and concatenate all pyramid layer
    upsampled_pyramid = []
    upsample_rate = 1               # initial upsample rate

    for i, p in enumerate(pyramid[::-1]):
        upsample_rate *= upsample_rates[i]
        if upsample_rate > 1:
            p = UpSampling2D(size=(upsample_rate,upsample_rate))(p)
        upsampled_pyramid.append(p)

    x = Concatenate()(upsampled_pyramid)

    # final convolution
    x = Conv2D(classes, (3, 3), padding='same')(x)
    x = Activation(activation)(x)

    if last_upsample > 1:
        x = UpSampling2D(size=(last_upsample,last_upsample))(x)

    model = Model(backbone.input, x)
    return model