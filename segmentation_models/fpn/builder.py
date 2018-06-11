from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.models import Model

from .blocks import pyramid_block


def build_fpn(backbone, skips, classes=21, activation='softmax', upsample_rates=(2,2,2),
              pyramid_filters=256, segmentation_filters=128, last_upsample=4):

    if len(upsample_rates) != len(skips):
        raise ValueError('Number of intermediate feature maps and upsample steps should match')

    feature_maps = [backbone.layers[i].output for i in skips]
    feature_maps.insert(0, backbone.output)
    upsample_rates =  list(upsample_rates)
    upsample_rates.insert(0, 1)

    # top - down path, build pyramid
    m = None
    pyramid = []
    for i, c in enumerate(feature_maps):
        m, p = pyramid_block(pyramid_filters=pyramid_filters,
                            segmentation_filters=segmentation_filters,
                            upsample_rate=upsample_rates[i])(c, m)
        pyramid.append(p)


    # upsample and concatenate all pyramid layer
    upsampled_pyramid = []
    upsample_rate = 1

    for i, p in enumerate(pyramid[::-1]):
        upsample_rate *= upsample_rates[i]
        if upsample_rate > 1:
            p = UpSampling2D(size=(upsample_rate,upsample_rate))(p)
        upsampled_pyramid.append(p)

    x = Concatenate()(upsampled_pyramid)

    # final convolution
    x = Conv2D(classes, (3, 3), padding='same')(x)
    x = Activation(activation)(x)
    x = UpSampling2D(size=(last_upsample,last_upsample))(x)

    model = Model(backbone.input, x)
    return model