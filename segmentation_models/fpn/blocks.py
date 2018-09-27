from keras.layers import Add

from ..common import Conv2DBlock
from ..common import ResizeImage
from ..utils import to_tuple


def pyramid_block(pyramid_filters=256, segmentation_filters=128, upsample_rate=2,
                  use_batchnorm=False, stage=0):
    """
    Pyramid block according to:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    This block generate `M` and `P` blocks.

    Args:
        pyramid_filters: integer, filters in `M` block of top-down FPN branch
        segmentation_filters: integer, number of filters in segmentation head,
            basically filters in convolution layers between `M` and `P` blocks
        upsample_rate: integer, uspsample rate for `M` block of top-down FPN branch
        use_batchnorm: bool, include batchnorm in convolution blocks

    Returns:
        Pyramid block function (as Keras layers functional API)
    """
    def layer(c, m=None):

        x = Conv2DBlock(pyramid_filters, (1, 1),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        name='pyramid_stage_{}'.format(stage))(c)

        if m is not None:
            up = ResizeImage(to_tuple(upsample_rate))(m)
            x = Add()([x, up])

        # segmentation head
        p = Conv2DBlock(segmentation_filters, (3, 3),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        name='segm1_stage_{}'.format(stage))(x)

        p = Conv2DBlock(segmentation_filters, (3, 3),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        name='segm2_stage_{}'.format(stage))(p)
        m = x

        return m, p
    return layer
