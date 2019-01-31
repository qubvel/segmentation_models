from .builder import build_psp
from ..utils import freeze_model
from ..utils import legacy_support
from ..backbones import get_backbone

PSP_BASE_LAYERS = {
    'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3'),
    'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4'),
    'resnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet152': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnext101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'inceptionv3': (228, 86, 16),
    'inceptionresnetv2': (594, 260, 16),
    'densenet121': (311, 139, 51),
    'densenet169': (367, 139, 51),
    'densenet201': (479, 139, 51),
}


def _get_layer_by_factor(backbone_name, factor):
    if factor == 4:
        return PSP_BASE_LAYERS[backbone_name][-1]
    elif factor == 8:
        return PSP_BASE_LAYERS[backbone_name][-2]
    elif factor == 16:
        return PSP_BASE_LAYERS[backbone_name][-3]
    else:
        raise ValueError('Unsupported factor - `{}`, Use 4, 8 or 16.'.format(factor))


def _shape_guard(factor, shape):
    h, w = shape[:2]
    min_size = factor * 6

    res = (h % min_size != 0 or w % min_size != 0 or
           h < min_size or w < min_size)
    if res:
        raise ValueError('Wrong shape {}, input H and W should '.format(shape) +
                         'be divisible by `{}`'.format(min_size))


old_args_map = {
    'freeze_encoder': 'encoder_freeze',
    'use_batchnorm': 'psp_use_batchnorm',
    'dropout': 'psp_dropout',
    'input_tensor': None,  # removed
}


@legacy_support(old_args_map)
def PSPNet(backbone_name='vgg16',
           input_shape=(384, 384, 3),
           classes=21,
           activation='softmax',
           encoder_weights='imagenet',
           encoder_freeze=False,
           downsample_factor=8,
           psp_conv_filters=512,
           psp_pooling_type='avg',
           psp_use_batchnorm=True,
           psp_dropout=None,
           final_interpolation='bilinear',
           **kwargs):
    """PSPNet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model used as feature
                extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``.
            ``H`` and ``W`` should be divisible by ``6 * downsample_factor`` and **NOT** ``None``!
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
                (e.g. ``sigmoid``, ``softmax``, ``linear``).
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        downsample_factor: one of 4, 8 and 16. Downsampling rate or in other words backbone depth
            to construct PSP module on it.
        psp_conv_filters: number of filters in ``Conv2D`` layer in each PSP block.
        psp_pooling_type: one of 'avg', 'max'. PSP block pooling type (maximum or average).
        psp_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.
        psp_dropout: dropout rate between 0 and 1.
        final_interpolation: ``duc`` or ``bilinear`` - interpolation type for final
            upsampling layer.

    Returns:
        ``keras.models.Model``: **PSPNet**

    .. _PSPNet:
        https://arxiv.org/pdf/1612.01105.pdf

    """

    # control image input shape
    _shape_guard(downsample_factor, input_shape)

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=None,
                            weights=encoder_weights,
                            include_top=False)

    psp_layer = _get_layer_by_factor(backbone_name, downsample_factor)

    model = build_psp(backbone,
                      psp_layer,
                      last_upsampling_factor=downsample_factor,
                      classes=classes,
                      conv_filters=psp_conv_filters,
                      pooling_type=psp_pooling_type,
                      activation=activation,
                      use_batchnorm=psp_use_batchnorm,
                      dropout=psp_dropout,
                      final_interpolation=final_interpolation)

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone)

    model.name = 'psp-{}'.format(backbone_name)

    return model
