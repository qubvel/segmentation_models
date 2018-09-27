from .builder import build_psp
from ..utils import freeze_model
from ..backbones import get_backbone


PSP_BASE_LAYERS = {
    'vgg16':                ('block5_conv3', 'block4_conv3', 'block3_conv3'),
    'vgg19':                ('block5_conv4', 'block4_conv4', 'block3_conv4'),
    'resnet18':             ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet34':             ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet50':             ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet101':            ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet152':            ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnext50':            ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnext101':           ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'inceptionv3':          (228, 86, 16),
    'inceptionresnetv2':    (594, 260, 16),
    'densenet121':          (311, 139, 51),
    'densenet169':          (367, 139, 51),
    'densenet201':          (479, 139, 51),
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


def PSPNet(backbone_name='vgg16',
           input_shape=(384, 384, 3),
           input_tensor=None,
           encoder_weights='imagenet',
           freeze_encoder=False,
           downsample_factor=8,
           psp_conv_filters=512,
           psp_pooling_type='avg',
           use_batchnorm=True,
           dropout=None,
           final_interpolation='bilinear',
           classes=21,
           activation='softmax'):
    """
    Exploit the capability of global context information by different-regionbased
    context aggregation through pyramid pooling module together with the proposed
    pyramid scene parsing network (PSPNet).

    https://arxiv.org/pdf/1612.01105.pdf

    Args:
        backbone_name: (str) look at list of available backbones.
        input_shape: (tuple) dimensions of input data (H, W, C).
            H and W should be divisible by (6 * `downsample_factor`) and **NOT** `None`!
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization), 'imagenet' (pre-
            training on ImageNet)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Use-
            ful for fine-tuning
        downsample_factor: int, one of 4, 8 and 16. Specify layer of backbone or
            backbone depth to construct PSP module on it.
        psp_conv_filters: (int), number of filters in `Conv2D` layer in each psp block
        psp_pooling_type: 'avg' or 'max', psp block pooling type (maximum or average)
        use_batchnorm: (bool) if True add batch normalisation layer between
            `Conv2D` ad `Activation` layers
        dropout: None or float in range 0-1, if specified add SpatialDropout after PSP module
        final_interpolation: 'duc' or 'bilinear' - interpolation type for final
            upsampling layer.
        classes: (int) a number of classes for output
        activation: (str) one of keras activations


    Returns:
        keras Model instance
    """

    # control image input shape
    _shape_guard(downsample_factor, input_shape)

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=input_tensor,
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
                      use_batchnorm=use_batchnorm,
                      dropout=dropout,
                      final_interpolation=final_interpolation)

    # lock encoder weights for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    model.name = 'psp-{}'.format(backbone_name)

    return model