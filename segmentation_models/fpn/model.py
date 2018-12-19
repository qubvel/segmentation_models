from .builder import build_fpn
from ..backbones import get_backbone
from ..utils import freeze_model


DEFAULT_FEATURE_PYRAMID_LAYERS = {
    'vgg16':            ('block5_conv3', 'block4_conv3', 'block3_conv3'),
    'vgg19':            ('block5_conv4', 'block4_conv4', 'block3_conv4'),
    'resnet18':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet34':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet50':         ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet101':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnet152':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnext50':        ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'resnext101':       ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1'),
    'inceptionv3':          (228, 86, 16),
    'inceptionresnetv2':    (594, 260, 16),
    'densenet121':          (311, 139, 51),
    'densenet169':          (367, 139, 51),
    'densenet201':          (479, 139, 51),
}


def FPN(backbone_name='vgg16',
        input_shape=(None, None, 3),
        input_tensor=None,
        encoder_weights='imagenet',
        freeze_encoder=False,
        fpn_layers='default',
        pyramid_block_filters=256,
        segmentation_block_filters=128,
        upsample_rates=(2, 2, 2),
        last_upsample=4,
        interpolation='bilinear',
        use_batchnorm=True,
        classes=21,
        activation='softmax',
        dropout=None):
    """FPN_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
                case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
                able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model
                (works only if ``encoder_weights`` is ``None``).
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        freeze_encoder: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        fpn_layers: a list of layer numbers or names starting from top of the model.
                Each of these layers will be used to build features pyramid. If ``default`` is used
                layer names are taken from ``DEFAULT_FEATURE_PYRAMID_LAYERS``.
        pyramid_block_filters: a number of filters in Feature Pyramid Block of FPN_.
        segmentation_block_filters: a number of filters in Segmentation Head of FPN_.
        upsample_rates: list of rates for upsampling pyramid blocks (to make them same spatial resolution).
        last_upsample: rate for upsumpling concatenated pyramid predictions to
            match spatial resolution of input data.
        interpolation: interpolation type for upsampling layers, on of ``nearest``, ``bilinear``.
        use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.
        dropout: dropout rate between 0 and 1.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer (e.g. ``sigmoid``, ``softmax``, ``linear``).

    Returns:
        ``keras.models.Model``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=input_tensor,
                            weights=encoder_weights,
                            include_top=False)

    if fpn_layers == 'default':
        fpn_layers = DEFAULT_FEATURE_PYRAMID_LAYERS[backbone_name]

    model = build_fpn(backbone, fpn_layers,
                      classes=classes,
                      pyramid_filters=pyramid_block_filters,
                      segmentation_filters=segmentation_block_filters,
                      upsample_rates=upsample_rates,
                      use_batchnorm=use_batchnorm,
                      dropout=dropout,
                      last_upsample=last_upsample,
                      interpolation=interpolation,
                      activation=activation)

    if freeze_encoder:
        freeze_model(backbone)

    model.name = 'fpn-{}'.format(backbone.name)

    return model
