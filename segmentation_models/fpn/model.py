from .builder import build_fpn
from ..backbones import get_backbone, get_feature_layers
from ..utils import freeze_model
from ..utils import legacy_support

old_args_map = {
    'freeze_encoder': 'encoder_freeze',
    'fpn_layers': 'encoder_features',
    'use_batchnorm': 'pyramid_use_batchnorm',
    'dropout': 'pyramid_dropout',
    'interpolation': 'final_interpolation',
    'upsample_rates': None,  # removed
    'last_upsample': None,  # removed
}


@legacy_support(old_args_map)
def FPN(backbone_name='vgg16',
        input_shape=(None, None, 3),
        input_tensor=None,
        classes=21,
        activation='softmax',
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        pyramid_block_filters=256,
        pyramid_use_batchnorm=True,
        pyramid_dropout=None,
        final_interpolation='bilinear',
        **kwargs):
    """FPN_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
                case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
                able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model
                (works only if ``encoder_weights`` is ``None``).
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer (e.g. ``sigmoid``, ``softmax``, ``linear``).
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
                Each of these layers will be used to build features pyramid. If ``default`` is used
                layer names are taken from ``DEFAULT_FEATURE_PYRAMID_LAYERS``.
        pyramid_block_filters: a number of filters in Feature Pyramid Block of FPN_.
        pyramid_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.
        pyramid_dropout: spatial dropout rate for feature pyramid in range (0, 1).
        final_interpolation: interpolation type for upsampling layers, on of ``nearest``, ``bilinear``.

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

    if encoder_features == 'default':
        encoder_features = get_feature_layers(backbone_name, n=3)

    upsample_rates = [2] * len(encoder_features)
    last_upsample = 2 ** (5 - len(encoder_features))

    model = build_fpn(backbone, encoder_features,
                      classes=classes,
                      pyramid_filters=pyramid_block_filters,
                      segmentation_filters=pyramid_block_filters // 2,
                      upsample_rates=upsample_rates,
                      use_batchnorm=pyramid_use_batchnorm,
                      dropout=pyramid_dropout,
                      last_upsample=last_upsample,
                      interpolation=final_interpolation,
                      activation=activation)

    if encoder_freeze:
        freeze_model(backbone)

    model.name = 'fpn-{}'.format(backbone.name)

    return model
