from .builder import build_linknet
from ..utils import freeze_model
from ..utils import legacy_support
from ..backbones import get_backbone, get_feature_layers

old_args_map = {
    'freeze_encoder': 'encoder_freeze',
    'skip_connections': 'encoder_features',
    'upsample_layer': 'decoder_block_type',
    'n_upsample_blocks': None,  # removed
    'input_tensor': None,  # removed
    'upsample_kernel_size': None,  # removed
}


@legacy_support(old_args_map)
def Linknet(backbone_name='vgg16',
            input_shape=(None, None, 3),
            classes=1,
            activation='sigmoid',
            encoder_weights='imagenet',
            encoder_freeze=False,
            encoder_features='default',
            decoder_filters=(None, None, None, None, 16),
            decoder_use_batchnorm=True,
            decoder_block_type='upsampling',
            **kwargs):
    """Linknet_ is a fully convolution neural network for fast image semantic segmentation

    Note:
        This implementation by default has 4 skip connections (original - 3).

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
                    extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
                case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
                able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
                    Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
                    layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks,
            for block with skip connection a number of filters is equal to number of filters in
            corresponding encoder block (estimates automatically and can be passed as ``None`` value).
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                    is used.
        decoder_block_type: one of
                    - `upsampling`:  use ``Upsampling2D`` keras layer
                    - `transpose`:   use ``Transpose2D`` keras layer

    Returns:
        ``keras.models.Model``: **Linknet**

    .. _Linknet:
        https://arxiv.org/pdf/1707.03718.pdf
    """

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=None,
                            weights=encoder_weights,
                            include_top=False)

    if encoder_features == 'default':
        encoder_features = get_feature_layers(backbone_name, n=4)

    model = build_linknet(backbone,
                          classes,
                          encoder_features,
                          decoder_filters=decoder_filters,
                          upsample_layer=decoder_block_type,
                          activation=activation,
                          n_upsample_blocks=len(decoder_filters),
                          upsample_rates=(2, 2, 2, 2, 2),
                          upsample_kernel_size=(3, 3),
                          use_batchnorm=decoder_use_batchnorm)

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone)

    model.name = 'link-{}'.format(backbone_name)

    return model
