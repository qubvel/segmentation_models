from keras_applications import get_submodules_from_kwargs

from ._common_blocks import Conv2dBn
from ._utils import freeze_model, filter_keras_submodules
from ..backbones.backbones_factory import Backbones

backend = None
layers = None
models = None
keras_utils = None


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DoubleConv3x3BnReLU(filters, use_batchnorm, name=None):
    name1, name2 = None, None
    if name is not None:
        name1 = name + 'a'
        name2 = name + 'b'

    def wrapper(input_tensor):
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name1)(input_tensor)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name2)(x)
        return x

    return wrapper


def FPNBlock(pyramid_filters, stage):
    conv0_name = 'fpn_stage_p{}_pre_conv'.format(stage)
    conv1_name = 'fpn_stage_p{}_conv'.format(stage)
    add_name = 'fpn_stage_p{}_add'.format(stage)
    up_name = 'fpn_stage_p{}_upsampling'.format(stage)

    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip):
        # if input tensor channels not equal to pyramid channels
        # we will not be able to sum input tensor and skip
        # so add extra conv layer to transform it
        input_filters = backend.int_shape(input_tensor)[channels_axis]
        if input_filters != pyramid_filters:
            input_tensor = layers.Conv2D(
                filters=pyramid_filters,
                kernel_size=(1, 1),
                kernel_initializer='he_uniform',
                name=conv0_name,
            )(input_tensor)

        skip = layers.Conv2D(
            filters=pyramid_filters,
            kernel_size=(1, 1),
            kernel_initializer='he_uniform',
            name=conv1_name,
        )(skip)

        x = layers.UpSampling2D((2, 2), name=up_name)(input_tensor)
        x = layers.Add(name=add_name)([x, skip])

        return x

    return wrapper


# ---------------------------------------------------------------------
#  FPN Decoder
# ---------------------------------------------------------------------

def build_fpn(
        backbone,
        skip_connection_layers,
        pyramid_filters=256,
        segmentation_filters=128,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        aggregation='sum',
        dropout=None,
):
    input_ = backbone.input
    x = backbone.output

    # building decoder blocks with skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # build FPN pyramid
    p5 = FPNBlock(pyramid_filters, stage=5)(x, skips[0])
    p4 = FPNBlock(pyramid_filters, stage=4)(p5, skips[1])
    p3 = FPNBlock(pyramid_filters, stage=3)(p4, skips[2])
    p2 = FPNBlock(pyramid_filters, stage=2)(p3, skips[3])

    # add segmentation head to each
    s5 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage2')(p2)

    # upsampling to same resolution
    s5 = layers.UpSampling2D((8, 8), interpolation='nearest', name='upsampling_stage5')(s5)
    s4 = layers.UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage4')(s4)
    s3 = layers.UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage3')(s3)

    # aggregating results
    if aggregation == 'sum':
        x = layers.Add(name='aggregation_sum')([s2, s3, s4, s5])
    elif aggregation == 'concat':
        concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x = layers.Concatenate(axis=concat_axis, name='aggregation_concat')([s2, s3, s4, s5])
    else:
        raise ValueError('Aggregation parameter should be in ("sum", "concat"), '
                         'got {}'.format(aggregation))

    if dropout:
        x = layers.SpatialDropout2D(dropout, name='pyramid_dropout')(x)

    # final stage
    x = Conv3x3BnReLU(segmentation_filters, use_batchnorm, name='final_stage')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='final_upsampling')(x)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  FPN Model
# ---------------------------------------------------------------------

def FPN(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=21,
        activation='softmax',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        pyramid_block_filters=256,
        pyramid_use_batchnorm=True,
        pyramid_aggregation='concat',
        pyramid_dropout=None,
        **kwargs
):
    """FPN_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
                case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
                able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        weights: optional, path to model weights.
        activation: name of one of ``keras.activations`` for last model layer (e.g. ``sigmoid``, ``softmax``, ``linear``).
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
                Each of these layers will be used to build features pyramid. If ``default`` is used
                layer names are taken from ``DEFAULT_FEATURE_PYRAMID_LAYERS``.
        pyramid_block_filters: a number of filters in Feature Pyramid Block of FPN_.
        pyramid_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.
        pyramid_aggregation: one of 'sum' or 'concat'. The way to aggregate pyramid blocks.
        pyramid_dropout: spatial dropout rate for feature pyramid in range (0, 1).

    Returns:
        ``keras.models.Model``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """
    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_fpn(
        backbone=backbone,
        skip_connection_layers=encoder_features,
        pyramid_filters=pyramid_block_filters,
        segmentation_filters=pyramid_block_filters // 2,
        use_batchnorm=pyramid_use_batchnorm,
        dropout=pyramid_dropout,
        activation=activation,
        classes=classes,
        aggregation=pyramid_aggregation,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
