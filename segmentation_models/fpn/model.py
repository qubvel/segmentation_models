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
    """
    Implementation of FPN head for segmentation models according to:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    Args:
        backbone_name: (str) see available backbones
        classes: (int) a number of classes for output
        input_shape: (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        fpn_layers: (list) of layer names or indexes, used for pyramid
        pyramid_block_filters: (int) number of filters in `M` blocks of top-down FPN branch
        segmentation_block_filters: (int) number of filters in `P` blocks of FPN
        upsample_rates: (tuple) rates for upsampling pyramid blocks
        last_upsample: (int) rate for upsumpling concatenated pyramid predictions to
            match spatial resolution of input data
        interpolation: (str) interpolation type for upsampling layers - 'nearest' or 'bilinear'
        use_batchnorm: (bool) if True add batch normalisation layer between `Conv2D` ad `Activation` layers
        activation: (str) one of keras activations
        dropout: None or float [0, 1), dropout rate

    Returns:
        keras.models.Model

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
