from .builder import build_linknet
from ..utils import freeze_model
from ..backbones import get_backbone


DEFAULT_SKIP_CONNECTIONS = {
    'vgg16':                ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2'),
    'vgg19':                ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2'),
    'resnet18':             ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet34':             ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet50':             ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet101':            ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet152':            ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnext50':            ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnext101':           ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'inceptionv3':          (228, 86, 16, 9),
    'inceptionresnetv2':    (594, 260, 16, 9),
    'densenet121':          (311, 139, 51, 4),
    'densenet169':          (367, 139, 51, 4),
    'densenet201':          (479, 139, 51, 4),
}


def Linknet(backbone_name='vgg16',
            input_shape=(None, None, 3),
            input_tensor=None,
            encoder_weights='imagenet',
            freeze_encoder=False,
            skip_connections='default',
            n_upsample_blocks=5,
            decoder_filters=(None, None, None, None, 16),
            decoder_use_batchnorm=True,
            upsample_layer='upsampling',
            upsample_kernel_size=(3, 3),
            classes=1,
            activation='sigmoid'):
    """
    Version of Linkent model (https://arxiv.org/pdf/1707.03718.pdf)
    This implementation by default has 4 skip connection links (original - 3).

    Args:
        backbone_name: (str) look at list of available backbones.
        input_shape: (tuple) dimensions of input data (H, W, C)
        input_tensor: keras tensor
        encoder_weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet)
        freeze_encoder: (bool) Set encoder layers weights as non-trainable. Useful for fine-tuning
        skip_connections: if 'default' is used take default skip connections,
        decoder_filters: (tuple of int) a number of convolution filters in decoder blocks,
            for block with skip connection a number of filters is equal to number of filters in
            corresponding encoder block (estimates automatically and can be passed as `None` value).
        decoder_use_batchnorm: (bool) if True add batch normalisation layer between `Conv2D` ad `Activation` layers
        n_upsample_blocks: (int) a number of upsampling blocks in decoder
        upsample_layer: (str) one of 'upsampling' and 'transpose'
        upsample_kernel_size: (tuple of int) convolution kernel size in upsampling block
        classes: (int) a number of classes for output
        activation: (str) one of keras activations

    Returns:
        model: instance of Keras Model

    """

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            input_tensor=input_tensor,
                            weights=encoder_weights,
                            include_top=False)

    if skip_connections == 'default':
        skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name]

    model = build_linknet(backbone,
                          classes,
                          skip_connections,
                          decoder_filters=decoder_filters,
                          upsample_layer=upsample_layer,
                          activation=activation,
                          n_upsample_blocks=n_upsample_blocks,
                          upsample_rates=(2, 2, 2, 2, 2),
                          upsample_kernel_size=upsample_kernel_size,
                          use_batchnorm=decoder_use_batchnorm)

    # lock encoder weights for fine-tuning
    if freeze_encoder:
        freeze_model(backbone)

    model.name = 'link-{}'.format(backbone_name)

    return model
