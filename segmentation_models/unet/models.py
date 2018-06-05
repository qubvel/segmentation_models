from .builder import build_unet
from ..backbones import ResNet18
from ..backbones import ResNet34
from ..backbones import ResNet50
from ..backbones import ResNet101
from ..backbones import ResNet152

resnet_skips = ['bn_data', 'relu0', 'stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1'][::-1]


def UResNet18(input_shape, classes, decoder_filters=16, decoder_block_type='upsampling',
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet18(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_unet(backbone, classes, decoder_filters,
                       resnet_skips, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet18'

    return model


def UResNet34(input_shape, classes, decoder_filters=16, decoder_block_type='upsampling',
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet34(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_unet(backbone, classes, decoder_filters,
                       resnet_skips, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet34'

    return model


def UResNet50(input_shape, classes, decoder_filters=16, decoder_block_type='upsampling',
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet50(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_unet(backbone, classes, decoder_filters,
                       resnet_skips, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet50'

    return model


def UResNet101(input_shape, classes, decoder_filters=16, decoder_block_type='upsampling',
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet101(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_unet(backbone, classes, decoder_filters,
                       resnet_skips, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet101'

    return model


def UResNet152(input_shape, classes, decoder_filters=16, decoder_block_type='upsampling',
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet152(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_unet(backbone, classes, decoder_filters,
                       resnet_skips, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet152'

    return model

