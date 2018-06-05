from .builder import build_unet
from ..backbones import ResNet18


def UResNet18(input_shape, classes, decoder_filters=16, decoder_block_type='upsampling',
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet18(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    skip_layers = [1, 5, 28, 47, 66][::-1]
    model = build_unet(backbone, classes, decoder_filters,
                       skip_layers, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet18'

    return model
