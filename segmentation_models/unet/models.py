from .builder import build_unet
from ..backbones import ResNet18
from ..backbones import ResNet34
from ..backbones import ResNet50
from ..backbones import ResNet101
from ..backbones import ResNet152
from ..backbones import InceptionV3
from ..backbones import InceptionResNetV2
from keras.applications import DenseNet121
from keras.applications import DenseNet169
from keras.applications import DenseNet201


resnet_skips = list(reversed(['bn_data', 'relu0', 'stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1']))


def UResNet18(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet18(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_unet(backbone, classes, decoder_filters,
                       resnet_skips, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet18'

    return model


def UResNet34(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet34(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_unet(backbone, classes, decoder_filters,
                       resnet_skips, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet34'

    return model


def UResNet50(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet50(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_unet(backbone, classes, decoder_filters,
                       resnet_skips, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet50'

    return model


def UResNet101(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
               encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet101(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_unet(backbone, classes, decoder_filters,
                       resnet_skips, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet101'

    return model


def UResNet152(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
               encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet152(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_unet(backbone, classes, decoder_filters,
                       resnet_skips, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet152'

    return model


def UInceptionV3(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
                 encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = InceptionV3(input_shape=input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    skip_connections = list(reversed([9, 16, 86, 228]))
    model = build_unet(backbone, classes, decoder_filters,
                       skip_connections, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-inception_v3'

    return model


def UInceptionResNetV2(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
                       encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = InceptionResNetV2(input_shape=input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    skip_connections = list(reversed([9, 16, 260, 594]))
    model = build_unet(backbone, classes, decoder_filters,
                       skip_connections, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-inception_resnet_v2'

    return model


def UDenseNet121(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
                 encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = DenseNet121(input_shape=input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    skip_connections = list(reversed([4, 51, 139, 311]))
    model = build_unet(backbone, classes, decoder_filters,
                       skip_connections, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-densenet121'

    return model


def UDenseNet169(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
                 encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = DenseNet169(input_shape=input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    skip_connections = list(reversed([4, 51, 139, 367]))
    model = build_unet(backbone, classes, decoder_filters,
                       skip_connections, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-densenet169'

    return model


def UDenseNet201(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
                 encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = DenseNet201(input_shape=input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    skip_connections = list(reversed([4, 51, 139, 479]))
    model = build_unet(backbone, classes, decoder_filters,
                       skip_connections, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-densenet201'

    return model