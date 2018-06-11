from .builder import build_fpn
from ..backbones import ResNet18
from ..backbones import ResNet34
from ..backbones import ResNet50
from ..backbones import ResNet101
from ..backbones import ResNet152


def FPNResNet18(input_shape, classes=21, pyramid_filters=256, segmentation_filters=128,
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet18(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    layers = list(reversed(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1']))
    model = build_fpn(backbone, layers, classes=classes,
                      pyramid_filters=pyramid_filters,
                      segmentation_filters=segmentation_filters,
                      activation=activation, upsample_rates=(2,2,2),
                      **kwargs)
    model.name = 'fpn-resnet18'

    return model


def FPNResNet34(input_shape, classes=21, pyramid_filters=256, segmentation_filters=128,
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet34(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    layers = list(reversed(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1']))
    model = build_fpn(backbone, layers, classes=classes,
                      pyramid_filters=pyramid_filters,
                      segmentation_filters=segmentation_filters,
                      activation=activation, upsample_rates=(2,2,2),
                      **kwargs)
    model.name = 'fpn-resnet34'

    return model


def FPNResNet50(input_shape, classes=21, pyramid_filters=256, segmentation_filters=128,
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet50(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    layers = list(reversed(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1']))
    model = build_fpn(backbone, layers, classes=classes,
                      pyramid_filters=pyramid_filters,
                      segmentation_filters=segmentation_filters,
                      activation=activation, upsample_rates=(2,2,2),
                      **kwargs)
    model.name = 'fpn-resnet50'

    return model


def FPNResNet101(input_shape, classes=21, pyramid_filters=256, segmentation_filters=128,
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet101(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    layers = list(reversed(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1']))
    model = build_fpn(backbone, layers, classes=classes,
                      pyramid_filters=pyramid_filters,
                      segmentation_filters=segmentation_filters,
                      activation=activation, upsample_rates=(2,2,2),
                      **kwargs)
    model.name = 'fpn-resnet101'

    return model


def FPNResNet152(input_shape, classes=21, pyramid_filters=256, segmentation_filters=128,
              encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = ResNet152(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    layers = list(reversed(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1']))
    model = build_fpn(backbone, layers, classes=classes,
                      pyramid_filters=pyramid_filters,
                      segmentation_filters=segmentation_filters,
                      activation=activation, upsample_rates=(2,2,2),
                      **kwargs)
    model.name = 'fpn-resnet152'

    return model



