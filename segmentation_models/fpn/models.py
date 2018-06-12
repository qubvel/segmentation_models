from functools import wraps
from .builder import build_fpn
from ..backbones import ResNet18
from ..backbones import ResNet34
from ..backbones import ResNet50
from ..backbones import ResNet101
from ..backbones import ResNet152
from ..backbones import InceptionV3
from ..backbones import InceptionResNetV2
from ..backbones import ResNeXt50
from ..backbones import ResNeXt101
from keras.applications import DenseNet121
from keras.applications import DenseNet169
from keras.applications import DenseNet201

from ..utils import reverse


default_params = {
    'input_tensor': None,
    'encoder_weights': None,
    'pyramid_filters': 256,
    'segmentation_filters': 128,
    'use_batchnorm': False,
    'dropout': None,
    'activation': 'sigmoid',
    'last_upsampling_type': 'nn',
    'upsample_rates': (2,2,2),
    'last_upsample': 4,
}


def _abstract_model(
        input_shape, classes,
        input_tensor=default_params['input_tensor'],
        encoder_weights=default_params['encoder_weights'],
        pyramid_filters=default_params['pyramid_filters'],
        segmentation_filters=default_params['segmentation_filters'],
        use_batchnorm=default_params['use_batchnorm'],
        dropout=default_params['dropout'],
        activation=default_params['activation'],
        last_upsample=default_params['last_upsample'],
        last_upsampling_type=default_params['last_upsampling_type']):
    """
    FPN model alias

    Args:
        input_shape: input spatial dimention (H, W, C), only `channels_last` mode is supported
        classes: integer, number of target classes
        encoder_weights: str, pre-trained weights for backbone model, e.g. 'imagenet'
        input_tensor: keras input tensor
        classes: int, number of output feature maps
        activation: activation in last layer, e.g. 'sigmoid' or 'softmax'
        upsample_rates: tuple of integers, scaling rates between pyramid blocks
        pyramid_filters: int, number of filters in `M` blocks of top-down FPN branch
        segmentation_filters: int, number of filters in `P` blocks of FPN
        last_upsample: rate for upsumpling concatenated pyramid predictions to
            match spatial resolution of input data
        last_upsampling_type: 'nn' or 'bilinear'
        dropout: float [0, 1), dropout rate
        use_batchnorm: bool, include batch normalization to FPN between `conv`
            and `relu` layers

    Returns:
        keras `Model`, Feature Pyramid Model for semantic segmentation
    """
    raise NotImplementedError


def FPNModel(backbone_fn, layers, name='fpn-model', 
             input_shape=(None, None, 3), classes=21,
             input_tensor=default_params['input_tensor'],
             encoder_weights=default_params['encoder_weights'],
             pyramid_filters=default_params['pyramid_filters'],
             segmentation_filters=default_params['segmentation_filters'],
             use_batchnorm=default_params['use_batchnorm'],
             dropout=default_params['dropout'],
             activation=default_params['activation'],
             last_upsampling_type=default_params['last_upsampling_type'],
             last_upsample=default_params['last_upsample'],
             upsample_rates=default_params['upsample_rates']):

    backbone = backbone_fn(input_shape, input_tensor=input_tensor,
                        weights=encoder_weights, include_top=False)

    model = build_fpn(backbone, layers, classes=classes,
                      pyramid_filters=pyramid_filters,
                      segmentation_filters=segmentation_filters,
                      activation=activation, upsample_rates=upsample_rates,
                      use_batchnorm=use_batchnorm, dropout=dropout,
                      last_upsample=last_upsample,
                      last_upsampling_type=last_upsampling_type)

    model.name = name
    return model


@wraps(_abstract_model)
def FPNResNet18(input_shape, classes, **kwargs):
    layers = reverse(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1'])
    return FPNModel(ResNet18, layers, input_shape=input_shape,
                    name='fpn-resnet18', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNResNet34(input_shape, classes, **kwargs):
    layers = reverse(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1'])
    return FPNModel(ResNet34, layers, input_shape=input_shape,
                    name='fpn-resnet34', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNResNet50(input_shape, classes, **kwargs):
    layers = reverse(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1'])
    return FPNModel(ResNet50, layers, input_shape=input_shape,
                    name='fpn-resnet50', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNResNet101(input_shape, classes, **kwargs):
    layers = reverse(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1'])
    return FPNModel(ResNet101, layers, input_shape=input_shape,
                    name='fpn-resnet101', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNResNet152(input_shape, classes, **kwargs):
    layers = reverse(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1'])
    return FPNModel(ResNet152, layers, input_shape=input_shape,
                    name='fpn-resnet152', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNInceptionV3(input_shape, classes, **kwargs):
    layers = reverse([16, 86, 228])
    return FPNModel(InceptionV3, layers, input_shape=input_shape,
                    name='fpn-inception_v3', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNInceptionResNetV2(input_shape, classes, **kwargs):
    layers = reverse([16, 260, 594])
    return FPNModel(InceptionResNetV2, layers, input_shape=input_shape,
                    name='fpn-inception_resnet_v2', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNDenseNet121(input_shape, classes, **kwargs):
    layers = reverse([51, 139, 311])
    return FPNModel(DenseNet121, layers, input_shape=input_shape,
                    name='fpn-densenet121', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNDenseNet169(input_shape, classes, **kwargs):
    layers = reverse([51, 139, 367])
    return FPNModel(DenseNet169, layers, input_shape=input_shape,
                    name='fpn-densenet121', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNDenseNet201(input_shape, classes, **kwargs):
    layers = reverse([51, 139, 479])
    return FPNModel(DenseNet201, layers, input_shape=input_shape,
                    name='fpn-densenet121', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNResNeXt50(input_shape, classes, **kwargs):
    layers = reverse(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1'])
    return FPNModel(ResNeXt50, layers, input_shape=input_shape,
                    name='fpn-resnext50', classes=classes, **kwargs)


@wraps(_abstract_model)
def FPNResNeXt101(input_shape, classes, **kwargs):
    layers = reverse(['stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1'])
    return FPNModel(ResNeXt101, layers, input_shape=input_shape,
                    name='fpn-resnext50', classes=classes, **kwargs)