
from .classification_models.classification_models import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .classification_models.classification_models import ResNeXt50, ResNeXt101

from .inception_resnet_v2 import InceptionResNetV2
from .inception_v3 import InceptionV3

from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import VGG16
from keras.applications import VGG19


backbones = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "resnext50": ResNeXt50,
    "resnext101": ResNeXt101,
    "inceptionresnetv2": InceptionResNetV2,
    "inceptionv3": InceptionV3,
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,

}

def get_backbone(name, *args, **kwargs):
    return backbones[name](*args, **kwargs)