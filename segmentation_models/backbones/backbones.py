import keras.applications as ka
import classification_models as cm

from .inception_resnet_v2 import InceptionResNetV2
from .inception_v3 import InceptionV3


backbones = {
    "vgg16": ka.VGG16,
    "vgg19": ka.VGG19,
    "resnet18": cm.ResNet18,
    "resnet34": cm.ResNet34,
    "resnet50": cm.ResNet50,
    "resnet101": cm.ResNet101,
    "resnet152": cm.ResNet152,
    "resnext50": cm.ResNeXt50,
    "resnext101": cm.ResNeXt101,
    "inceptionresnetv2": InceptionResNetV2,
    "inceptionv3": InceptionV3,
    "densenet121": ka.DenseNet121,
    "densenet169": ka.DenseNet169,
    "densenet201": ka.DenseNet201,

}

def get_backbone(name, *args, **kwargs):
    return backbones[name](*args, **kwargs)