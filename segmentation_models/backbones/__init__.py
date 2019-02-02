from classification_models import Classifiers

from . import inception_resnet_v2 as irv2
from . import inception_v3 as iv3

Classifiers._models.update({
    'inceptionresnetv2': [irv2.InceptionResNetV2, irv2.preprocess_input],
    'inceptionv3': [iv3.InceptionV3, iv3.preprocess_input],
})


def get_backbone(name, *args, **kwargs):
    return Classifiers.get_classifier(name)(*args, **kwargs)


def get_preprocessing(name):
    return Classifiers.get_preprocessing(name)
