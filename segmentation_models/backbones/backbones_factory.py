import copy
import efficientnet.model as eff
import efficientnet.preprocessing as effp
from classification_models.models_factory import ModelsFactory

from . import inception_resnet_v2 as irv2
from . import inception_v3 as iv3


class BackbonesFactory(ModelsFactory):
    _default_feature_layers = {

        # List of layers to take features from backbone in the following order:
        # (x16, x8, x4, x2, x1) - `x4` mean that features has 4 times less spatial
        # resolution (Height x Width) than input image.

        # VGG
        'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
        'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),

        # ResNets
        'resnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnet152': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),

        # ResNeXt
        'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'resnext101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),

        # Inception
        'inceptionv3': (228, 86, 16, 9),
        'inceptionresnetv2': (594, 260, 16, 9),

        # DenseNet
        'densenet121': (311, 139, 51, 4),
        'densenet169': (367, 139, 51, 4),
        'densenet201': (479, 139, 51, 4),

        # SE models
        'seresnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'seresnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
        'seresnet50': (233, 129, 59, 4),
        'seresnet101': (522, 129, 59, 4),
        'seresnet152': (811, 197, 59, 4),
        'seresnext50': (1065, 577, 251, 4),
        'seresnext101': (2442, 577, 251, 4),
        'senet154': (6837, 1614, 451, 12),

        # Mobile Nets
        'mobilenet': ('conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'),
        'mobilenetv2': ('block_13_expand_relu', 'block_6_expand_relu', 'block_3_expand_relu',
                        'block_1_expand_relu'),

        # EfficientNets
        'efficientnetb0': (169, 77, 47, 17),
        'efficientnetb1': (246, 122, 76, 30),
        'efficientnetb2': (246, 122, 76, 30),
        'efficientnetb3': (278, 122, 76, 30),

        # weights are not released
        'efficientnetb4': (342, 154, 92, 30),
        'efficientnetb5': (419, 199, 121, 43),
        #     'efficientnetb6': (483, 231, 137, 43),
        #     'efficientnetb7': (592, 276, 166, 56),

    }

    _models_update = {
        'inceptionresnetv2': [irv2.InceptionResNetV2, irv2.preprocess_input],
        'inceptionv3': [iv3.InceptionV3, iv3.preprocess_input],

        'efficientnetb0': [eff.EfficientNetB0, effp.preprocess_input],
        'efficientnetb1': [eff.EfficientNetB1, effp.preprocess_input],
        'efficientnetb2': [eff.EfficientNetB2, effp.preprocess_input],
        'efficientnetb3': [eff.EfficientNetB3, effp.preprocess_input],
        'efficientnetb4': [eff.EfficientNetB4, effp.preprocess_input],
        'efficientnetb5': [eff.EfficientNetB5, effp.preprocess_input],

        #     'efficientnetb6': [eff.EfficientNetB6, eff.preprocess_input],
        #     'efficientnetb7': [eff.EfficientNetB7, eff.preprocess_input],
    }

    @property
    def models(self):
        all_models = copy.copy(self._models)
        all_models.update(self._models_update)
        return all_models

    def get_backbone(self, name, *args, **kwargs):
        model_fn, _ = self.get(name)
        model = model_fn(*args, **kwargs)
        return model

    def get_feature_layers(self, name, n=5):
        return self._default_feature_layers[name][:n]

    def get_preprocessing(self, name):
        return self.get(name)[1]
