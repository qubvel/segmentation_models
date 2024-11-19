import copy
import efficientnet.model as eff
from classification_models.models_factory import ModelsFactory

from . import inception_resnet_v2 as irv2
from . import inception_v3 as iv3
from . import darknet53 as dkn53
from . import mobilenet_v3 as mbnv3

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
        'seresnet50': (246, 136, 62, 4),
        'seresnet101': (552, 136, 62, 4),
        'seresnet152': (858, 208, 62, 4),
        'seresnext50': (1078, 584, 254, 4),
        'seresnext101': (2472, 584, 254, 4),
        'senet154': (6884, 1625, 454, 12),

        # Mobile Nets
        'mobilenet': ('conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'),
        'mobilenetv2': ('block_13_expand_relu', 'block_6_expand_relu', 'block_3_expand_relu',
                        'block_1_expand_relu'),
        'mobilenetv3': ('Conv_1', 'activation_29', 'activation_15', 'activation_6'),
        #'mobilenetv3large': ('Conv_1', 'activation_29', 'activation_15', 'activation_6'),
        'mobilenetv3small': ('activation_31', 'activation_22', 'activation_7', 'activation_3'),

        # EfficientNets
        'efficientnetb0': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb1': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb2': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb3': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb4': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb5': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb6': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb7': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),

        # DarkNets
        'darknet53': ('activation_58', 'activation_37', 'activation_16', 'activation_7'),  # 204 equals conv2d_58 (14, 14, 512), 131 equals conv2d_37 (28, 28, 256)


        #'darknet53': (204, 131, 'activation_16', 'activation_7'),  # 204 equals conv2d_58 (14, 14, 512), 131 equals conv2d_37 (28, 28, 256)

    }


    _models_update = {
        'inceptionresnetv2': [irv2.InceptionResNetV2, irv2.preprocess_input],
        'inceptionv3': [iv3.InceptionV3, iv3.preprocess_input],

        'efficientnetb0': [eff.EfficientNetB0, eff.preprocess_input],
        'efficientnetb1': [eff.EfficientNetB1, eff.preprocess_input],
        'efficientnetb2': [eff.EfficientNetB2, eff.preprocess_input],
        'efficientnetb3': [eff.EfficientNetB3, eff.preprocess_input],
        'efficientnetb4': [eff.EfficientNetB4, eff.preprocess_input],
        'efficientnetb5': [eff.EfficientNetB5, eff.preprocess_input],
        'efficientnetb6': [eff.EfficientNetB6, eff.preprocess_input],
        'efficientnetb7': [eff.EfficientNetB7, eff.preprocess_input],

        'darknet53': [dkn53.csp_darknet53, dkn53.preprocess_input],

        'mobilenetv3': [mbnv3.MobileNetV3Large, mbnv3.preprocess_input],
        'mobilenetv3small': [mbnv3.MobileNetV3Small, mbnv3.preprocess_input],
    }
    # currently not supported
    _models_delete = ['resnet50v2', 'resnet101v2', 'resnet152v2',
                      'nasnetlarge', 'nasnetmobile', 'xception']

    @property
    def models(self):
        all_models = copy.copy(self._models)
        all_models.update(self._models_update)
        for k in self._models_delete:
            del all_models[k]
        return all_models

    def get_backbone(self, name, *args, **kwargs):
        model_fn, _ = self.get(name)
        model = model_fn(*args, **kwargs)
        return model

    def get_feature_layers(self, name, n=5):
        return self._default_feature_layers[name][:n]

    def get_preprocessing(self, name):
        return self.get(name)[1]


Backbones = BackbonesFactory()
