"""
Image pre-processing functions.
Images are assumed to be read in uint8 format (range 0-255).
"""

import keras.applications as ka

identical = lambda x: x
bgr_transpose = lambda x: x[..., ::-1]

models_preprocessing = {
    'vgg16': ka.vgg16.preprocess_input,
    'vgg19': ka.vgg19.preprocess_input,
    'resnet18': bgr_transpose,
    'resnet34': bgr_transpose,
    'resnet50': bgr_transpose,
    'resnet101': bgr_transpose,
    'resnet152': bgr_transpose,
    'resnext50': identical,
    'resnext101': identical,
    'densenet121': ka.densenet.preprocess_input,
    'densenet169': ka.densenet.preprocess_input,
    'densenet201': ka.densenet.preprocess_input,
    'inceptionv3': ka.inception_v3.preprocess_input,
    'inceptionresnetv2': ka.inception_resnet_v2.preprocess_input,
}


def get_preprocessing(backbone):
    """Returns pre-processing function for image data according to name of backbone

    Args:
        backbone (str): name of classification model

    Returns:
        ``callable``: preprocessing_function
    """
    return models_preprocessing[backbone]
