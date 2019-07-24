import functools
from .backbones_factory import BackbonesFactory


def inject_keras_modules(func):
    import keras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = keras.backend
        kwargs['layers'] = keras.layers
        kwargs['models'] = keras.models
        kwargs['utils'] = keras.utils
        return func(*args, **kwargs)

    return wrapper


get_backbone = inject_keras_modules(BackbonesFactory().get_backbone)
get_preprocessing = inject_keras_modules(BackbonesFactory().get_preprocessing)

get_feature_layers = BackbonesFactory().get_feature_layers
get_names = BackbonesFactory().models_names

