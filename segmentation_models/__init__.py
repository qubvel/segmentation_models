import os
import functools
from .__version__ import __version__
from . import base

_KERAS_FRAMEWORK_NAME = 'keras'
_TF_KERAS_FRAMEWORK_NAME = 'tf.keras'

_DEFAULT_KERAS_FRAMEWORK = _KERAS_FRAMEWORK_NAME
_KERAS_FRAMEWORK = None
_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None
_KERAS_LOSSES = None


def inject_global_losses(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['losses'] = _KERAS_LOSSES
        return func(*args, **kwargs)

    return wrapper


def inject_global_submodules(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = _KERAS_BACKEND
        kwargs['layers'] = _KERAS_LAYERS
        kwargs['models'] = _KERAS_MODELS
        kwargs['utils'] = _KERAS_UTILS
        return func(*args, **kwargs)

    return wrapper


def filter_kwargs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_kwargs = {k: v for k, v in kwargs.items() if k in ['backend', 'layers', 'models', 'utils']}
        return func(*args, **new_kwargs)

    return wrapper


def framework():
    """Return name of Segmentation Models framework"""
    return _KERAS_FRAMEWORK


def set_framework(name):
    """Set framework for Segmentation Models

    Args:
        name (str): one of ``keras``, ``tf.keras``, case insensitive.

    Raises:
        ValueError: in case of incorrect framework name.
        ImportError: in case framework is not installed.

    """
    name = name.lower()

    if name == _KERAS_FRAMEWORK_NAME:
        import keras
        import efficientnet.keras  # init custom objects
    elif name == _TF_KERAS_FRAMEWORK_NAME:
        from tensorflow import keras
        import efficientnet.tfkeras  # init custom objects
    else:
        raise ValueError('Not correct module name `{}`, use `{}` or `{}`'.format(
            name, _KERAS_FRAMEWORK_NAME, _TF_KERAS_FRAMEWORK_NAME))

    global _KERAS_BACKEND, _KERAS_LAYERS, _KERAS_MODELS
    global _KERAS_UTILS, _KERAS_LOSSES, _KERAS_FRAMEWORK

    _KERAS_FRAMEWORK = name
    _KERAS_BACKEND = keras.backend
    _KERAS_LAYERS = keras.layers
    _KERAS_MODELS = keras.models
    _KERAS_UTILS = keras.utils
    _KERAS_LOSSES = keras.losses

    # allow losses/metrics get keras submodules
    base.KerasObject.set_submodules(
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils,
    )


# set default framework
_framework = os.environ.get('SM_FRAMEWORK', _DEFAULT_KERAS_FRAMEWORK)
try:
    set_framework(_framework)
except ImportError:
    other = _TF_KERAS_FRAMEWORK_NAME if _framework == _KERAS_FRAMEWORK_NAME else _KERAS_FRAMEWORK_NAME
    set_framework(other)

print('Segmentation Models: using `{}` framework.'.format(_KERAS_FRAMEWORK))

# import helper modules
from . import losses
from . import metrics
from . import utils

# wrap segmentation models with framework modules
from .backbones.backbones_factory import Backbones
from .models.unet import Unet as _Unet
from .models.pspnet import PSPNet as _PSPNet
from .models.linknet import Linknet as _Linknet
from .models.fpn import FPN as _FPN

Unet = inject_global_submodules(_Unet)
PSPNet = inject_global_submodules(_PSPNet)
Linknet = inject_global_submodules(_Linknet)
FPN = inject_global_submodules(_FPN)
get_available_backbone_names = Backbones.models_names


def get_preprocessing(name):
    preprocess_input = Backbones.get_preprocessing(name)
    # add bakcend, models, layers, utils submodules in kwargs
    preprocess_input = inject_global_submodules(preprocess_input)
    # delete other kwargs
    # keras-applications preprocessing raise an error if something
    # except `backend`, `layers`, `models`, `utils` passed in kwargs
    preprocess_input = filter_kwargs(preprocess_input)
    return preprocess_input


__all__ = [
    'Unet', 'PSPNet', 'FPN', 'Linknet',
    'set_framework', 'framework',
    'get_preprocessing', 'get_available_backbone_names',
    'losses', 'metrics', 'utils',
    '__version__',
]
