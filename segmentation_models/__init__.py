name = "segmentation_models"

import os
import functools
from .__version__ import __version__

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


def inject_global_submodules(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = _KERAS_BACKEND
        kwargs['layers'] = _KERAS_LAYERS
        kwargs['models'] = _KERAS_MODELS
        kwargs['utils'] = _KERAS_UTILS
        return func(*args, **kwargs)

    return wrapper


def set_framework(name='keras'):
    if name == 'keras':
        import keras
        import efficientnet.keras  # init custom objects
    elif name == 'tensorflow' or name == 'tf.keras' or name == 'tensorflow.keras':
        from tensorflow import keras
        import efficientnet.tfkeras  # init custom objects
    else:
        raise ValueError('Not correct module name')

    global _KERAS_BACKEND, _KERAS_LAYERS, _KERAS_MODELS, _KERAS_UTILS

    _KERAS_BACKEND = keras.backend
    _KERAS_LAYERS = keras.layers
    _KERAS_MODELS = keras.models
    _KERAS_UTILS = keras.utils


if os.environ.get('TF_KERAS', False):
    set_framework(name='tf.keras')
else:
    try:
        set_framework(name='keras')
    except ImportError:
        set_framework(name='keras')

from .models.unet import Unet as _Unet
from .models.pspnet import PSPNet as _PSPNet
from .models.linknet import Linknet as _Linknet
from .models.fpn import FPN as _FPN

Unet = inject_global_submodules(_Unet)
PSPNet = inject_global_submodules(_PSPNet)
Linknet = inject_global_submodules(_Linknet)
FPN = inject_global_submodules(_FPN)
