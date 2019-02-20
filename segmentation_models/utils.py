""" Utility functions for segmentation models """
import warnings
import numpy as np
from functools import wraps
from keras.layers import BatchNormalization
from keras.models import model_from_json


def legacy_support(kwargs_map):
    """
    Decorator which map old kwargs to new ones

    Args:
        kwargs_map: dict 'old_argument: 'new_argument' (None if removed)

    """
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            # rename arguments
            for old_arg, new_arg in kwargs_map.items():
                if old_arg in kwargs.keys():
                    if new_arg is None:
                        raise TypeError("got an unexpected keyword argument '{}'".format(old_arg))
                    warnings.warn('`{old_arg}` is deprecated and will be removed '
                                  'in future releases, use `{new_arg}` instead.'.format(old_arg=old_arg, new_arg=new_arg))
                    kwargs[new_arg] = kwargs[old_arg]

            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_layer_number(model, layer_name):
    """
    Help find layer in Keras model by name
    Args:
        model: Keras `Model`
        layer_name: str, name of layer

    Returns:
        index of layer

    Raises:
        ValueError: if model does not contains layer with such name
    """
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError('No layer with name {} in  model {}.'.format(layer_name, model.name))


def extract_outputs(model, layers, include_top=False):
    """
    Help extract intermediate layer outputs from model
    Args:
        model: Keras `Model`
        layer: list of integers/str, list of layers indexes or names to extract output
        include_top: bool, include final model layer output

    Returns:
        list of tensors (outputs)
    """
    layers_indexes = ([get_layer_number(model, l) if isinstance(l, str) else l
                      for l in layers])
    outputs = [model.layers[i].output for i in layers_indexes]

    if include_top:
        outputs.insert(0, model.output)

    return outputs


def reverse(l):
    """Reverse list"""
    return list(reversed(l))


# decorator for models aliases, to add doc string
def add_docstring(doc_string=None):
    def decorator(fn):
        if fn.__doc__:
            fn.__doc__ += doc_string
        else:
            fn.__doc__ = doc_string

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def recompile(model):
    model.compile(model.optimizer, model.loss, model.metrics)


def freeze_model(model):
    """model all layers non trainable, excluding BatchNormalization layers"""
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    return


def set_trainable(model):
    """Set all layers of model trainable and recompile it

    Note:
        Model is recompiled using same optimizer, loss and metrics::

            model.compile(model.optimizer, model.loss, model.metrics)

    Args:
        model (``keras.models.Model``): instance of keras model

    """
    for layer in model.layers:
        layer.trainable = True
    recompile(model)


def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))


def set_regularization(model,
                       kernel_regularizer=None,
                       bias_regularizer=None,
                       activity_regularizer=None,
                       beta_regularizer=None,
                       gamma_regularizer=None
                       ):
    """Set regularizers to all layers

    Note:
       Returned model's config is updated correctly

    Args:
        model (``keras.models.Model``): instance of keras model
        kernel_regularizer(``regularizer`): regularizer of kernels
        bias_regularizer(``regularizer``): regularizer of bias
        activity_regularizer(``regularizer``): regularizer of activity
        gamma_regularizer(``regularizer``): regularizer of gamma of BatchNormalization
        beta_regularizer(``regularizer``): regularizer of beta of BatchNormalization

    Return:
        out (``Model``): config updated model
    """

    for layer in model.layers:
        # set kernel_regularizer
        if kernel_regularizer is not None and hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = kernel_regularizer
        # set bias_regularizer
        if bias_regularizer is not None and hasattr(layer, 'bias_regularizer'):
            layer.bias_regularizer = bias_regularizer
        # set activity_regularizer
        if activity_regularizer is not None and hasattr(layer, 'activity_regularizer'):
            layer.activity_regularizer = activity_regularizer

        # set beta and gamma of BN layer
        if beta_regularizer is not None and hasattr(layer, 'beta_regularizer'):
            layer.beta_regularizer = beta_regularizer

        if gamma_regularizer is not None and hasattr(layer, 'gamma_regularizer'):
            layer.gamma_regularizer = gamma_regularizer

    out = model_from_json(model.to_json())
    out.set_weights(model.get_weights())

    return out
