import os
import pytest
import random
import six
import numpy as np

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models import Linknet
from segmentation_models import PSPNet
from segmentation_models import FPN
from segmentation_models import get_available_backbone_names

if sm.framework() == sm._TF_KERAS_FRAMEWORK_NAME:
    from tensorflow import keras
elif sm.framework() == sm._KERAS_FRAMEWORK_NAME:
    import keras
else:
    raise ValueError('Incorrect framework {}'.format(sm.framework()))

def get_backbones():
    is_travis = os.environ.get('TRAVIS', False)
    exclude = ['senet154', 'efficientnetb6', 'efficientnetb7']
    backbones = get_available_backbone_names()

    if is_travis:
        backbones = [b for b in backbones if b not in exclude]
    return backbones


BACKBONES = get_backbones()


def _select_names(names):
    is_full = os.environ.get('FULL_TEST', False)
    if not is_full:
        return [random.choice(names)]
    else:
        return names


def keras_test(func):
    """Function wrapper to clean up after TensorFlow tests.
    # Arguments
        func: test function to clean up after.
    # Returns
        A function wrapping the input function.
    """
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        keras.backend.clear_session()
        return output
    return wrapper


@keras_test
def _test_none_shape(model_fn, backbone, *args, **kwargs):

    # define number of channels
    input_shape = kwargs.get('input_shape', None)
    n_channels = 3 if input_shape is None else input_shape[-1]

    # create test sample
    x = np.ones((1, 32, 32, n_channels))

    # define model and process sample
    model = model_fn(backbone, *args, **kwargs)
    y = model.predict(x)

    # check output dimensions
    assert x.shape[:-1] == y.shape[:-1]


@keras_test
def _test_shape(model_fn, backbone, input_shape, *args, **kwargs):

    # create test sample
    x = np.ones((1, *input_shape))

    # define model and process sample
    model = model_fn(backbone, input_shape=input_shape, *args, **kwargs)
    y = model.predict(x)

    # check output dimensions
    assert x.shape[:-1] == y.shape[:-1]


@pytest.mark.parametrize('backbone', _select_names(BACKBONES))
def test_unet(backbone):
    _test_none_shape(
        Unet, backbone, encoder_weights=None)

    _test_none_shape(
        Unet, backbone, encoder_weights='imagenet')

    _test_shape(
        Unet, backbone, input_shape=(256, 256, 4), encoder_weights=None)


@pytest.mark.parametrize('backbone', _select_names(BACKBONES))
def test_linknet(backbone):
    _test_none_shape(
        Linknet, backbone, encoder_weights=None)

    _test_none_shape(
        Linknet, backbone, encoder_weights='imagenet')

    _test_shape(
        Linknet, backbone, input_shape=(256, 256, 4), encoder_weights=None)


@pytest.mark.parametrize('backbone', _select_names(BACKBONES))
def test_pspnet(backbone):

    _test_shape(
        PSPNet, backbone, input_shape=(384, 384, 4), encoder_weights=None)

    _test_shape(
        PSPNet, backbone, input_shape=(384, 384, 3), encoder_weights='imagenet')


@pytest.mark.parametrize('backbone', _select_names(BACKBONES))
def test_fpn(backbone):
    _test_none_shape(
        FPN, backbone, encoder_weights=None)

    _test_none_shape(
        FPN, backbone, encoder_weights='imagenet')

    _test_shape(
        FPN, backbone, input_shape=(256, 256, 4), encoder_weights=None)


if __name__ == '__main__':
    pytest.main([__file__])
