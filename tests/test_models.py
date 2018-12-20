import sys
import pytest
import random as rn
import six
import numpy as np
import keras.backend as K

sys.path.insert(0, '..')

from segmentation_models import Unet
from segmentation_models import Linknet
from segmentation_models import PSPNet
from segmentation_models import FPN
from segmentation_models.backbones import backbones as bkb


BACKBONES = list(bkb.backbones.keys())


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
        K.clear_session()
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


def test_unet():
    _test_none_shape(
        Unet, rn.choice(BACKBONES), encoder_weights=None)

    _test_none_shape(
        Unet, rn.choice(BACKBONES), encoder_weights='imagenet')

    _test_shape(
        Unet, rn.choice(BACKBONES), input_shape=(256, 256, 4), encoder_weights=None)


def test_linknet():
    _test_none_shape(
        Linknet, rn.choice(BACKBONES), encoder_weights=None)

    _test_none_shape(
        Linknet, rn.choice(BACKBONES), encoder_weights='imagenet')

    _test_shape(
        Linknet, rn.choice(BACKBONES), input_shape=(256, 256, 4), encoder_weights=None)


def test_pspnet():

    _test_shape(
        PSPNet, rn.choice(BACKBONES), input_shape=(384, 384, 4), encoder_weights=None)

    _test_shape(
        PSPNet, rn.choice(BACKBONES), input_shape=(384, 384, 3), encoder_weights='imagenet')


def test_fpn():
    _test_none_shape(
        FPN, rn.choice(BACKBONES), encoder_weights=None)

    _test_none_shape(
        FPN, rn.choice(BACKBONES), encoder_weights='imagenet')

    _test_shape(
        FPN, rn.choice(BACKBONES), input_shape=(256, 256, 4), encoder_weights=None)

if __name__ == '__main__':
    pytest.main([__file__])
