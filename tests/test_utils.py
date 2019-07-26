import pytest
import numpy as np

import segmentation_models as sm
from segmentation_models.utils import set_regularization
from segmentation_models import Unet

if sm.framework() == sm._TF_KERAS_FRAMEWORK_NAME:
    from tensorflow import keras
elif sm.framework() == sm._KERAS_FRAMEWORK_NAME:
    import keras
else:
    raise ValueError('Incorrect framework {}'.format(sm.framework()))

X1 = np.ones((1, 32, 32, 3))
Y1 = np.ones((1, 32, 32, 1))
MODEL = Unet
BACKBONE = 'resnet18'
CASE = (

    (X1, Y1, MODEL, BACKBONE),
)


def _test_regularizer(model, reg_model, x, y):

    def zero_loss(gt, pr):
        return pr * 0

    model.compile('Adam', loss=zero_loss, metrics=['binary_accuracy'])
    reg_model.compile('Adam', loss=zero_loss, metrics=['binary_accuracy'])

    loss_1, _ = model.test_on_batch(x, y)
    loss_2, _ = reg_model.test_on_batch(x, y)

    assert loss_1 == 0
    assert loss_2 > 0

    keras.backend.clear_session()


@pytest.mark.parametrize('case', CASE)
def test_kernel_reg(case):
    x, y, model_fn, backbone= case

    l1_reg = keras.regularizers.l1(0.1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, kernel_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    l2_reg = keras.regularizers.l2(0.1)
    model = model_fn(backbone, encoder_weights=None)
    reg_model = set_regularization(model, kernel_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)


"""
Note:
    backbone resnet18 use BN after each conv layer --- so no bias used in these conv layers
    skip the bias regularizer test

@pytest.mark.parametrize('case', CASE)
def test_bias_reg(case):
    x, y, model_fn, backbone = case

    l1_reg = regularizers.l1(1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, bias_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    l2_reg = regularizers.l2(1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, bias_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)
"""


@pytest.mark.parametrize('case', CASE)
def test_bn_reg(case):
    x, y, model_fn, backbone= case

    l1_reg = keras.regularizers.l1(1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, gamma_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    model = model_fn(backbone)
    reg_model = set_regularization(model, beta_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    l2_reg = keras.regularizers.l2(1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, gamma_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)

    model = model_fn(backbone)
    reg_model = set_regularization(model, beta_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)


@pytest.mark.parametrize('case', CASE)
def test_activity_reg(case):
    x, y, model_fn, backbone= case

    l2_reg = keras.regularizers.l2(1)
    model = model_fn(backbone)
    reg_model = set_regularization(model, activity_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)


if __name__ == '__main__':
    pytest.main([__file__])
