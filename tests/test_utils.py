import pytest
import numpy as np
# import keras.backend.tensorflow_backend as KTF
# import keras.backend as K
# import tensorflow as tf
from keras import regularizers

from segmentation_models.utils import set_regularization
from segmentation_models import Unet

X1 = np.ones((1, 32, 32, 3))
Y1 = np.ones((1, 32, 32, 1))
MODEL = Unet('resnet18')
CASE = (

    (X1, Y1, MODEL),
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


@pytest.mark.parametrize('case', CASE)
def test_kernel_reg(case):
    x, y, model= case

    l1_reg = regularizers.l1(0.1)
    reg_model = set_regularization(model, kernel_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    l2_reg = regularizers.l2(0.1)
    reg_model = set_regularization(model, kernel_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)


@pytest.mark.parametrize('case', CASE)
def test_bias_reg(case):
    x, y, model= case

    l1_reg = regularizers.l1(1)
    reg_model = set_regularization(model, bias_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    l2_reg = regularizers.l2(1)
    reg_model = set_regularization(model, bias_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)


@pytest.mark.parametrize('case', CASE)
def test_bn_reg(case):
    x, y, model= case

    l1_reg = regularizers.l1(1)
    reg_model = set_regularization(model, gamma_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)
    reg_model = set_regularization(model, beta_regularizer=l1_reg)
    _test_regularizer(model, reg_model, x, y)

    l2_reg = regularizers.l2(1)
    reg_model = set_regularization(model, gamma_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)
    reg_model = set_regularization(model, beta_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)


@pytest.mark.parametrize('case', CASE)
def test_activity_reg(case):
    x, y, model= case

    l2_reg = regularizers.l2(1)
    reg_model = set_regularization(model, activity_regularizer=l2_reg)
    _test_regularizer(model, reg_model, x, y)


if __name__ == '__main__':
    pytest.main([__file__])
