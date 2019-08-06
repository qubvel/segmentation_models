import pytest
import numpy as np

import segmentation_models as sm
from segmentation_models.metrics import IOUScore, FScore
from segmentation_models.losses import JaccardLoss, DiceLoss

if sm.framework() == sm._TF_KERAS_FRAMEWORK_NAME:
    from tensorflow import keras
elif sm.framework() == sm._KERAS_FRAMEWORK_NAME:
    import keras
else:
    raise ValueError('Incorrect framework {}'.format(sm.framework()))

METRICS = [
    IOUScore,
    FScore,
]

LOSSES = [
    JaccardLoss,
    DiceLoss,
]

GT0 = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    dtype='float32',
)

GT1 = np.array(
    [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ],
    dtype='float32',
)

PR1 = np.array(
    [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ],
    dtype='float32',
)

PR2 = np.array(
    [
        [0, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
    ],
    dtype='float32',
)

PR3 = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
    ],
    dtype='float32',
)

IOU_CASES = (

    (GT0, GT0, 1.00),
    (GT1, GT1, 1.00),

    (GT0, PR1, 0.00),
    (GT0, PR2, 0.00),
    (GT0, PR3, 0.00),

    (GT1, PR1, 0.50),
    (GT1, PR2, 1. / 3.),
    (GT1, PR3, 0.00),
)

F1_CASES = (

    (GT0, GT0, 1.00),
    (GT1, GT1, 1.00),

    (GT0, PR1, 0.00),
    (GT0, PR2, 0.00),
    (GT0, PR3, 0.00),

    (GT1, PR1, 2. / 3.),
    (GT1, PR2, 0.50),
    (GT1, PR3, 0.00),
)

F2_CASES = (

    (GT0, GT0, 1.00),
    (GT1, GT1, 1.00),

    (GT0, PR1, 0.00),
    (GT0, PR2, 0.00),
    (GT0, PR3, 0.00),

    (GT1, PR1, 5. / 9.),
    (GT1, PR2, 0.50),
    (GT1, PR3, 0.00),
)


def _to_4d(x):
    if x.ndim == 2:
        return x[None, :, :, None]
    elif x.ndim == 3:
        return x[None, :, :]


def _add_4d(x):
    if x.ndim == 3:
        return x[..., None]


@pytest.mark.parametrize('case', IOU_CASES)
def test_iou_metric(case):
    gt, pr, res = case
    gt = _to_4d(gt)
    pr = _to_4d(pr)
    iou_score = IOUScore(smooth=10e-12)
    score = keras.backend.eval(iou_score(gt, pr))
    assert np.allclose(score, res)


@pytest.mark.parametrize('case', IOU_CASES)
def test_jaccrad_loss(case):
    gt, pr, res = case
    gt = _to_4d(gt)
    pr = _to_4d(pr)
    jaccard_loss = JaccardLoss(smooth=10e-12)
    score = keras.backend.eval(jaccard_loss(gt, pr))
    assert np.allclose(score, 1 - res)


def _test_f_metric(case, beta=1):
    gt, pr, res = case
    gt = _to_4d(gt)
    pr = _to_4d(pr)
    f_score = FScore(beta=beta, smooth=10e-12)
    score = keras.backend.eval(f_score(gt, pr))
    assert np.allclose(score, res)


@pytest.mark.parametrize('case', F1_CASES)
def test_f1_metric(case):
    _test_f_metric(case, beta=1)


@pytest.mark.parametrize('case', F2_CASES)
def test_f2_metric(case):
    _test_f_metric(case, beta=2)


@pytest.mark.parametrize('case', F1_CASES)
def test_dice_loss(case):
    gt, pr, res = case
    gt = _to_4d(gt)
    pr = _to_4d(pr)
    dice_loss = DiceLoss(smooth=10e-12)
    score = keras.backend.eval(dice_loss(gt, pr))
    assert np.allclose(score, 1 - res)


@pytest.mark.parametrize('func', METRICS + LOSSES)
def test_per_image(func):
    gt = np.stack([GT0, GT1], axis=0)
    pr = np.stack([PR1, PR2], axis=0)

    gt = _add_4d(gt)
    pr = _add_4d(pr)

    # calculate score per image
    score_1 = keras.backend.eval(func(per_image=True, smooth=10e-12)(gt, pr))
    score_2 = np.mean([
        keras.backend.eval(func(smooth=10e-12)(_to_4d(GT0), _to_4d(PR1))),
        keras.backend.eval(func(smooth=10e-12)(_to_4d(GT1), _to_4d(PR2))),
    ])
    assert np.allclose(score_1, score_2)


@pytest.mark.parametrize('func', METRICS + LOSSES)
def test_per_batch(func):
    gt = np.stack([GT0, GT1], axis=0)
    pr = np.stack([PR1, PR2], axis=0)

    gt = _add_4d(gt)
    pr = _add_4d(pr)

    # calculate score per batch
    score_1 = keras.backend.eval(func(per_image=False, smooth=10e-12)(gt, pr))

    gt1 = np.concatenate([GT0, GT1], axis=0)
    pr1 = np.concatenate([PR1, PR2], axis=0)
    score_2 = keras.backend.eval(func(per_image=True, smooth=10e-12)(_to_4d(gt1), _to_4d(pr1)))

    assert np.allclose(score_1, score_2)


@pytest.mark.parametrize('case', IOU_CASES)
def test_threshold_iou(case):
    gt, pr, res = case
    gt = _to_4d(gt)
    pr = _to_4d(pr) * 0.51
    iou_score = IOUScore(smooth=10e-12, threshold=0.5)
    score = keras.backend.eval(iou_score(gt, pr))
    assert np.allclose(score, res)


if __name__ == '__main__':
    pytest.main([__file__])
