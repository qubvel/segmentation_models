import pytest
import numpy as np
import keras.backend as K

from segmentation_models.metrics import iou_score

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


def _to_4d(x):
    return x[None, :, :, None]


@pytest.mark.parametrize('case', IOU_CASES)
def test_iou_metric(case):
    gt, pr, res = case
    gt = _to_4d(gt)
    pr = _to_4d(pr)
    score = K.eval(iou_score(gt, pr))
    assert np.allclose(score, res)
