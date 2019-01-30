import keras.backend as K
from keras.utils.generic_utils import get_custom_objects

__all__ = [
    'iou_score', 'jaccard_score', 'f1_score', 'f2_score', 'dice_score',
    'get_f_score', 'get_dice_score', 'get_iou_score', 'get_jaccard_score',
]

SMOOTH = 1e-12


# ============================ Jaccard/IoU score ============================


def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    intersection = K.sum(gt * pr, axis=axes)
    union = K.sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = K.mean(iou, axis=0)

    # weighted mean per class
    iou = K.mean(iou * class_weights)

    return iou


def get_iou_score(class_weights=1., smooth=SMOOTH, per_image=True):
    def score(gt, pr):
        return iou_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image)

    return score


jaccard_score = iou_score
get_jaccard_score = get_iou_score

# Update custom objects
get_custom_objects().update({
    'iou_score': iou_score,
    'jaccard_score': jaccard_score,
})


# ============================== F/Dice - score ==============================

def f_score(gt, pr, class_weights=1, beta=1, smooth=SMOOTH, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # mean per image
    if per_image:
        score = K.mean(score, axis=0)

    # weighted mean per class
    score = K.mean(score * class_weights)

    return score


def get_f_score(class_weights=1, beta=1, smooth=SMOOTH, per_image=True):
    def score(gt, pr):
        return f_score(gt, pr, class_weights=class_weights, beta=beta, smooth=smooth, per_image=per_image)

    return score


f1_score = get_f_score(beta=1)
f2_score = get_f_score(beta=2)
dice_score = f1_score
get_dice_score = get_f_score

# Update custom objects
get_custom_objects().update({
    'f1_score': f1_score,
    'f2_score': f2_score,
    'dice_score': dice_score,
})
