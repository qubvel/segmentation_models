from ._base import Loss
from . import functional as F

SMOOTH = 1e-5


class JaccardLoss(Loss):

    def __init__(self, class_weights=None, per_image=True, smooth=SMOOTH):
        super(JaccardLoss, self).__init__(name='jaccard_loss')
        self.class_weights = class_weights or 1
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr, **kwargs):
        return 1 - F.iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None,
            **kwargs
        )


class DiceLoss(Loss):

    def __init__(self, beta=1, class_weights=None, per_image=True, smooth=SMOOTH):
        super(DiceLoss, self).__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights or 1
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr, **kwargs):
        return 1 - F.f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None,
            **kwargs
        )
