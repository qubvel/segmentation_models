from ._base import Loss
from . import functional as F

SMOOTH = 1e-5


class JaccardLoss(Loss):

    def __init__(self, class_weights=None, per_image=True, smooth=SMOOTH):
        super(JaccardLoss, self).__init__(name='jaccard_loss')
        self.class_weights = class_weights or 1
        self.per_image = per_image
        self.smooth = smooth

    def call(self, gt, pr, **kwargs):
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

    def call(self, gt, pr, **kwargs):
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


class BCELoss(Loss):

    def __init__(self):
        super().__init__(name='binary_crossentropy')

    def call(self, gt, pr, **kwargs):
        return F.bianary_crossentropy(gt, pr, **kwargs)


class CELoss(Loss):

    def __init__(self, class_weights=None):
        super().__init__(name='categorical_crossentropy')
        self.class_weights = class_weights

    def call(self, gt, pr, **kwargs):
        return F.categorical_crossentropy(gt, pr, self.class_weights, **kwargs)


class FocalLoss(Loss):

    def __init__(self, alpha=0.25, gamma=2.):
        super(FocalLoss, self).__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def call(self, gt, pr, **kwargs):
        return F.categorical_focal_loss(gt, pr, self.alpha, self.gamma, **kwargs)


class BinaryFocalLoss(Loss):

    def __init__(self, alpha=0.25, gamma=2.):
        super(BinaryFocalLoss, self).__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def call(self, gt, pr, **kwargs):
        return F.binary_focal_loss(gt, pr, self.alpha, self.gamma, **kwargs)
