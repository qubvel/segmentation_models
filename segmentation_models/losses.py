from .base import Loss
from .base import functional as F

SMOOTH = 1e-5


class JaccardLoss(Loss):

    def __init__(self, class_weights=None, per_image=True, smooth=SMOOTH):
        super().__init__(name='jaccard_loss')
        self.class_weights = class_weights or 1.
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
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights or 1.
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


class BinaryCELoss(Loss):

    def __init__(self):
        super().__init__(name='binary_crossentropy')

    def call(self, gt, pr, **kwargs):
        return F.bianary_crossentropy(gt, pr, **kwargs)


class CategoricalCELoss(Loss):

    def __init__(self, class_weights=None):
        super().__init__(name='categorical_crossentropy')
        self.class_weights = class_weights

    def call(self, gt, pr, **kwargs):
        return F.categorical_crossentropy(gt, pr, self.class_weights, **kwargs)


class CategoricalFocalLoss(Loss):

    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def call(self, gt, pr, **kwargs):
        return F.categorical_focal_loss(gt, pr, self.alpha, self.gamma, **kwargs)


class BinaryFocalLoss(Loss):

    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def call(self, gt, pr, **kwargs):
        return F.binary_focal_loss(gt, pr, self.alpha, self.gamma, **kwargs)


# aliases
jaccard_loss = JaccardLoss()
dice_loss = DiceLoss()

binary_focal_loss = BinaryFocalLoss()
categorical_focal_loss = CategoricalFocalLoss()

binary_crossentropy = BinaryCELoss()
categorical_crossentropy = CategoricalCELoss()

# loss combinations
bce_dice_loss = binary_crossentropy + dice_loss
bce_jaccard_loss = binary_crossentropy + jaccard_loss

cce_dice_loss = categorical_crossentropy + dice_loss
cce_jaccard_loss = categorical_crossentropy + jaccard_loss

binary_focal_dice_loss = binary_focal_loss + dice_loss
binary_focal_jaccard_loss = binary_focal_loss + jaccard_loss

categorical_focal_dice_loss = categorical_focal_loss + dice_loss
categorical_focal_jaccard_loss = categorical_focal_loss + jaccard_loss
