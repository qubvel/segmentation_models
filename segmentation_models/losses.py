from .base import Loss
from .base import functional as F

SMOOTH = 1e-5


class JaccardLoss(Loss):
    r"""Jaccard loss function for imbalanced datasets:

    .. math:: L(A, B) = 1 - \frac{A \cap B}{A \cup B}

    Args:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        callable: jaccard_loss
    """

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
    r"""Dice loss function for imbalanced datasets:

    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        callable: dice_loss
    """

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
    """Creates a criterion that measures the Binary Cross Entropy between the
    ground truth (gt) and the prediction (pr)

    Returns:
        callable: binary_crossentropy
    """

    def __init__(self):
        super().__init__(name='binary_crossentropy')

    def call(self, gt, pr, **kwargs):
        return F.bianary_crossentropy(gt, pr, **kwargs)


class CategoricalCELoss(Loss):
    """Creates a criterion that measures the Categorical Cross Entropy between the
    ground truth (gt) and the prediction (pr)

    Returns:
        callable: categorical_crossentropy
    """

    def __init__(self, class_weights=None):
        super().__init__(name='categorical_crossentropy')
        self.class_weights = class_weights

    def call(self, gt, pr, **kwargs):
        return F.categorical_crossentropy(gt, pr, self.class_weights, **kwargs)


class CategoricalFocalLoss(Loss):
    r"""Implementation of Focal Loss from the paper in multiclass classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)

    Args:
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0

    """

    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def call(self, gt, pr, **kwargs):
        return F.categorical_focal_loss(gt, pr, self.alpha, self.gamma, **kwargs)


class BinaryFocalLoss(Loss):
    r"""Implementation of Focal Loss from the paper in binary classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr) \
               - (1 - gt) * alpha * (pr^gamma) * log(1 - pr)

    Args:
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0

    """

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
