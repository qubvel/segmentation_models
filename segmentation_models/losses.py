from .base import Loss
from .base import functional as F

SMOOTH = 1e-5


class JaccardLoss(Loss):
    r"""Creates a criterion to measure Jaccard loss:

    .. math:: L(A, B) = 1 - \frac{A \cap B}{A \cup B}

    Args:
        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        per_image: If ``True`` loss is calculated for each image in batch and then averaged,
            else loss is calculated for the whole batch.
        smooth: Value to avoid division by zero.

    Returns:
         A callable ``jaccard_loss`` instance. Can be used in ``model.compile(...)`` function
         or combined with other losses.

    Example:

    .. code:: python

        loss = JaccardLoss()
        model.compile('SGD', loss=loss)
    """

    def __init__(self, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='jaccard_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return 1 - F.iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None,
            **self.submodules
        )


class DiceLoss(Loss):
    r"""Creates a criterion to measure Dice loss:

    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: L(tp, fp, fn) = \frac{(1 + \beta^2) \cdot tp} {(1 + \beta^2) \cdot fp + \beta^2 \cdot fn + fp}

    where:
         - tp - true positives;
         - fp - false positives;
         - fn - false negatives;

    Args:
        beta: Float or integer coefficient for precision and recall balance.
        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        per_image: If ``True`` loss is calculated for each image in batch and then averaged,
        else loss is calculated for the whole batch.
        smooth: Value to avoid division by zero.

    Returns:
        A callable ``dice_loss`` instance. Can be used in ``model.compile(...)`` function`
        or combined with other losses.

    Example:

    .. code:: python

        loss = DiceLoss()
        model.compile('SGD', loss=loss)
    """

    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return 1 - F.f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None,
            **self.submodules
        )


class BinaryCELoss(Loss):
    """Creates a criterion that measures the Binary Cross Entropy between the
    ground truth (gt) and the prediction (pr).

    .. math:: L(gt, pr) = - gt \cdot \log(pr) - (1 - gt) \cdot \log(1 - pr)

    Returns:
        A callable ``binary_crossentropy`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.

    Example:

    .. code:: python

        loss = BinaryCELoss()
        model.compile('SGD', loss=loss)
    """

    def __init__(self):
        super().__init__(name='binary_crossentropy')

    def __call__(self, gt, pr):
        return F.binary_crossentropy(gt, pr, **self.submodules)


class CategoricalCELoss(Loss):
    """Creates a criterion that measures the Categorical Cross Entropy between the
    ground truth (gt) and the prediction (pr).

    .. math:: L(gt, pr) = - gt \cdot \log(pr)

    Args:
        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

    Returns:
        A callable ``categorical_crossentropy`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.

    Example:

    .. code:: python

        loss = CategoricalCELoss()
        model.compile('SGD', loss=loss)
    """

    def __init__(self, class_weights=None, class_indexes=None):
        super().__init__(name='categorical_crossentropy')
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes

    def __call__(self, gt, pr):
        return F.categorical_crossentropy(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            **self.submodules
        )


class CategoricalFocalLoss(Loss):
    r"""Creates a criterion that measures the Categorical Focal Loss between the
    ground truth (gt) and the prediction (pr).

    .. math:: L(gt, pr) = - gt \cdot \alpha \cdot (1 - pr)^\gamma \cdot \log(pr)

    Args:
        alpha: Float or integer, the same as weighting factor in balanced cross entropy, default 0.25.
        gamma: Float or integer, focusing parameter for modulating factor (1 - p), default 2.0.
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

    Returns:
        A callable ``categorical_focal_loss`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.

    Example:

        .. code:: python

            loss = CategoricalFocalLoss()
            model.compile('SGD', loss=loss)
    """

    def __init__(self, alpha=0.25, gamma=2., class_indexes=None):
        super().__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.class_indexes = class_indexes

    def __call__(self, gt, pr):
        return F.categorical_focal_loss(
            gt,
            pr,
            alpha=self.alpha,
            gamma=self.gamma,
            class_indexes=self.class_indexes,
            **self.submodules
        )


class BinaryFocalLoss(Loss):
    r"""Creates a criterion that measures the Binary Focal Loss between the
    ground truth (gt) and the prediction (pr).

    .. math:: L(gt, pr) = - gt \alpha (1 - pr)^\gamma \log(pr) - (1 - gt) \alpha pr^\gamma \log(1 - pr)

    Args:
        alpha: Float or integer, the same as weighting factor in balanced cross entropy, default 0.25.
        gamma: Float or integer, focusing parameter for modulating factor (1 - p), default 2.0.

    Returns:
        A callable ``binary_focal_loss`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.

    Example:

    .. code:: python

        loss = BinaryFocalLoss()
        model.compile('SGD', loss=loss)
    """

    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, gt, pr):
        return F.binary_focal_loss(gt, pr, alpha=self.alpha, gamma=self.gamma, **self.submodules)


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
