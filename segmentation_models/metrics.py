from .base import Metric
from .base import functional as F

SMOOTH = 1e-5


class IOUScore(Metric):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:

    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}

    Args:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
       A callable ``iou_score`` instance. Can be used in ``model.compile(...)`` function.

    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index

    Example:

    .. code:: python

        metric = IOUScore()
        model.compile('SGD', loss=loss, metrics=[metric])
    """

    def __init__(self, class_weights=None, threshold=None, per_image=True, smooth=SMOOTH):
        super().__init__(name='iou_score')
        self.class_weights = class_weights or 1
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return F.iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=self.threshold,
            **self.submodules
        )


class FScore(Metric):
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:

    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: L(tp, fp, fn) = \frac{(1 + \beta^2) \cdot tp} {(1 + \beta^2) \cdot fp + \beta^2 \cdot fn + fp}

    where:
         - tp - true positives;
         - fp - false positives;
         - fn - false negatives;

    Args:
        beta: f-score coefficient
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        A callable ``f_score`` instance. Can be used in ``model.compile(...)`` function.

    Example:

    .. code:: python

        metric = FScore()
        model.compile('SGD', loss=loss, metrics=[metric])
    """

    def __init__(self, beta=1, class_weights=None, threshold=None, per_image=True, smooth=SMOOTH):
        super().__init__(name='f{}-score'.format(beta))
        self.beta = beta
        self.class_weights = class_weights or 1
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth,

    def __call__(self, gt, pr):
        return F.f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=self.threshold,
            **self.submodules
        )
