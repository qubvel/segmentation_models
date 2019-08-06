from .base import Metric
from .base import functional as F

SMOOTH = 1e-5


class IOUScore(Metric):

    def __init__(self, class_weights=None, threshold=None, per_image=True, smooth=SMOOTH):
        super(IOUScore, self).__init__(name='iou_score')
        self.class_weights = class_weights or 1
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth

    def call(self, gt, pr, **kwargs):
        return F.iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=self.threshold,
            **kwargs
        )


class FScore(Metric):

    def __init__(self, beta=1, class_weights=None, threshold=None, per_image=True, smooth=SMOOTH):
        super(FScore, self).__init__(name='f{}-score'.format(beta))
        self.beta = beta
        self.class_weights = class_weights or 1
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth,

    def call(self, gt, pr, **kwargs):
        return F.f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=self.threshold,
            **kwargs
        )
