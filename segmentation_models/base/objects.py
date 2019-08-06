class KerasObject:
    _backend = None
    _models = None
    _layers = None
    _utils = None

    def __init__(self, name=None):
        if self.backend is None:
            raise RuntimeError('You cannot use `KerasObjects` with None submodules.')

        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, name):
        self._name = name

    @classmethod
    def set_submodules(cls, backend, layers, models, utils):
        cls._backend = backend
        cls._layers = layers
        cls._models = models
        cls._utils = utils

    @property
    def backend(self):
        return self._backend

    @property
    def layers(self):
        return self._layers

    @property
    def models(self):
        return self._models

    @property
    def utils(self):
        return self._utils

    def call(self, *args, **kwargs):
        raise NotImplementedError


class Metric(KerasObject):

    def __call__(self, *args, **kwargs):
        kwargs['backend'] = self.backend
        self.call(*args, **kwargs)


class Loss(KerasObject):

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __call__(self, *args, **kwargs):
        kwargs['backend'] = self.backend
        self.call(*args, **kwargs)


class MultipliedLoss(Loss):

    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split('+')) > 1:
            name = '{} * ({})'.format(multiplier, loss.__name__)
        else:
            name = '{} * {}'.format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def call(self, gt, pr, **kwargs):
        return self.multiplier * self.loss(gt, pr, **kwargs)


class SumOfLosses(Loss):

    def __init__(self, l1, l2):
        name = '{} + {}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def call(self, gt, pr, **kwargs):
        return self.l1(gt, pr, **kwargs) + self.l2(gt, pr, **kwargs)
