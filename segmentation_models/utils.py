""" Utility functions for segmentation models """

from keras_applications import get_submodules_from_kwargs
from . import inject_global_submodules


def set_trainable(model, recompile=True, **kwargs):
    """Set all layers of model trainable and recompile it

    Note:
        Model is recompiled using same optimizer, loss and metrics::

            model.compile(
                model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
                loss_weights=model.loss_weights,
                sample_weight_mode=model.sample_weight_mode,
                weighted_metrics=model.weighted_metrics,
            )

    Args:
        model (``keras.models.Model``): instance of keras model

    """
    for layer in model.layers:
        layer.trainable = True

    if recompile:
        model.compile(
            model.optimizer,
            loss=model.loss,
            metrics=model.metrics,
            loss_weights=model.loss_weights,
            sample_weight_mode=model.sample_weight_mode,
            weighted_metrics=model.weighted_metrics,
        )


@inject_global_submodules
def set_regularization(
        model,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        beta_regularizer=None,
        gamma_regularizer=None,
        **kwargs
):
    """Set regularizers to all layers

    Note:
       Returned model's config is updated correctly

    Args:
        model (``keras.models.Model``): instance of keras model
        kernel_regularizer(``regularizer`): regularizer of kernels
        bias_regularizer(``regularizer``): regularizer of bias
        activity_regularizer(``regularizer``): regularizer of activity
        gamma_regularizer(``regularizer``): regularizer of gamma of BatchNormalization
        beta_regularizer(``regularizer``): regularizer of beta of BatchNormalization

    Return:
        out (``Model``): config updated model
    """
    _, _, models, _ = get_submodules_from_kwargs(kwargs)

    for layer in model.layers:
        # set kernel_regularizer
        if kernel_regularizer is not None and hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = kernel_regularizer
        # set bias_regularizer
        if bias_regularizer is not None and hasattr(layer, 'bias_regularizer'):
            layer.bias_regularizer = bias_regularizer
        # set activity_regularizer
        if activity_regularizer is not None and hasattr(layer, 'activity_regularizer'):
            layer.activity_regularizer = activity_regularizer

        # set beta and gamma of BN layer
        if beta_regularizer is not None and hasattr(layer, 'beta_regularizer'):
            layer.beta_regularizer = beta_regularizer

        if gamma_regularizer is not None and hasattr(layer, 'gamma_regularizer'):
            layer.gamma_regularizer = gamma_regularizer

    out = models.model_from_json(model.to_json())
    out.set_weights(model.get_weights())

    return out
