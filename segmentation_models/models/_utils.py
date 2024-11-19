from keras_applications import get_submodules_from_kwargs


def freeze_model(model, fraction=1.0, **kwargs):
    """Set layers non trainable, excluding BatchNormalization layers.
       If a fraction is specified, only a fraction of the layers are
       frozen (starting with the earliest layers)"""
    _, layers, _, _ = get_submodules_from_kwargs(kwargs)
    for layer in model.layers[:int(len(model.layers) * fraction)]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return


def filter_keras_submodules(kwargs):
    """Selects only arguments that define keras_application submodules. """
    submodule_keys = kwargs.keys() & {'backend', 'layers', 'models', 'utils'}
    return {key: kwargs[key] for key in submodule_keys}
