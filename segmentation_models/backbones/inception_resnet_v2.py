"""Inception-ResNet V2 model for Keras.
Model naming and structure follows TF-slim implementation
(which has some additional layers and different number of
filters from the original arXiv paper):
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py
Pre-trained ImageNet weights are also converted from TF-slim,
which can be found in:
https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
# Reference
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras_applications import imagenet_utils
from keras_applications import get_submodules_from_kwargs

BASE_WEIGHT_URL = ('https://github.com/fchollet/deep-learning-models/'
                   'releases/download/v0.7/')

backend = None
layers = None
models = None
keras_utils = None


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              activation_dtype=None,
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        activation_dtype: Optional type parameter to force activations
            to be treated in certain type. Used when mixed_precision is enabled.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        if activation_dtype is None:
            x = layers.Activation(activation, name=ac_name)(x)
        else:
            x = layers.Activation(activation, name=ac_name, dtype=activation_dtype)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu', activation_dtype=None):
    """Adds a Inception-ResNet block.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`
    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
        activation_dtype: Optional type parameter to force activations
            to be treated in certain type. Used when mixed_precision is enabled.
    # Returns
        Output tensor for the block.
    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1, activation_dtype=activation_dtype)
        branch_1 = conv2d_bn(x, 32, 1, activation_dtype=activation_dtype)
        branch_1 = conv2d_bn(branch_1, 32, 3, activation_dtype=activation_dtype)
        branch_2 = conv2d_bn(x, 32, 1, activation_dtype=activation_dtype)
        branch_2 = conv2d_bn(branch_2, 48, 3, activation_dtype=activation_dtype)
        branch_2 = conv2d_bn(branch_2, 64, 3, activation_dtype=activation_dtype)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1, activation_dtype=activation_dtype)
        branch_1 = conv2d_bn(x, 128, 1, activation_dtype=activation_dtype)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7], activation_dtype=activation_dtype)
        branch_1 = conv2d_bn(branch_1, 192, [7, 1], activation_dtype=activation_dtype)
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1, activation_dtype=activation_dtype)
        branch_1 = conv2d_bn(x, 192, 1, activation_dtype=activation_dtype)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3], activation_dtype=activation_dtype)
        branch_1 = conv2d_bn(branch_1, 256, [3, 1], activation_dtype=activation_dtype)
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   backend.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv',
                   activation_dtype=activation_dtype)

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        if activation_dtype is None:
            x = layers.Activation(activation, name=block_name + '_ac')(x)
        else:
            x = layers.Activation(activation, name=block_name + '_ac', dtype=activation_dtype)(x)
    return x


def InceptionResNetV2(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      activation_dtype=None,
                      **kwargs):
    """Instantiates the Inception-ResNet v2 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
    # Returns
        A Keras `Model` instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = imagenet_utils._obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='same', activation_dtype=activation_dtype)
    x = conv2d_bn(x, 32, 3, padding='same', activation_dtype=activation_dtype)
    x = conv2d_bn(x, 64, 3, padding='same', activation_dtype=activation_dtype)
    x = layers.MaxPooling2D(3, strides=2, padding='same', activation_dtype=activation_dtype)(x)
    x = conv2d_bn(x, 80, 1, padding='same', activation_dtype=activation_dtype)
    x = conv2d_bn(x, 192, 3, padding='same', activation_dtype=activation_dtype)
    x = layers.MaxPooling2D(3, strides=2, padding='same', activation_dtype=activation_dtype)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1, padding='same', activation_dtype=activation_dtype)
    branch_1 = conv2d_bn(x, 48, 1, padding='same', activation_dtype=activation_dtype)
    branch_1 = conv2d_bn(branch_1, 64, 5, padding='same', activation_dtype=activation_dtype)
    branch_2 = conv2d_bn(x, 64, 1, padding='same', activation_dtype=activation_dtype)
    branch_2 = conv2d_bn(branch_2, 96, 3, padding='same', activation_dtype=activation_dtype)
    branch_2 = conv2d_bn(branch_2, 96, 3, padding='same', activation_dtype=activation_dtype)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same', activation_dtype=activation_dtype)(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, padding='same', activation_dtype=activation_dtype)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='same', activation_dtype=activation_dtype)
    branch_1 = conv2d_bn(x, 256, 1, padding='same', activation_dtype=activation_dtype)
    branch_1 = conv2d_bn(branch_1, 256, 3, padding='same', activation_dtype=activation_dtype)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='same', activation_dtype=activation_dtype)
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='same', activation_dtype=activation_dtype)(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1, padding='same', activation_dtype=activation_dtype)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='same', activation_dtype=activation_dtype)
    branch_1 = conv2d_bn(x, 256, 1, padding='same', activation_dtype=activation_dtype)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='same', activation_dtype=activation_dtype)
    branch_2 = conv2d_bn(x, 256, 1, padding='same', activation_dtype=activation_dtype)
    branch_2 = conv2d_bn(branch_2, 288, 3, padding='same', activation_dtype=activation_dtype)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='same', activation_dtype=activation_dtype)
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='same', activation_dtype=activation_dtype)(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx,
                                   activation_dtype=activation_dtype)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10,
                               activation_dtype=activation_dtype)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b', activation_dtype=activation_dtype)

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if activation_dtype is None:
            x = layers.Dense(classes, activation='softmax', name='predictions')(x)
        else:
            x = layers.Dense(classes, name='dense_logists')(x)
            x = layers.Activation('softmax', dtype=activation_dtype, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='inception_resnet_v2')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            fname = ('inception_resnet_v2_weights_'
                     'tf_dim_ordering_tf_kernels_notop.h5')
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
