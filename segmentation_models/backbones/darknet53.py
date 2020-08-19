import os

from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras import backend as K
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Reshape, Flatten, Softmax, GlobalMaxPooling2D, \
    ZeroPadding2D, Add, Concatenate
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.utils.layer_utils import get_source_inputs

from neural_network_manager.nnm_models.yolo_v4_modules.layers import DarknetConv2D, DarknetConv2D_BN_Mish, compose
from neural_network_manager.nnm_models.yolo_v4_modules.yolo4_darknet import BASE_WEIGHT_PATH


def csp_darknet53(input_shape=None,
                  input_tensor=None,
                  include_top=True,
                  weights='imagenet',
                  pooling=None,
                  classes=1000,
                  **kwargs):
    """Generate cspdarknet53 model for Imagenet classification."""

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=28,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    x = csp_darknet53_body(img_input)

    if include_top:
        model_name = 'cspdarknet53'
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Reshape((1, 1, 1024))(x)
        x = DarknetConv2D(classes, (1, 1))(x)
        x = Flatten()(x)
        x = Softmax(name='Predictions/Softmax')(x)
    else:
        model_name = 'cspdarknet53_headless'
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            file_name = 'cspdarknet53_weights_tf_dim_ordering_tf_kernels_224.h5'
            weight_path = BASE_WEIGHT_PATH + file_name
        else:
            file_name = 'cspdarknet53_weights_tf_dim_ordering_tf_kernels_224_no_top.h5'
            weight_path = BASE_WEIGHT_PATH + file_name

        weights_path = get_file(file_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(x)

    res_connection = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(x)

    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Mish(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (3, 3)))(x)
        x = Add()([x, y])

    x = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(x)
    x = Concatenate()([x, res_connection])

    return DarknetConv2D_BN_Mish(num_filters, (1, 1))(x)


def csp_darknet53_body(x):
    '''CSPDarknet53 body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Mish(32, (3, 3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x