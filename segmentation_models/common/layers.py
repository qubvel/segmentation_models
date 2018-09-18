from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects

from .functions import resize_images


class ResizeImage(Layer):
    """ResizeImage layer for 2D inputs.
    Repeats the rows and columns of the data
    by factor[0] and factor[1] respectively.
    # Arguments
        factor: int, or tuple of 2 integers.
            The upsampling factors for rows and columns.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        interpolation: A string, one of `nearest` or `bilinear`.
            Note that CNTK does not support yet the `bilinear` upscaling
            and that with Theano, only `factor=(2, 2)` is possible.
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, upsampled_rows, upsampled_cols)`
    """

    @interfaces.legacy_upsampling2d_support
    def __init__(self, factor=(2, 2), data_format='channels_last', interpolation='nearest', **kwargs):
        super(ResizeImage, self).__init__(**kwargs)
        self.data_format = data_format
        self.factor = conv_utils.normalize_tuple(factor, 2, 'factor')
        self.input_spec = InputSpec(ndim=4)
        if interpolation not in ['nearest', 'bilinear']:
            raise ValueError('interpolation should be one '
                             'of "nearest" or "bilinear".')
        self.interpolation = interpolation

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.factor[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.factor[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.factor[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.factor[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        return resize_images(inputs, self.factor[0], self.factor[1],
                             self.data_format, self.interpolation)

    def get_config(self):
        config = {'factor': self.factor,
                  'data_format': self.data_format}
        base_config = super(ResizeImage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'ResizeImage': ResizeImage})
