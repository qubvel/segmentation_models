from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa
from segmentation_models.backbones.backbones_factory import Backbones
from tensorflow.keras import layers
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import ResizeMethod

BACKBONE_FEATURES = {
    'resnet34': ['stage3_unit6_relu2', 'stage1_unit3_relu2'],
    'resnet50': ['stage3_unit6_relu3', 'stage1_unit3_relu3'],
    'resnet101': ['stage3_unit23_relu3', 'stage1_unit3_relu3'],
    'resnet152': ['stage3_unit36_relu3', 'stage1_unit3_relu3']
}


class DeepLabV3PlusDecoder(layers.Layer):
    def __init__(
            self,
            filters=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
            highres_filters=48,
            dropout=0.5,
            separable_aspp=False
    ):
        super(DeepLabV3PlusDecoder, self).__init__(name="decoder")
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.aspp = ASPP(filters, atrous_rates, separable=separable_aspp, droput=dropout)

        scale_factor = 2 if output_stride == 8 else 4
        self.up = layers.UpSampling2D(interpolation='bilinear', size=scale_factor)

        self.conv = layers.Conv2D(highres_filters, kernel_size=1, use_bias=False)
        self.batch = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, features, *args, **kwargs):
        aspp_features = self.aspp(features[0])
        aspp_features = self.up(aspp_features)
        high_res_features = self.conv(features[1])
        high_res_features = self.batch(high_res_features)
        high_res_features = self.relu(high_res_features)
        concat_features = tf.keras.layers.Concatenate(axis=-1)([aspp_features, high_res_features])
        return concat_features


class ConvBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(
            filters,
            use_bias=False,
            **kwargs
        )
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        return self.relu(x)


class SeparableConvBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(SeparableConvBlock, self).__init__()
        self.conv = layers.SeparableConv2D(
            filters,
            use_bias=False,
            **kwargs
        )
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        return self.relu(x)


class ASPPPooling(layers.Layer):
    def __init__(self, filters):
        super(ASPPPooling, self).__init__()
        self.avg_pooling = tfa.layers.AdaptiveAveragePooling2D(1)
        self.conv = layers.Conv2D(filters, kernel_size=1, use_bias=False)
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, *args, **kwargs):
        size = x.shape[1:-1]
        x = self.avg_pooling(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return tf.image.resize(x, size, method=ResizeMethod.BILINEAR)


class ASPP(layers.Layer):
    def __init__(self, filters, atrous_rates, separable=False, droput=0.5):
        super(ASPP, self).__init__()
        self.layers = []

        self.layers.append(ConvBlock(filters, kernel_size=1))

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvBlock = ConvBlock if not separable else SeparableConvBlock

        self.layers.append(ASPPConvBlock(filters, dilation_rate=rate1, kernel_size=3, padding='same'))
        self.layers.append(ASPPConvBlock(filters, dilation_rate=rate2, kernel_size=3, padding='same'))
        self.layers.append(ASPPConvBlock(filters, dilation_rate=rate3, kernel_size=3, padding='same'))
        self.layers.append(ASPPPooling(filters))

        self.conv = layers.Conv2D(filters, kernel_size=1, use_bias=False)
        self.batch = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.drop = layers.Dropout(droput)
        self.sep_conv = layers.SeparableConv2D(filters, kernel_size=3, use_bias=False, padding="same")
        self.batch_2 = layers.BatchNormalization()
        self.relu_2 = layers.ReLU()

    def call(self, x, *args, **kwargs):
        res = []
        for layer in self.layers:
            res.append(layer(x))
        res = tf.keras.layers.Concatenate(axis=-1)(res)
        x = self.conv(res)
        x = self.batch(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.sep_conv(x)
        x = self.batch_2(x)
        return self.relu(x)


class SegmentationHead(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, activation=None, upsampling=1):
        super(SegmentationHead, self).__init__()
        self.conv2d = layers.Conv2D(out_channels, kernel_size=kernel_size)
        self.upsampling = layers.UpSampling2D(
            size=upsampling,
            interpolation='bilinear'
            ) if upsampling > 1 else tf.identity
        self.activation = layers.Activation(activation)

    def call(self, inputs, *args, **kwargs):
        x = self.conv2d(inputs)
        x = self.upsampling(x)
        return self.activation(x)


class DeepLabV3Plus(tf.keras.Model):
    def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_weights: Optional[str] = "imagenet",
            encoder_output_stride: int = 16,
            decoder_atrous_rates: tuple = (12, 24, 36),
            input_shape=(128, 128, 3),
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            highres_filters: int = 48,
            separable_aspp: bool = False,
            dropout: float = 0.5,
            **kwargs
    ):
        super(DeepLabV3Plus, self).__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride)
            )

        backbone_features = BACKBONE_FEATURES.get(encoder_name)

        if backbone_features is None:
            raise ValueError(f"Encoder {encoder_name} is not supported")

        encoder = Backbones.get_backbone(
            encoder_name,
            input_shape=input_shape,
            weights=encoder_weights,
            include_top=False,
            **kwargs
        )

        input_1 = encoder.inputs
        output_1 = encoder.get_layer(backbone_features[0]).output
        self.features1 = tf.keras.Model(inputs=input_1, outputs=output_1)

        input_2 = encoder.inputs
        output_2 = encoder.get_layer(backbone_features[1]).output
        self.features2 = tf.keras.Model(inputs=input_2, outputs=output_2)

        self.decoder = DeepLabV3PlusDecoder(
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            highres_filters=highres_filters,
            separable_aspp=separable_aspp,
            dropout=dropout,
        )

        self.segmentation_head = SegmentationHead(
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

    def call(self, inputs):
        feature1 = self.features1(inputs)
        feature2 = self.features2(inputs)
        x = self.decoder([feature1, feature2])
        return self.segmentation_head(x)
