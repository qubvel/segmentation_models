Tutorial
========

Quick start
~~~~~~~~~~~

Binary segmentation
+++++++++++++++++++

Consider a problem of binary segmentation (e.g. segmentation of forest
on remote sensing data) where we have:
 - ``x`` - input images of shape (H, W, C)
 - ``y`` - labeled binary masks of shape (H, W)

If you already familiar with Keras_ API all you need to start is just to
import one of segmentation models and start training:

.. note::

    For binary segmentation ``Unet`` based on  ``resnet34`` backbone is a
    good way to start your experiments.

.. code-block:: python

    from segmentation_models import Unet
    from segmentation_models.backbones import get_preprocessing

    # define backbone name
    BACKBONE = 'resnet34'

    # prepare/load data
    x, y = ...

    preprocessing_fn = get_preprocessing(BACKBONE)
    x = preprocessing_fn(x)

    # prepare model
    model = Unet(backbone_name=BACKBONE, encoder_weights='imagenet')
    model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

    # train model
    model.fit(x, y, epochs=20)

Multi class segmentation
++++++++++++++++++++++++

In case you have ``N`` classes as a target (``N > 1``)
you simply have to provide following arguments while
initializing your model:

.. code-block:: python

    model = Unet(backbone_name=BACKBONE, encoder_weights='imagenet',
                 classes=N, activation='softmax')

.. _Keras:
    https://keras.io

Models and Backbones
~~~~~~~~~~~~~~~~~~~~

Avaliable models
++++++++++++++++
 - Unet
 - FPN
 - Linknet
 - PSPNet

Avaliable backbones
+++++++++++++++++++

=================== ===================== =====================
Backbone model      Name                  Weights
=================== ===================== =====================
VGG16               ``vgg16``             ``imagenet``
VGG19               ``vgg19``             ``imagenet``
ResNet18            ``resnet18``          ``imagenet``
ResNet34            ``resnet34``          ``imagenet``
ResNet50            ``resnet50``          ``imagenet``\  \ ``imagenet11k-places365ch``
ResNet101           ``resnet101``         ``imagenet``
ResNet152           ``resnet152``         ``imagenet``\  \ ``imagenet11k``
ResNeXt50           ``resnext50``         ``imagenet``
ResNeXt101          ``resnext101``        ``imagenet``
DenseNet121         ``densenet121``       ``imagenet``
DenseNet169         ``densenet169``       ``imagenet``
DenseNet201         ``densenet201``       ``imagenet``
Inception V3        ``inceptionv3``       ``imagenet``
Inception ResNet V2 ``inceptionresnetv2`` ``imagenet``
=================== ===================== =====================
