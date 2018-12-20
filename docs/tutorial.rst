Tutorial
========

**Segmentation models** is python library with Neural Networks
for `Image Segmentation`_ based on Keras_ (Tensorflow_) framework.

**The main features** of this library are:

-  High level API (just two lines to create NN)
-  **4** models architectures for binary and multi class segmentation (including legendary **Unet**)
-  **15** available backbones for each architecture
-  All backbones have **pre-trained** weights for faster convergence and higher results

Avaliable models
++++++++++++++++

 - Unet_
 - Linknet_
 - FPN_
 - PSPNet_

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


Fine tuning
+++++++++++

Some times, it is useful to train only randomly initialized
*decoder* in order not to damage weights of properly trained
*encoder* with huge gradients during first steps of training.
In this case, all you need is just pass ``freeze_encoder = True`` argument
while initializing the model.

.. code-block:: python

    from segmentation_models import Unet
    from segmentation_models.utils import set_trainable

    model = Unet(backbone_name='resnet34', encoder_weights='imagenet', freeze_encoder=True)
    model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

    # pretrain model decoder
    model.fit(x, y, epochs=2)

    # release all layers for training
    set_trainable(model) # set all layers trainable and recompile model

    # continue training
    model.fit(x, y, epochs=100)


Training with non-RGB data
++++++++++++++++++++++++++

In case you have non RGB images (e.g. grayscale or some medical/remote sensing data)
you have few different options:

1. Train network from scratch with randomly initialized weights

.. code-block:: python

    from segmentation_models import Unet

    # read/scale/preprocess data
    x, y = ...

    # define number of channels
    N = x.shape[-1]

    # define model
    model = Unet(backbone_name='resnet34', encoder_weights=None, input_shape=(None, None, N))

    # continue with usual steps: compile, fit, etc..

2. Add extra convolution layer to map ``N -> 3`` channels data and train with pretrained weights

.. code-block:: python

    from segmentation_models import Unet
    from keras.layers import Input, Conv2D
    from keras.models import Model

    # read/scale/preprocess data
    x, y = ...

    # define number of channels
    N = x.shape[-1]

    base_model = Unet(backbone_name='resnet34', encoder_weights='imagenet')

    inp = Input(shape=(None, None, N))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)

    model = Model(inp, out, name=base_model.name)

    # continue with usual steps: compile, fit, etc..

.. _Image Segmentation:
    https://en.wikipedia.org/wiki/Image_segmentation

.. _Tensorflow:
    https://www.tensorflow.org/

.. _Keras:
    https://keras.io

.. _Unet:
    https://arxiv.org/pdf/1505.04597

.. _Linknet:
    https://arxiv.org/pdf/1707.03718.pdf

.. _PSPNet:
    https://arxiv.org/pdf/1612.01105.pdf

.. _FPN:
    http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf