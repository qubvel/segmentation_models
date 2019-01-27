.. raw:: html

    <p align="center">
      <img width="400" height="200" src="https://cdn1.imggmi.com/uploads/2019/1/27/d7d9d327ea3340445bd82ec5377c56c7-full.png">
    </p>
    
    <h1 align="center"> Segmentation Models </h1>
    
    <p align="center">
      <a href="https://badge.fury.io/py/segmentation-models" alt="PyPI">
        <img src="https://badge.fury.io/py/segmentation-models.svg" /></a>
      <a href="https://segmentation-models.readthedocs.io/en/latest/?badge=latest" alt="Documentation">
        <img src="https://readthedocs.org/projects/segmentation-models/badge/?version=latest" /></a>
      <a href="https://travis-ci.com/qubvel/segmentation_models" alt="Build Status">
        <img src="https://travis-ci.com/qubvel/segmentation_models.svg?branch=master" /></a>
    </p>

**Segmentation models** is python library with Neural Networks for
`Image
Segmentation <https://en.wikipedia.org/wiki/Image_segmentation>`__ based
on `Keras <https://keras.io>`__
(`Tensorflow <https://www.tensorflow.org/>`__) framework.

**The main features** of this library are:

-  High level API (just two lines to create NN)
-  **4** models architectures for binary and multi class segmentation
   (including legendary **Unet**)
-  **15** available backbones for each architecture
-  All backbones have **pre-trained** weights for faster and better
   convergence

Latest **documentation** is avaliable on `Read the
Docs <https://segmentation-models.readthedocs.io/en/latest/>`__

Avaliable models:
~~~~~~~~~~~~~~~~~

-  `Unet <https://arxiv.org/abs/1505.04597>`__
-  `FPN <http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf>`__
-  `Linknet <https://arxiv.org/abs/1707.03718>`__
-  `PSPNet <https://arxiv.org/abs/1612.01105>`__

Avaliable backbones:
~~~~~~~~~~~~~~~~~~~~

+-----------------------+-------------------------+-----------------------------------------------+
| Backbone model        | Name                    | Weights                                       |
+=======================+=========================+===============================================+
| VGG16                 | ``vgg16``               | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| VGG19                 | ``vgg19``               | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| ResNet18              | ``resnet18``            | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| ResNet34              | ``resnet34``            | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| ResNet50              | ``resnet50``            | ``imagenet``\ \ ``imagenet11k-places365ch``   |
+-----------------------+-------------------------+-----------------------------------------------+
| ResNet101             | ``resnet101``           | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| ResNet152             | ``resnet152``           | ``imagenet``\ \ ``imagenet11k``               |
+-----------------------+-------------------------+-----------------------------------------------+
| ResNeXt50             | ``resnext50``           | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| ResNeXt101            | ``resnext101``          | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| DenseNet121           | ``densenet121``         | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| DenseNet169           | ``densenet169``         | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| DenseNet201           | ``densenet201``         | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| Inception V3          | ``inceptionv3``         | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+
| Inception ResNet V2   | ``inceptionresnetv2``   | ``imagenet``                                  |
+-----------------------+-------------------------+-----------------------------------------------+

Requirements
~~~~~~~~~~~~

1) Python 3.5+
2) Keras >=2.1.0
3) Tensorflow >= 1.4

Installation
~~~~~~~~~~~~

Installing via pip
^^^^^^^^^^^^^^^^^^

``$ pip install segmentation-models``

Using latest version in your project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    $ pip install git+https://github.com/qubvel/segmentation_models

Code examples
~~~~~~~~~~~~~

Train Unet model:

.. code:: python

    from segmentation_models import Unet
    from segmentation_models.backbones import get_preprocessing

    # prepare data
    x, y = ...

    preprocessing_fn = get_preprocessing('resnet34')
    x = preprocessing_fn(x)

    # prepare model
    model = Unet(backbone_name='resnet34', encoder_weights='imagenet')
    model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

    # train model
    model.fit(x, y)

Train FPN model:

.. code:: python

    from segmentation_models import FPN

    model = FPN(backbone_name='resnet34', encoder_weights='imagenet')

Useful trick
^^^^^^^^^^^^

Freeze encoder weights for fine-tuning during first epochs of training:

.. code:: python

    from segmentation_models import FPN
    from segmentation_models.utils import set_trainable

    model = FPN(backbone_name='resnet34', encoder_weights='imagenet', freeze_encoder=True)
    model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

    # pretrain model decoder
    model.fit(x, y, epochs=2)

    # release all layers for training
    set_trainable(model) # set all layers trainable and recompile model

    # continue training
    model.fit(x, y, epochs=100)

Change Log
~~~~~~~~~~
CHANGELOG.md_

.. _CHANGELOG.md: https://github.com/qubvel/segmentation_models/blob/readme/CHANGELOG.md
