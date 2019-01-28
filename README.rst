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

Table of Contents
~~~~~~~~~~~~~~~~~
 - `Quick start`_
 - `Models and Backbones`_
 - `Installation`_
 - `Documentation`_
 - `Change log`_
 - `Licence`_
 
Quick start
~~~~~~~~~~~
Since the library is built on the Keras framework, created segmentaion model is just a Keras Model, wich can be created as easy as:

.. code:: python

    from segmentation_models import Unet
    
    model = Unet()
    
Depending on the task, you can change the network architecture by choosing backbones with fewer or more parameters and use pretrainded weights for to initialize it:

.. code:: python

    model = Unet('resnet34', encoder_weights='imagenet')

Change number of output classes in the model:

.. code:: python

    model = Unet('resnet34', classes=3, activation='softmax')
    
Change input shape of the model:

.. code:: python

    model = Unet('resnet34', input_shape=(None, None, 6), encoder_weights=None)

Same manimulations can be done with ``Linknet``,``PSPNet`` and ``FPN``. For more detailed information about models API and  use cases read Documentation_.

Models and Backbones
~~~~~~~~~~~~~~~~~~~~
**Models**

-  `Unet <https://arxiv.org/abs/1505.04597>`__
-  `FPN <http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf>`__
-  `Linknet <https://arxiv.org/abs/1707.03718>`__
-  `PSPNet <https://arxiv.org/abs/1612.01105>`__

**Backbones**

.. table:: 

    ===========  ===== 
    Type         Names
    ===========  =====
    VGG          ``'vgg16' 'vgg19'``
    ResNet       ``'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'``
    ResNeXt      ``'resnext50' 'resnet101'``
    DenseNet     ``'densenet121' 'densenet169' 'densenet201'`` 
    Inception    ``'inceptionv3' 'inceptionresnetv2'``
    ===========  =====

.. epigraph::
    All backbones have weights trained on 2012 ILSVRC ImageNet dataset (``encoder_weights='imagenet'``). 


Installation
~~~~~~~~~~~~

**Requirements**

1) Python 3.5+
2) Keras >= 2.1.0
3) Tensorflow >= 1.8

**Pip package**

.. code:: bash

    $ pip install segmentation-models

**Latest version**

.. code:: bash

    $ pip install git+https://github.com/qubvel/segmentation_models
    
Documentation
~~~~~~~~~~~~~
Latest **documentation** is avaliable on `Read the
Docs <https://segmentation-models.readthedocs.io/en/latest/>`__

Change Log
~~~~~~~~~~
To see important changes between versions look at CHANGELOG.md_

Licence
~~~~~~~
Project is distributed under `MIT Licence`_.

.. _CHANGELOG.md: https://github.com/qubvel/segmentation_models/blob/readme/CHANGELOG.md
.. _`MIT Licence`: https://github.com/qubvel/segmentation_models/blob/readme/LICENCE
