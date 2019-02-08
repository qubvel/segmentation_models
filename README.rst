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
-  **25** available backbones for each architecture
-  All backbones have **pre-trained** weights for faster and better
   convergence

Table of Contents
~~~~~~~~~~~~~~~~~
 - `Quick start`_
 - `Simple training pipeline`_
 - `Models and Backbones`_
 - `Installation`_
 - `Documentation`_
 - `Change log`_
 - `License`_
 
Quick start
~~~~~~~~~~~
Since the library is built on the Keras framework, created segmentaion model is just a Keras Model, which can be created as easy as:

.. code:: python

    from segmentation_models import Unet
    
    model = Unet()
    
Depending on the task, you can change the network architecture by choosing backbones with fewer or more parameters and use pretrainded weights to initialize it:

.. code:: python

    model = Unet('resnet34', encoder_weights='imagenet')

Change number of output classes in the model:

.. code:: python

    model = Unet('resnet34', classes=3, activation='softmax')
    
Change input shape of the model:

.. code:: python

    model = Unet('resnet34', input_shape=(None, None, 6), encoder_weights=None)
   
Simple training pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from segmentation_models import Unet
   from segmentation_models.backbones import get_preprocessing
   from segmentation_models.losses import bce_jaccard_loss
   from segmentation_models.metrics import iou_score
   
   BACKBONE = 'resnet34'
   preprocess_input = get_prepocessing(BACKBONE)
   
   # load your data
   x_train, y_train, x_val, y_val = load_data(...)
   
   # preprocess input
   x_train = preprocess_input(x_train)
   x_val = preprocess_input(x_val)
   
   # define model
   model = Unet(BACKBONE, encoder_weights='imagenet')
   model.complile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
   
   # fit model
   model.fit(
       x=x_train, 
       y=y_train, 
       batch_size=16, 
       epochs=100,
       validation_data=(x_val, y_val),
   )
   

Same manimulations can be done with ``Linknet``, ``PSPNet`` and ``FPN``. For more detailed information about models API and  use cases `Read the Docs <https://segmentation-models.readthedocs.io/en/latest/>`__.

Models and Backbones
~~~~~~~~~~~~~~~~~~~~
**Models**

-  `Unet <https://arxiv.org/abs/1505.04597>`__
-  `FPN <http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf>`__
-  `Linknet <https://arxiv.org/abs/1707.03718>`__
-  `PSPNet <https://arxiv.org/abs/1612.01105>`__

============= ==============
Unet          Linknet
============= ==============
|unet_image|  |linknet_image|
============= ==============
============= ==============
PSPNet        FPN
============= ==============
|psp_image|   |fpn_image|
============= ==============

.. _Unet: https://github.com/qubvel/segmentation_models/blob/readme/LICENSE
.. _Linknet: https://arxiv.org/abs/1707.03718
.. _PSPNet: https://arxiv.org/abs/1612.01105
.. _FPN: http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

.. |unet_image| image:: https://cdn1.imggmi.com/uploads/2019/2/8/a83ca5cfeff0e9ec497b6641522b4ac2-full.png  
.. |linknet_image| image:: https://cdn1.imggmi.com/uploads/2019/2/8/1a996c4ef05531ff3861d80823c373d9-full.png 
.. |psp_image| image:: https://cdn1.imggmi.com/uploads/2019/2/8/aaabb97f89197b40e4879a7299b3c801-full.png
.. |fpn_image| image:: https://cdn1.imggmi.com/uploads/2019/2/8/af00f11ef6bc8a64efd29ed873fcb0c4-full.png

**Backbones**

.. table:: 

    ===========  ===== 
    Type         Names
    ===========  =====
    VGG          ``'vgg16' 'vgg19'``
    ResNet       ``'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'``
    SE-ResNet    ``'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'``
    ResNeXt      ``'resnext50' 'resnet101'``
    SE-ResNeXt   ``'seresnext50' 'seresnet101'``
    SENet154     ``'senet154'``
    DenseNet     ``'densenet121' 'densenet169' 'densenet201'`` 
    Inception    ``'inceptionv3' 'inceptionresnetv2'``
    MobileNet    ``'mobilenet' 'mobilenetv2'``
    ===========  =====

.. epigraph::
    All backbones have weights trained on 2012 ILSVRC ImageNet dataset (``encoder_weights='imagenet'``). 


Installation
~~~~~~~~~~~~

**Requirements**

1) Python 3.5+
2) Keras >= 2.2.0
3) Image-classifiers == 0.2.0
4) Tensorflow 1.9 (tested)

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

License
~~~~~~~
Project is distributed under `MIT Licence`_.

.. _CHANGELOG.md: https://github.com/qubvel/segmentation_models/blob/readme/CHANGELOG.md
.. _`MIT Licence`: https://github.com/qubvel/segmentation_models/blob/readme/LICENSE
