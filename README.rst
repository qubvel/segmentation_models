.. raw:: html

    <p align="center">
      <img width="400" height="200" src="https://github.com/qubvel/segmentation_models/blob/master/images/logo.png">
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
 - `Citing`_
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

Change number of output classes in the model (choose your case):

.. code:: python
    
    # binary segmentation (this parameters are default when you call Unet('resnet34')
    model = Unet('resnet34', classes=1, activation='sigmoid')
    
.. code:: python
    
    # multiclass segmentation with non overlapping class masks (your classes + background)
    model = Unet('resnet34', classes=3, activation='softmax')
    
.. code:: python
    
    # multiclass segmentation with independent overlapping/non-overlapping class masks
    model = Unet('resnet34', classes=3, activation='sigmoid')
    
    
Change input shape of the model:

.. code:: python
    
    # if you set input channels not equal to 3, you have to set encoder_weights=None
    # how to handle such case with encoder_weights='imagenet' described in docs
    model = Unet('resnet34', input_shape=(None, None, 6), encoder_weights=None)
   
Simple training pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from segmentation_models import Unet
   from segmentation_models.backbones import get_preprocessing
   from segmentation_models.losses import bce_jaccard_loss
   from segmentation_models.metrics import iou_score
   
   BACKBONE = 'resnet34'
   preprocess_input = get_preprocessing(BACKBONE)
   
   # load your data
   x_train, y_train, x_val, y_val = load_data(...)
   
   # preprocess input
   x_train = preprocess_input(x_train)
   x_val = preprocess_input(x_val)
   
   # define model
   model = Unet(BACKBONE, encoder_weights='imagenet')
   model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
   
   # fit model
   # if you use data generator use model.fit_generator(...) instead of model.fit(...)
   # more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
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

.. |unet_image| image:: https://github.com/qubvel/segmentation_models/blob/master/images/unet.png
.. |linknet_image| image:: https://github.com/qubvel/segmentation_models/blob/master/images/linknet.png
.. |psp_image| image:: https://github.com/qubvel/segmentation_models/blob/master/images/pspnet.png
.. |fpn_image| image:: https://github.com/qubvel/segmentation_models/blob/master/images/fpn.png

**Backbones**

.. table:: 

    =============  ===== 
    Type           Names
    =============  =====
    VGG            ``'vgg16' 'vgg19'``
    ResNet         ``'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'``
    SE-ResNet      ``'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'``
    ResNeXt        ``'resnext50' 'resnext101'``
    SE-ResNeXt     ``'seresnext50' 'seresnext101'``
    SENet154       ``'senet154'``
    DenseNet       ``'densenet121' 'densenet169' 'densenet201'`` 
    Inception      ``'inceptionv3' 'inceptionresnetv2'``
    MobileNet      ``'mobilenet' 'mobilenetv2'``
    EfficientNet   ``'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3'``
    =============  =====

.. epigraph::
    All backbones have weights trained on 2012 ILSVRC ImageNet dataset (``encoder_weights='imagenet'``). 


Installation
~~~~~~~~~~~~

**Requirements**

1) Python 3.5+
2) Keras >= 2.2.0
3) Keras Application >= 1.0.7
4) Image Classifiers == 0.2.0
5) Tensorflow 1.9 (tested)

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

Citing
~~~~~~~~

.. code::

    @misc{Yakubovskiy:2019,
      Author = {Pavel Yakubovskiy},
      Title = {Segmentation Models},
      Year = {2019},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/qubvel/segmentation_models}}
    } 

License
~~~~~~~
Project is distributed under `MIT Licence`_.

.. _CHANGELOG.md: https://github.com/qubvel/segmentation_models/blob/master/CHANGELOG.md
.. _`MIT Licence`: https://github.com/qubvel/segmentation_models/blob/master/LICENSE
