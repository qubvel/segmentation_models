.. raw:: html

    <p align="center">
      <img src="https://i.ibb.co/GtxGS8m/Segmentation-Models-V1-Side-3-1.png">
      <b>Python library with Neural Networks for Image Segmentation based on <a href=https://www.keras.io>Keras</a> and <a href=https://www.tensorflow.org>TensorFlow</a>.
      </b>
      <br></br>

      <a href="https://badge.fury.io/py/segmentation-models" alt="PyPI">
        <img src="https://badge.fury.io/py/segmentation-models.svg" /></a>
      <a href="https://segmentation-models.readthedocs.io/en/latest/?badge=latest" alt="Documentation">
        <img src="https://readthedocs.org/projects/segmentation-models/badge/?version=latest" /></a>
      <a href="https://travis-ci.com/qubvel/segmentation_models" alt="Build Status">
        <img src="https://travis-ci.com/qubvel/segmentation_models.svg?branch=master" /></a>
    </p>


**The main features** of this library are:

-  High level API (just two lines of code to create model for segmentation)
-  **4** models architectures for binary and multi-class image segmentation
   (including legendary **Unet**)
-  **25** available backbones for each architecture
-  All backbones have **pre-trained** weights for faster and better
   convergence
- Helpful segmentation losses (Jaccard, Dice, Focal) and metrics (IoU, F-score)

**Important note**

    Some models of version ``1.*`` are not compatible with previously trained models,
    if you have such models and want to load them - roll back with:

    $ pip install -U segmentation-models==0.2.1

Table of Contents
~~~~~~~~~~~~~~~~~
 - `Quick start`_
 - `Simple training pipeline`_
 - `Examples`_
 - `Models and Backbones`_
 - `Installation`_
 - `Documentation`_
 - `Change log`_
 - `Citing`_
 - `License`_
 
Quick start
~~~~~~~~~~~
Library is build to work together with Keras and TensorFlow Keras frameworks

.. code:: python

    import segmentation_models as sm
    # Segmentation Models: using `keras` framework.

By default it tries to import ``keras``, if it is not installed, it will try to start with ``tensorflow.keras`` framework.
There are several ways to choose framework:

- Provide environment variable ``SM_FRAMEWORK=keras`` / ``SM_FRAMEWORK=tf.keras`` before import ``segmentation_models``
- Change framework ``sm.set_framework('keras')`` /  ``sm.set_framework('tf.keras')``

You can also specify what kind of ``image_data_format`` to use, segmentation-models works with both: ``channels_last`` and ``channels_first``.
This can be useful for further model conversion to Nvidia TensorRT format or optimizing model for cpu/gpu computations.

.. code:: python

    import keras
    # or from tensorflow import keras

    keras.backend.set_image_data_format('channels_last')
    # or keras.backend.set_image_data_format('channels_first')

Created segmentation model is just an instance of Keras Model, which can be build as easy as:

.. code:: python
    
    model = sm.Unet()
    
Depending on the task, you can change the network architecture by choosing backbones with fewer or more parameters and use pretrainded weights to initialize it:

.. code:: python

    model = sm.Unet('resnet34', encoder_weights='imagenet')

Change number of output classes in the model (choose your case):

.. code:: python
    
    # binary segmentation (this parameters are default when you call Unet('resnet34')
    model = sm.Unet('resnet34', classes=1, activation='sigmoid')
    
.. code:: python
    
    # multiclass segmentation with non overlapping class masks (your classes + background)
    model = sm.Unet('resnet34', classes=3, activation='softmax')
    
.. code:: python
    
    # multiclass segmentation with independent overlapping/non-overlapping class masks
    model = sm.Unet('resnet34', classes=3, activation='sigmoid')
    
    
Change input shape of the model:

.. code:: python
    
    # if you set input channels not equal to 3, you have to set encoder_weights=None
    # how to handle such case with encoder_weights='imagenet' described in docs
    model = Unet('resnet34', input_shape=(None, None, 6), encoder_weights=None)
   
Simple training pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import segmentation_models as sm

    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # load your data
    x_train, y_train, x_val, y_val = load_data(...)

    # preprocess input
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # define model
    model = sm.Unet(BACKBONE, encoder_weights='imagenet')
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

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

Same manipulations can be done with ``Linknet``, ``PSPNet`` and ``FPN``. For more detailed information about models API and  use cases `Read the Docs <https://segmentation-models.readthedocs.io/en/latest/>`__.

Examples
~~~~~~~~
Models training examples:
 - [Jupyter Notebook] Binary segmentation (`cars`) on CamVid dataset `here <https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb>`__.
 - [Jupyter Notebook] Multi-class segmentation (`cars`, `pedestrians`) on CamVid dataset `here <https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb>`__.

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
    EfficientNet   ``'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7'``
    =============  =====

.. epigraph::
    All backbones have weights trained on 2012 ILSVRC ImageNet dataset (``encoder_weights='imagenet'``). 


Installation
~~~~~~~~~~~~

**Requirements**

1) python 3
2) keras >= 2.2.0 or tensorflow >= 1.13
3) keras-applications >= 1.0.7, <=1.0.8
4) image-classifiers == 1.0.*
5) efficientnet == 1.0.*

**PyPI stable package**

.. code:: bash

    $ pip install -U segmentation-models

**PyPI latest package**

.. code:: bash

    $ pip install -U --pre segmentation-models

**Source latest version**

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
      Author = {Pavel Iakubovskii},
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
