Installation
============

**Requirements**
 - Python 3.X
 - Keras >=2.1.0
 - Tensorflow >= 1.4
 - scikit-image

.. note::

    This library does not have Tensorflow_ in a requirements
    for installation. Please, choose suitable version ('cpu'/'gpu')
    and install it manually using official Guide_.

.. _Guide:
    https://www.tensorflow.org/install/

.. _Tensorflow:
    https://www.tensorflow.org/

To install library execute at the command line::

 $ pip install segmentation-models

Or use latest version available on github.com::

 $ git clone --recurse-submodules -j8 https://github.com/qubvel/segmentation_models.git
 $ cd segmentation_models
 $ python setup.py install
