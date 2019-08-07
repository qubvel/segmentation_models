Segmentation Models Python API
==============================


Getting started with segmentation models is easy.

Unet
~~~~
.. autofunction:: segmentation_models.Unet

Linknet
~~~~~~~
.. autofunction:: segmentation_models.Linknet

FPN
~~~
.. autofunction:: segmentation_models.FPN

PSPNet
~~~~~~
.. autofunction:: segmentation_models.PSPNet

metrics
~~~~~~~
.. autofunction:: segmentation_models.metrics.IOUScore
.. autofunction:: segmentation_models.metrics.FScore

losses
~~~~~~
.. autofunction:: segmentation_models.losses.JaccardLoss
.. autofunction:: segmentation_models.losses.DiceLoss
.. autofunction:: segmentation_models.losses.BinaryCELoss
.. autofunction:: segmentation_models.losses.CategoricalCELoss
.. autofunction:: segmentation_models.losses.BinaryFocalLoss
.. autofunction:: segmentation_models.losses.CategoricalFocalLoss

utils
~~~~~
.. autofunction:: segmentation_models.utils.set_trainable