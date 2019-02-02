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
.. autofunction:: segmentation_models.metrics.iou_score
.. autofunction:: segmentation_models.metrics.f_score

losses
~~~~~~
.. autofunction:: segmentation_models.losses.jaccard_loss
.. autofunction:: segmentation_models.losses.dice_loss


utils
~~~~~
.. autofunction:: segmentation_models.backbones.get_preprocessing
.. autofunction:: segmentation_models.utils.set_trainable