# Change Log

**Version 0.2.0** 

###### Areas of improvement

 - New backbones (SE-ResNets, SE-ResNeXts, SENet154, MobileNets)
 - Metrcis:  
    - `iou_score` / `jaccard_score`
    - `f_score` / `dice_score`
 - Losses:  
    - `jaccard_loss` 
    - `bce_jaccard_loss`
    - `cce_jaccard_loss`
    - `dice_loss`
    - `bce_dice_loss`
    - `cce_dice_loss`
  - Documentation [Read the Docs](https://segmentation-models.readthedocs.io)
  - Tests + Travis-CI 
    
###### API changes

 - Some parameters renamed (see API docs)
 - `encoder_freeze=True` does not `freeze` BatchNormalization layer of encoder

###### Thanks

[@IlyaOvodov](https://github.com/IlyaOvodov) [#15](https://github.com/qubvel/segmentation_models/issues/15) [#37](https://github.com/qubvel/segmentation_models/pull/37) investigation of `align_corners` parameter in `ResizeImage` layer  
[@NiklasDL](https://github.com/NiklasDL) [#29](https://github.com/qubvel/segmentation_models/issues/29) investigation about convolution kernel in PSPNet final layers

**Version 0.1.2**  

###### Areas of improvement

 - Added PSPModel
 - Prepocessing functions for all backbones: 
```python
from segmentation_models.backbones import get_preprocessing

preprocessing_fn = get_preprocessing('resnet34')
X = preprocessing_fn(x)
```
###### API changes
- Default param `use_batchnorm=True` for all decoders
- FPN model `Upsample2D` layer renamed to `ResizeImage`

**Version 0.1.1**  
 - Added `Linknet` model
 - Keras 2.2+ compatibility (fixed import of `_obtain_input_shape`)
 - Small code improvements and bug fixes

**Version 0.1.0**  
 - `Unet` and `FPN` models
