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
