[![PyPI version](https://badge.fury.io/py/segmentation-models.svg)](https://badge.fury.io/py/segmentation-models)
# Segmentation models Zoo
Segmentation models with pretrained backbones

### Avaliable models:
 - [Unet](https://arxiv.org/abs/1505.04597)
 - [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
 - [Linknet](https://arxiv.org/abs/1707.03718)
 - [PSPNet](https://arxiv.org/abs/1612.01105)

### Avaliable backbones:
| Backbone model      |Name| Weights    |
|---------------------|:--:|:------------:|
| VGG16               |`vgg16`| `imagenet` |
| VGG19               |`vgg19`| `imagenet` |
| ResNet18            |`resnet18`| `imagenet` |
| ResNet34            |`resnet34`| `imagenet` |
| ResNet50            |`resnet50`| `imagenet`<br>`imagenet11k-places365ch` |
| ResNet101           |`resnet101`| `imagenet` |
| ResNet152           |`resnet152`| `imagenet`<br>`imagenet11k` |
| ResNeXt50           |`resnext50`| `imagenet` |
| ResNeXt101          |`resnext101`| `imagenet` |
| DenseNet121         |`densenet121`| `imagenet` |
| DenseNet169         |`densenet169`| `imagenet` |
| DenseNet201         |`densenet201`| `imagenet` |
| Inception V3        |`inceptionv3`| `imagenet` |
| Inception ResNet V2 |`inceptionresnetv2`| `imagenet` |

### Requirements
1) Python 3.6 or higher
2) Keras >=2.1.0
3) Tensorflow >= 1.4

### Installation  

#### Installing via pip  
`$ pip install segmentation_models`

#### Using latest version in your project
```bash
$ git clone https://github.com/qubvel/segmentation_models.git
$ cd segmentation_models
$ git submodule update --init --recursive
```

### Code examples

Train Unet model:  
```python
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
```
Train FPN model:  
```python
from segmentation_models import FPN

model = FPN(backbone_name='resnet34', encoder_weights='imagenet')
```

#### Useful trick
Freeze encoder weights for fine-tuning during first epochs of training:
```python
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
```

### TODO
- [x] Update Unet API
- [x] Update FPN API
- [x] Add Linknet models
- [x] Add PSP models
- [ ] Add DPN backbones

### Change Log

**Version 0.1.2**  

###### Areas of improvement

 - Added PSPModel
 - Prepocessing functions for all backbones: 
```python
from segmentation_models import get_preprocessing

preprocessing_fn = get_preprocessing('resnet34')
X = preprocessing_fn(x)
```
###### API changes
- Default param 'use_batchnorm=True` for all decoders
- FPN model `Upsample2D` layer renamed to `ResizeImage`

**Version 0.1.1**  
 - Added `Linknet` model
 - Keras 2.2+ compatibility (fixed import of `_obtain_input_shape`)
 - Small code improvements and bug fixes

**Version 0.1.0**  
 - `Unet` and `FPN` models
