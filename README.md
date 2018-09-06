# Segmentation models Zoo
Segmentation models with pretrained backbones

#### Unet and FPN like models
| Backbone model      |Name| Weights    | UNet |  FPN | 
|---------------------|:--:|:------------:|:------:|:------:| 
| VGG16               |`vgg16`| `imagenet` | +    | +    | 
| VGG19               |`vgg19`| `imagenet` | +    | +    | 
| ResNet18            |`resnet18`| `imagenet` | +    | +    | 
| ResNet34            |`resnet34`| `imagenet` | +    | +    | 
| ResNet50            |`resnet50`| `imagenet`<br>`imagenet11k-places365ch` | +    | +    | 
| ResNet101           |`resnet101`| `imagenet` | +    | +    | 
| ResNet152           |`resnet152`| `imagenet`<br>`imagenet11k`<br>`imagenet11k-places365ch` | +    | +    | 
| ResNeXt50           |`resnext50`| `imagenet` | +    | +    | 
| ResNeXt101          |`resnext101`| `imagenet` | +    | +    | 
| DenseNet121         |`densenet121`| `imagenet` | +    | +    | 
| DenseNet169         |`densenet169`| `imagenet` | +    | +    | 
| DenseNet201         |`densenet201`| `imagenet` | +    | +    | 
| Inception V3        |`inceptionv3`| `imagenet` | +    | +    | 
| Inception ResNet V2 |`inceptionresnetv2`| `imagenet` | +    | +    | 


### Installation
1) Clone repositoriy to your project  
```bash
$ git clone https://github.com/qubvel/segmentation_models.git
```
2) Update submodules  
```bash
$ cd segmentation_models
$ git submodule update --init --recursive
```

### Code examples

Train Unet model:  
```python
from segmentation_models import Unet

# prepare data
x, y = ...

# prepare model
model = Unet(backbone_name='resnet34', encoder_weigths='imagenet')
model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

# train model
model.fit(x, y)
```
Train FPN model:  
```python
from segmentation_models import FPN

model = FPN(backbone_name='resnet34', encoder_weigths='imagenet')
```

#### Useful trick
Freeze encoder weights for fine-tuning during first epochs of training:
```python
from segmentation_models import FPN
from segmentation_models.utils import set_trainable

model = FPN(backbone_name='resnet34', encoder_weigths='imagenet', freeze_encoder=True)
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
- [ ] Add Linknet models
- [ ] Add PSP models
- [ ] Add DPN backbones
