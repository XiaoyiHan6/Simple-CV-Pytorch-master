# Simple-CV-Pytorch-master

This code includes detection and classification tasks in Computer Vision, and semantic segmentation task will be added
later.

For classification, I reproduced LeNet5, VGG16, AlexNet, ResNet. Then I will reproduce GoogLeNet, ResNetXt, MobileNet,
ShuffleNet, EiffcientNet, etc.

For object detection, I reproduced RetinaNet. (I broke the code up into modules, such as backbones, necks, heads,
loss,etc.
This makes it easier to modify and add code.) Of course, other object detection algorithms will be added later.

You should create **checkpoint**(model save), **log**, **results** and **tenshorboard**(loss visualization) file
package.

## Compiling environment

Install requirements with pip (you can put requirements.txt to venv/Scripts/, if you need it.)

```
pip install -r requirements.txt
```

```
python == 3.9.12

torch == 1.11.0+cu113

torchvision== 0.11.0+cu113

torchaudio== 0.12.0+cu113

pycocotools == 2.0.4

numpy

Cython

matplotlib

opencv-python 

skimage

tqdm

thop
```

## Dataset Path

Please, watch FolderOrganization.txt ( There are more details. )

## Folder Organization

```

I use Ubuntu20.04 (OS).

project path: /data/PycharmProject

Simple-CV-master path: /data/PycharmProject/Simple-CV-Pytorch-master
|
|----checkpoints ( resnet50-19c8e357.pth \COCO_ResNet50.pth[RetinaNet]\ VOC_ResNet50.pth[RetinaNet] )
|
|            |----cifar.py （ null, I just use torchvision.datasets.ImageFolder ）
|            |----CIAR_labels.txt
|            |----coco.py
|            |----coco_eval.py
|            |----coco_labels.txt
|----data----|----__init__.py
|            |----config.py ( path )
|            |----imagenet.py ( null, I just use torchvision.datasets.ImageFolder )
|            |----ImageNet_labels.txt
|            |----voc0712.py
|            |----voc_eval.py
|            |----voc_labels.txt
|                                     |----crash_helmet.jpg
|----images----|----classification----|----sunflower.jpg
|              |                      |----photocopier.jpg
|              |                      |----automobile.jpg
|              |
|              |----detection----|----000001.jpg
|                                |----000001.xml
|                                |----000002.jpg
|                                |----000002.xml
|                                |----000003.jpg
|                                |----000003.xml
|
|----log(XXX[ detection or classification ]_XXX[  train or test or eval ].info.log)
|
|              |----__init__.py
|              | 
|              |              |----__init.py
|              |----anchor----|----RetinaNetAnchors.py
|              |
|              |               |----lenet5.py
|              |               |----alexnet.py
|              |----basenet----|----vgg.py
|              |               |----resnet.py    
|              |
|              |                 |----DarkNetBackbone.py
|              |----backbones----|----__init__.py ( Don't finish writing )
|              |                 |----ResNetBackbone.py
|              |                 |----VovNetBackbone.py
|              |                 
|              |                 
|              |
|----models----|----heads----|----__init.py
|              |             |----RetinaNetHeads.py
|              |
|              |              |----RetinaNetLoss.py      
|              |----losses----|----__init.py
|              |
|              |             |----FPN.py
|              |----necks----|----__init__.py
|              |             |-----FPN.txt
|              |
|              |----RetinaNet.py
|
|----results ( eg: detection ( VOC or COCO AP ) )
|
|----tensorboard ( Loss visualization )
|
|----tools                       |----eval.py
|         |----classification----|----train.py
|         |                      |----test.py
|         |               
|         |               
|         |
|         |                 |----eval_coco.py
|         |                 |----eval_voc.py
|         |----detection----|----test.py
|                           |----train.py
|                      
|
|             |----AverageMeter.py
|             |----BBoxTransform.py
|             |----ClipBoxes.py
|             |----Sampler.py 
|             |----iou.py            
|----utils----|----__init__.py
|             |----accuracy.py
|             |----augmentations.py
|             |----collate.py
|             |----get_logger.py
|             |----nms.py
|             |----path.py
|
|----FolderOrganization.txt
|
|----main.py
|
|----README.md
|
|----requirements.txt

```

## Run the program

### 1.classification

- Reproduce network architectures

  1.GoogLeNet

  2.ResNetXt

  3.MobileNet

  4.ShuffleNet

  5.EfficientNet

  (They should be placed in backbone of object detection, but they are used to extract features, just like
  classification networks)

  1.DarkNet

  2.VovNet

  (finished)

  1.LeNet5(models/basenets/lenet5.py)

  ```
  I add nn.BatchNorm2d(). This is because that I was so upset about the poor accuracy.
  basenet: lenet5 (image size: 32 * 32 * 3)
  dataset: cifar
  len(dataset): 50000, iter_size: 1562 
  batch_size: 32
  optim: SGD
  scheduler: MultiStepLR
  milestones: [15, 20, 30]
  weight_decay: 1e-4
  gamma: 0.1
  momentum: 0.9
  lr: 0.01
  epoch: 30
  ```
  [LeNet5](./images/icon/lenet5.png)

  |epoch|times|top1 acc (%)|top5 acc (%)|
  |--|--|--|--|--|
  |0|0h0min23s|50.00|93.75|
  |1|0h0min21s|62.50|96.88|
  |2|0h0min24s|65.62|96.88|
  |3|0h0min21s|53.12|96.88|
  |29|0h0min23s|75.00|100.00|

2.AlexNet(models/basenets/alexnet.py)

3.VGG(models/basenets/vgg.py)

4.ResNet(models/basenets/resnet.py)

- Run

```
#!/bin/bash
conda activate base
python /data/PycharmProject/Simple-CV-Pytorch-master/tools/classification/XXX.py(train.py or eval.py or test.py)
```

### 2.object detection

- Reproduce network architectures

  1.SSD

  2.Faster RCNN

  3.YOLO

  (finished)

  1.RetinaNet(models/RetinaNet.py)

- Run

```
#!/bin/bash
conda activate base
python /data/PycharmProject/Simple-CV-Pytorch-master/tools/detection/XXX.py(train.py or eval_coco.py or eval_voc.py or test.py)
```

### 3.semantic segmentation

- Reproduce network architectures

  1.FCN

  2.DeepLab

  3.U-Net
