# Simple-CV-Pytorch-master

This code includes detection and classification tasks in Computer Vision, and semantic segmentation task will be added
later.

For classification, I reproduced LeNet5, VGG16. Then I will reproduce AlexNet, GoogLeNet, ResNet, ResNetXt, MobileNet, ShuffleNet, EiffcientNet, etc.

For object detection, I reproduced RetinaNet. (I broke the code up into modules, such as backbones, necks, heads, loss,etc.
This makes it easier to modify and add code.) Of course, other object detection algorithms will be added later.

You should create **checkpoint**(model save), **log**, **results** and **tenshorboard**(loss visualization) file
package.

## Compiling environment

python == 3.9.12

torch == 1.11.0+cu113

torchvision== 0.11.0+cu113

torchaudio== 0.12.0+cu113

pycocotools == 2.0.4

numpy

Cython

matplotlib

opencv-python \ skimage

(tqdm: the progress bar of python)

(thop: the statistics tool of parameter number of network model)

## Folder Organization

```

I use Ubuntu20.04 (OS).

project path: /data/PycharmProject

Simple-CV-master path: /data/PycharmProject/Simple-CV-Pytorch-master
|
|----checkpoints ( resnet50-19c8e357.pth \COCO_ResNet50.pth[RetinaNet]\ VOC_ResNet50.pth[RetinaNet] )
|
|        |----cifar.py （ null, I just use torchvision.datasets.ImageFolder ）
|        |----cifar_labels.txt
|        |----coco.py
|        |----coco_eval.py
|        |----coco_labels.txt
|----data----|----__init__.py
|        |----config.py ( path )
|        |----imagenet.py ( null, I just use torchvision.datasets.ImageFolder )
|        |----ImageNet_labels.txt
|        |----voc0712.py
|        |----voc_eval.py
|        |----voc_labels.txt
|
|----images
|            |               |----crash_helmet.jpg
|            |----classification----|----sunflower.jpg
|            |               |----photocopier.jpg
|            |               
|            |            |----000001.jpg
|            |            |----000001.xml
|            |----detection----|----000002.jpg
|                         |----000002.xml
|                         |----000003.jpg
|                         |----000003.xml
|
|----log(XXX[ detection or classification ]_XXX[  train or test or eval ].info.log)
|
|           |----__init__.py
|           | 
|           |       |----__init.py
|           |----anchor----|----RetinaNetAnchors.py
|           |           
|           |            |----DarkNetBackbone.py
|           |----backbones----|----__init__.py ( Don't finish writing )
|           |            |----ResNetBackbone.py
|           |            |----VovNetBackbone.py
|           |            |----lenet5.py
|           |            |----vgg16.py
|           |
|----models----|----heads----|----__init.py
|           |       |----RetinaNetHeads.py
|           |
|           |       |----RetinaNetLoss.py      
|           |----losses----|----__init.py
|           |
|           |         |----FPN.py
|           |----necks----|----__init__.py
|           |         |-----FPN.txt
|           |
|           |----RetinaNet.py
|
|----results ( eg: detection ( VOC or COCO AP ) )
|
|----tensorboard ( Loss visualization )
|
|----tools
|         |----classification
|         |               |----eval.py
|         |               |----test.py
|         |               |----train.py
|         |
|         |            |----eval_coco.py
|         |            |----eval_voc.py
|         |----detection
|                      |----test.py
|                      |----train.py
|
|        |----AverageMeter.py
|        |----BBoxTransform.py
|        |----ClipBoxes.py
|        |----Sampler.py 
|        |----iou.py            
|----utils----|----__init__.py
|         |----accuracy.py
|         |----augmentations.py
|         |----collate.py
|         |----get_logger.py
|         |----nms.py
|         |----path.py
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
    1.AlexNet
    2.GoogLeNet
    3.ResNet
    4.ResNetXt
    5.MobileNet
    6.ShuffleNet
    7.EffidcientNet

    (maybe)
    1.DarkNet
    2.VovNet

    (finished)
    1.LeNet5(models/backbones/lenet5.py)
    2.VGG16(models/backbones/vgg16.py)

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
  1.RetinaNet

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
