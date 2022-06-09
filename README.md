# Simple-CV-Pytorch-master

This code includes detection and classification tasks in Computer Vision, and semantic segmentation task will be added later. 

For classification, I just use torchvision.models.XXX(packages) directly. 

For detection, I reproduced RetinaNet. (I broke the code up into modules, such as backbones, necks, heads, loss,etc. This makes it easier to modify and add code.) Of course, other object detection algorithms will be added later.

You should create **checkpoint**(model save), **log**, **results** and **tenshorboard**(loss visualization) file package.

## Compiling environment

python == 3.8.5

torch == 1.9.0

torchvision== 0.10.0

torchaudio== 0.9.0

pycocotools == 2.0.2

numpy

Cython

matplotlib

opencv-python \ skimage

(tqdm: the progress bar of python)

(thop: the statistics tool of parameter number of network model)

## Folder Organization

```

I use Ubuntu (OS).

project path: home/scz1174/run/hxy

Simple-CV-master path: home/scz1174/run/hxy/Simple-CV-master
|
|----checkpoints ( resnet50-19c8e357.pth \COCO_ResNet50.pth[RetinaNet]\ VOC_ResNet50.pth[RetinaNet] )
|
|        |----cifar.py （ null, I just use torchvision.datasets.ImageFolder ）
|        |----cifar10_labels.txt
|        |----cifar100_labels.txt
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

## run the program
1. run classification
```
#!/bin/bash
module load anaconda/2021.05
module load cuda/11.1
source activate py38
export PYTHONUNBUFFERED=1
python home/scz1174/hxy/Simple-CV-Pytorch-master/tools/classification/XXX.py(train.py or eval.py or test.py)
```
3. run detection 
```
#!/bin/bash
module load anaconda/2021.05
module load cuda/11.1
source activate py38
export PYTHONUNBUFFERED=1
python home/scz1174/hxy/Simple-CV-Pytorch-master/tools/detection/XXX.py(train.py or eval_coco.py or eval_voc.py or test.py)
```
