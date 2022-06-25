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

  (They should be placed in backbone of object detection, but they are used to extract features, just like classification networks)

  1.DarkNet

  2.VovNet

  (finished)

  1.LeNet5(models/basenets/lenet5.py)

  ```
  I add nn.BatchNorm2d(). This is because that I was so upset about the poor accuracy.

  ```
  [LeNet5](https://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=Lenet-5&step_word=&hs=0&pn=6&spn=0&di=7108135681917976577&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=0&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=undefined&cs=1300733551%2C2088996461&os=2357251395%2C2696632372&simid=1300733551%2C2088996461&adpicid=0&lpn=0&ln=754&fr=&fmq=1656408621595_R&fm=&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=0&oriquery=&objurl=https%3A%2F%2Fgimg2.baidu.com%2Fimage_search%2Fsrc%3Dhttp%3A%2F%2Fpic4.zhimg.com%2Fv2-487d887f21f85764b38975f3169020af_ipico.jpg%26refer%3Dhttp%3A%2F%2Fpic4.zhimg.com%26app%3D2002%26size%3Df9999%2C10000%26q%3Da80%26n%3D0%26g%3D0n%26fmt%3Dauto%3Fsec%3D1659000624%26t%3D113dba9d97bd6cfa4eab2641a8701e53&fromurl=ippr_z2C%24qAzdH3FAzdH3Fooo_z%26e3Bziti7_z%26e3Bv54AzdH3Fq7jfpt5gAzdH3F90dnma8b9AzdH3Fwgfoj6AzdH3F8lll099na9&gsm=7&rpstart=0&rpnum=0&islist=&querylist=&nojc=undefined&dyTabStr=MCw2LDQsMiw1LDMsMSw4LDcsOQ%3D%3D)


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
