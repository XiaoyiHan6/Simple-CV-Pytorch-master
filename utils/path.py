import sys
import os.path

# Gets home dir cross platform
# /data/PycharmProject/Simple-CV-Pytorch-master
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Path to store checkpoint model
# /data/PycharmProject/Simple-CV-Pytorch-master/checkpoints
CheckPoints = 'checkpoints'
CheckPoints = os.path.join(BASE_DIR, CheckPoints)

# Results
# /data/PycharmProject/Simple-CV-Pytorch-master/results
Results = 'results'
Results = os.path.join(BASE_DIR, Results)

# Path to store tensorboard load
# /data/PycharmProject/Simple-CV-Pytorch-master/tensorboard
tensorboard_log = 'tensorboard'
tensorboard_log = os.path.join(BASE_DIR, tensorboard_log)

# Path to save log
# /data/PycharmProject/Simple-CV-Pytorch-master/log
log = 'log'
log = os.path.join(BASE_DIR, log)

# Path to save classification train log
classification_train_log = 'classification_train'

# Path to save classification test log
classification_test_log = 'classification_test'

# Path to save classification eval log
classification_eval_log = 'classification_eval'

# Path to save detection train log
detection_train_log = 'detection_train'

# Path to save detection test log
detection_test_log = 'detection_test'

# Path to save detection eval log
detection_eval_log = 'detection_eval'

# classification evaluate model path
classification_evaluate = None

# detection evaluate model path
detection_evaluate = None

# Images detection path
# /data/PycharmProject/Simple-CV-Pytorch-master/images/detection
image_det = '000001.jpg'
images_det_path = 'images/detection'
images_det_path = os.path.join(BASE_DIR, images_det_path, image_det)

# Images classification path
# /data/PycharmProject/Simple-CV-Pytorch-master/images/classification
images_cls = 'photocopier.png'
images_cls_path = 'images/classification'
images_cls_root = os.path.join(BASE_DIR, images_cls_path, images_cls)

# Data
# classification
# /data/ImageNet/ILSVRC2012
ImageNet = "ImageNet/ILSVRC2012"
ImageNet_TRAIN_ROOT = os.path.join('/data', ImageNet, 'train')
ImageNet_EVAL_ROOT = os.path.join('/data', ImageNet, 'val')

# CIFAR10
# /data/cifar
CIFAR = 'cifar'
CIFAR_ROOT = os.path.join('/data', CIFAR)

# detection
# VOC
# /data/VOCdevkit
VOC = 'VOCdevkit'
VOC_path = os.path.join('/data', VOC)

# COCO
# COCO2017
# (train:118287), (val:5000), (test:40670)
# /data/coco/coco2017
COCO2017 = 'coco/coco2017'
COCO2017_path = os.path.join('/data', COCO2017)

# COCO2014
# (train:82783), (valminusminival:40504 random 35504 -> minival 5000) ~ COCO2017
# /data/coco/coco2014
# COCO2014 = 'coco/coco2014'
# COCO2014_path = os.path.join('/data', COCO2014)

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
MEANS = (104, 117, 123)
