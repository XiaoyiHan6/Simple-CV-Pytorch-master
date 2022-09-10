import os.path
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
# Gets home dir cross platform
# "/data/"

MyName = "PycharmProject"
Folder = "Simple-CV-Pytorch-master"

# Path to store checkpoint model
CheckPoints = 'checkpoints'
CheckPoints = os.path.join(BASE_DIR, MyName, Folder, CheckPoints)

# Results
Results = 'results'
Results = os.path.join(BASE_DIR, MyName, Folder, Results)

# Path to store tensorboard load
tensorboard_log = 'tensorboard'
tensorboard_log = os.path.join(BASE_DIR, MyName, Folder, tensorboard_log)

# Path to save log
log = 'log'
log = os.path.join(BASE_DIR, MyName, Folder, log)

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
detection_evaluate = 'ssd_voc_best.pth'

# Images detection path
image_det = '000001.jpg'
images_det_path = 'images/detection'
images_det_path = os.path.join(BASE_DIR, MyName, Folder, images_det_path, image_det)

# Images classification path
image_cls = 'automobile.png'
images_cls_path = 'images/classification'
images_cls_path = os.path.join(BASE_DIR, MyName, Folder, images_cls_path, image_cls)

# Data
DATAPATH = BASE_DIR

# ImageNet/ILSVRC2012
ImageNet = "ImageNet/ILSVRC2012"
ImageNet_Train_path = os.path.join(DATAPATH, ImageNet, 'train')
ImageNet_Eval_path = os.path.join(DATAPATH, ImageNet, 'val')

# CIFAR10
CIFAR = 'cifar'
CIFAR_path = os.path.join(DATAPATH, CIFAR)

# VOC0712
VOC0712 = 'VOCdevkit'
VOC_path = os.path.join(DATAPATH, VOC0712)

# coco
# COCO2017
# (train:118287), (val:5000), (test:40670)
COCO2017 = 'coco/coco2017'
COCO2017_path = os.path.join(DATAPATH, COCO2017)

# COCO2014
# (train:82783), (valminusminival:40504 random 35504 -> minival 5000) ~ COCO2017
# COCO2014 = 'coco/coco2014'
# COCO2014_path = os.path.join(DATAPATH, COCO2014)

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)
