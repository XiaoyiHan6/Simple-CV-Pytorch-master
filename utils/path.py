import os.path
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
# Gets home dir cross platform
# "/data/PycharmProject"

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
tensorboard_log = os.path.join(Folder, tensorboard_log)

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

# Classification evaluate model path
classification_evaluate = None

# Detection evaluate model path
detection_evaluate = 'COCO_ResNet50_4.pth'

# Images path
image = '000001.jpg'
images_path = 'images/detection'
images_path = os.path.join(BASE_DIR, MyName, Folder, images_path, image)

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
COCO2017 = 'coco2017'
COCO2017_path = os.path.join(DATAPATH, COCO2017)

# COCO2014
# (train:82783), (valminusminival:40504 random 35504 -> minival 5000) ~ COCO2017
# COCO2014 = 'coco2014'
# COCO2014_path = os.path.join(DATAPATH, COCO2014)
