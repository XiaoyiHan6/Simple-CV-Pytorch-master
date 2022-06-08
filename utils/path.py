import os.path

# Gets home dir cross platform
# "/home/scz1174"
HOME = os.path.expanduser("~")
MyName = "run/hxy/"
Folder = "CV-Pytorch-master"

# Path to store checkpoint model
CheckPoints = 'checkpoints'
CheckPoints = os.path.join(HOME, MyName, Folder, CheckPoints)

# Results
Results = 'results'
Results = os.path.join(HOME, MyName, Folder, Results)

# Path to store tensorboard load
tensorboard_log = 'tensorboard'
tensorboard_log = os.path.join(Folder, tensorboard_log)

# Path to save log
log = 'log'
log = os.path.join(HOME, MyName, Folder, log)

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
images_path = os.path.join(HOME, MyName, Folder, images_path, image)

# Data
DATAPATH = '/data/public'

# ImageNet2012
ImageNet2012 = 'imagenet2012'
ImageNet2012_Train_path = os.path.join(DATAPATH, ImageNet2012, 'train')
ImageNet2012_Eval_path = os.path.join(DATAPATH, ImageNet2012, 'val')

# ImageNetmini
ImageNetmini = 'imagenet-mini'
ImageNetmini_Train_path = os.path.join(DATAPATH, ImageNetmini, 'train')
ImageNetmini_Eval_path = os.path.join(DATAPATH, ImageNetmini, 'val')

# CIFAR10 or CIFAR100
CIFAR = 'cifar'
CIFAR_path = os.path.join(DATAPATH, CIFAR)

# VOC0712
VOC0712 = 'PascalVOC'
VOC_path = os.path.join(DATAPATH, VOC0712)

# coco
# COCO2017
# (train:118287), (val:5000), (test:40670)
COCO2017 = 'coco2017'
COCO2017_path = os.path.join(DATAPATH, COCO2017)

# COCO2014
# (train:82783), (valminusminival:40504 random 35504 -> minival 5000) ~ COCO2017
COCO2014 = 'coco2014'
COCO2014_path = os.path.join(DATAPATH, COCO2014)
