from .voc0712 import VOC_CLASSES as labelmap
from .voc0712 import VOC_ROOT, VocDetection
from .coco import COCO_CLASSES, COCO_ROOT, CocoDetection
from .imagenet import ImageNet_Train_ROOT, ImageNet_Eval_ROOT
from .cifar import CIFAR_ROOT
from .coco_eval import evaluate_coco
from .voc_eval import evaluate_voc
from .config import *
