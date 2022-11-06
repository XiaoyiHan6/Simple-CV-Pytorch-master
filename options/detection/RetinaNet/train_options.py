import os
import sys
import yaml
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from models.detection.RetinaNet.utils.collate import collate
from data.detection.RetinaNet.voc import VOC_ROOT, VocDetection
from data.detection.RetinaNet.coco import COCO_ROOT, CocoDetection
from utils.path import log, CheckPoints, tensorboard_log, detection_train_log
from models.detection.RetinaNet.utils.augmentations import Resize, RandomFlip, Normalize
from models.detection.RetinaNet.RetinaNet import resnet18_retinanet, resnet34_retinanet, \
    resnet50_retinanet, resnet101_retinanet, resnet152_retinanet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


# argparse
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch RetinNet Training')
    parser.add_argument('--training',
                        type=str,
                        default=True,
                        help='Model is training or testing')
    parser.add_argument('--pretrained',
                        type=str,
                        default=True,
                        help='Pretrained base model')
    parser.add_argument('--tensorboard',
                        type=str,
                        default=True,
                        help='Use tensorboard for loss visualization')
    parser.add_argument('--tensorboard_log',
                        type=str,
                        default=tensorboard_log,
                        help='Use tensorboard for loss visualization')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to train model')
    parser.add_argument('--log_folder',
                        type=str,
                        default=log)
    parser.add_argument('--log_name',
                        type=str,
                        default=detection_train_log)
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--save_folder',
                        type=str,
                        default=CheckPoints,
                        help='Directory for saving checkpoint models')
    parser.add_argument('--config',
                        type=str,
                        default='{}/configs/detection/retinanet_coco.yaml'.format(BASE_DIR),
                        help='configuration file *.yaml')

    return parser.parse_args()


# Load yaml
def parse_config(yaml_path):
    if not os.path.isfile(yaml_path):
        raise ValueError(f'{yaml_path} not exists.')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        try:
            dict = yaml.safe_load(f.read())
        except yaml.YAMLError:
            raise ValueError('Error parsing YAML file:' + yaml_path)
    return dict


class set_config(object):
    # Load yaml
    args = parse_args()
    if args.config:
        cfg = parse_config(args.config)
    else:
        raise ValueError('--config must be specified.')

    # Create the data loaders
    if cfg['DATA']['NAME'] == 'COCO':
        if cfg['DATA']['ROOT'] == VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset COCO')
        elif cfg['DATA']['ROOT'] is None:
            raise ValueError("WARNING: Using default COCO dataset, but " +
                             "--dataset_root was not specified.")

        dataset_train = CocoDetection(cfg['DATA']['ROOT'], set_name='train2017',
                                      transform=transforms.Compose([
                                          Normalize(),
                                          RandomFlip(),
                                          Resize()]))
        dataset_val = CocoDetection(cfg['DATA']['ROOT'], set_name='val2017',
                                    transform=transforms.Compose([
                                        Normalize(),
                                        Resize()]))
    elif cfg['DATA']['NAME'] == 'VOC':
        if cfg['DATA']['ROOT'] == COCO_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset_root VOC')
        elif cfg['DATA']['ROOT'] is None:
            raise ValueError('Must provide --dataset_root when training on VOC')

        dataset_train = VocDetection(cfg['DATA']['ROOT'],
                                     transform=transforms.Compose([
                                         Normalize(),
                                         RandomFlip(),
                                         Resize()]))
        dataset_val = VocDetection(cfg['DATA']['ROOT'], set_name=[('2007', 'test')],
                                   transform=transforms.Compose([
                                       Normalize(),
                                       Resize()]))
    else:
        raise ValueError('Dataset type not understood (must be VOC or COCO), exiting.')
    dataloader_train = DataLoader(dataset_train, num_workers=2, batch_size=cfg['OPTIMIZE']['BATCH_SIZE'],
                                  collate_fn=collate, shuffle=True,
                                  drop_last=True)
    iter_size = len(dataset_train) // cfg['OPTIMIZE']['BATCH_SIZE']

    if cfg['MODEL']['BACKBONE']['NAME'] == 'resnet':
        if cfg['MODEL']['BACKBONE']['DEPTH'] == 18:
            model = resnet18_retinanet(num_classes=dataset_train.num_classes(),
                                       pretrained=args.pretrained,
                                       training=args.training)

            model_eval = resnet18_retinanet(num_classes=dataset_val.num_classes(),
                                            training=False)
        elif cfg['MODEL']['BACKBONE']['DEPTH'] == 34:
            model = resnet34_retinanet(num_classes=dataset_train.num_classes(),
                                       pretrained=args.pretrained,
                                       training=args.training)

            model_eval = resnet34_retinanet(num_classes=dataset_val.num_classes(),
                                            training=False)
        elif cfg['MODEL']['BACKBONE']['DEPTH'] == 50:
            model = resnet50_retinanet(num_classes=dataset_train.num_classes(),
                                       pretrained=args.pretrained,
                                       training=args.training)

            model_eval = resnet50_retinanet(num_classes=dataset_val.num_classes(),
                                            training=False)
        elif cfg['MODEL']['BACKBONE']['DEPTH'] == 101:
            model = resnet101_retinanet(num_classes=dataset_train.num_classes(),
                                        pretrained=args.pretrained,
                                        training=args.training)

            model_eval = resnet101_retinanet(num_classes=dataset_val.num_classes(),
                                             training=False)
        elif cfg['MODEL']['BACKBONE']['DEPTH'] == 152:
            model = resnet152_retinanet(num_classes=dataset_train.num_classes(),
                                        pretrained=args.pretrained,
                                        training=args.training)

            model_eval = resnet152_retinanet(num_classes=dataset_val.num_classes(),
                                             training=False)
        else:
            raise ValueError("Unsupported RetinaNet Model depth!")
    else:
        raise ValueError('Unsupported model type!')


set_config = set_config()
# args
args = set_config.args

# cfg
cfg = set_config.cfg

# dataset_train
dataset_train = set_config.dataset_train

# dataloader_train
dataloader_train = set_config.dataloader_train

# dataset_val
dataset_val = set_config.dataset_val

# iter_size
iter_size = set_config.iter_size

# model
retinanet = set_config.model

retinanet_eval = set_config.model_eval

if __name__ == '__main__':
    print("args:{}\nconfig:{}.".format(args, cfg))
    for data in dataloader_train:
        imgs, targets = data['img'], data['annot']
        print("imgs:", imgs.shape)
        print("targets:", targets.shape)
