import os
import sys
import yaml
import argparse
from torchvision import transforms
from models.detection.RetinaNet.RetinaNet import RetinaNet
from data.detection.RetinaNet.voc import VOC_ROOT, VocDetection
from data.detection.RetinaNet.coco import COCO_ROOT, CocoDetection
from models.detection.RetinaNet.utils.augmentations import Resize, Normalize
from utils.path import log, Results, CheckPoints, detection_test_log, detection_evaluate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


# argparse
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch RetinNet Testing')
    parser.add_argument('--training',
                        type=str,
                        default=False,
                        help='Model is training or testing')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to train model')
    parser.add_argument('--log_folder',
                        type=str,
                        default=log)
    parser.add_argument('--log_name',
                        type=str,
                        default=detection_test_log)
    parser.add_argument('--Results',
                        type=str,
                        default=Results,
                        help='Save the result files.')
    parser.add_argument('--evaluate',
                        type=str,
                        default=detection_evaluate,
                        help='Checkpoint state_dict file to evaluate training from')
    parser.add_argument('--visual_threshold',
                        default=0.5,
                        type=float,
                        help='Final confidence threshold')
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

        dataset_test = CocoDetection(cfg['DATA']['ROOT'], set_name='val2017',
                                     transform=transforms.Compose([
                                         Normalize(),
                                         Resize()]))
    elif cfg['DATA']['NAME'] == 'VOC':
        if cfg['DATA']['ROOT'] == COCO_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset_root VOC')
        elif cfg['DATA']['ROOT'] is None:
            raise ValueError('Must provide --dataset_root when training on VOC')
        dataset_test = VocDetection(cfg['DATA']['ROOT'], set_name=[('2007', 'test')],
                                    transform=transforms.Compose([
                                        Normalize(),
                                        Resize()]))
    else:
        raise ValueError('Dataset type not understood (must be VOC or COCO), exiting.')

    if cfg['MODEL']['BACKBONE']['NAME'] == 'resnet':
        model = RetinaNet(resnet_type=cfg['MODEL']['BACKBONE']['NAME'] + str(cfg['MODEL']['BACKBONE']['DEPTH']),
                          num_classes=dataset_test.num_classes(),
                          training=args.training)
    else:
        raise ValueError('Unsupported backbones type!')


set_config = set_config()
# args
args = set_config.args

# cfg
cfg = set_config.cfg

# dataset_test
dataset_test = set_config.dataset_test

# model
model = set_config.model

if __name__ == '__main__':
    print("args:{}\nconfig:{}.".format(args, cfg))
    for data in dataset_test:
        imgs, targets = data['img'], data['annot']
        print("imgs:", imgs.shape)
        print("targets:", targets)
