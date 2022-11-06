import argparse
import os
import sys
import yaml
from data.detection.SSD.coco import COCO_ROOT, COCODetection
from data.detection.SSD.voc0712 import VOC_ROOT, VOCDetection
from utils.path import log, Results, CheckPoints, detection_evaluate, detection_test_log

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Testing')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--training',
                        type=str,
                        default=False,
                        help='Model is training or testing')
    parser.add_argument('--Results',
                        type=str,
                        default=Results,
                        help='Save the result files.')
    parser.add_argument('--evaluate',
                        type=str,
                        default=detection_evaluate,
                        help='Checkpoint state_dict file to evaluate training from')
    parser.add_argument('--save_folder',
                        type=str,
                        default=CheckPoints,
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log_folder',
                        type=str,
                        default=log,
                        help='Log Folder')
    parser.add_argument('--log_name',
                        type=str,
                        default=detection_test_log,
                        help='Log Name')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to train model')
    parser.add_argument('--visual_threshold',
                        default=0.6,
                        type=float,
                        help='Final confidence threshold')
    parser.add_argument('--config',
                        type=str,
                        default='{}/configs/detection/ssd300_coco.yaml'.format(BASE_DIR),
                        help='configuration file *.yaml')

    return parser.parse_args()


args = parse_args()


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
        dataset_test = COCODetection(cfg['DATA']['ROOT'], image_set='val2017')

    elif cfg['DATA']['NAME'] == 'VOC':
        if cfg['DATA']['ROOT'] == COCO_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset_root VOC')
        elif cfg['DATA']['ROOT'] is None:
            raise ValueError('Must provide --dataset_root when training on VOC')
        dataset_test = VOCDetection(cfg['DATA']['ROOT'], image_sets=[('2007', 'test')])
    else:
        raise ValueError('Dataset type not understood (must be VOC or COCO), exiting.')


set_config = set_config()

# args
args = set_config.args

# cfg
cfg = set_config.cfg

# dataset_test
dataset_test = set_config.dataset_test
