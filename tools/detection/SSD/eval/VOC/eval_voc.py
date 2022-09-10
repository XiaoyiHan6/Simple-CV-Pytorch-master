import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import time
import torch
import logging
import argparse
import numpy as np
from data import *
from voc_eval import evaluate_voc
from models.detection.SSD import SSD
from utils.get_logger import get_logger

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Evaluation')
    parser.add_argument('--dataset',
                        type=str,
                        default='VOC',
                        help='Dataset type, must be one of VOC.')
    parser.add_argument('--dataset_root',
                        type=str,
                        default=VOC_ROOT,
                        help='Path to COCO directory')
    parser.add_argument('--model',
                        type=str,
                        default='ssd',
                        help='Evaluation Model')
    parser.add_argument('--depth',
                        type=int,
                        default=0,
                        help='Model depth, including SSD of 0')
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
                        default=config.log)
    parser.add_argument('--log_name',
                        type=str,
                        default=config.detection_eval_log)
    parser.add_argument('--evaluate',
                        type=str,
                        default=config.detection_evaluate,
                        help='Checkpoint state_dict file to evaluate training from')
    parser.add_argument('--save_folder',
                        type=str,
                        default=config.checkpoint_path,
                        help='Directory for saving checkpoint models')

    return parser.parse_args()


args = parse_args()

# 1. Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)


def eval():
    # 2. Create the data loaders
    if args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset VOC')
        elif args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on VOC')

        dataset_val = VocDetection(args.dataset_root, image_sets=[('2007', 'test')])


    else:
        raise ValueError('Dataset type not understood (must be voc), exiting.')

    # 3. Create the model
    model = SSD(version=args.dataset,
                training=args.training,
                batch_norm=False)

    print("Using model ssd...")

    if args.cuda:
        if torch.cuda.is_available():
            model = model.cuda()
            model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    if args.evaluate:
        other, ext = os.path.splitext(args.evaluate)
        if ext == '.pkl' or '.pth':
            print("Loading weights into state dict...")
            model_load = os.path.join(args.save_folder, args.evaluate)
            model.load_state_dict(torch.load(model_load))

        else:
            print("Sorry only .pth and .pkl files supported.")

    logger.info(f"{args}")
    t0 = time.time()
    # 4. interference
    with torch.no_grad():
        aps, labelmap = evaluate_voc(dataset_val, model)

    if aps:
        for index, ap in enumerate(aps):
            logger.info(f"{labelmap[index]}:{ap:1.4f}")
        logger.info(f"Mean AP:{np.mean(aps):1.4f}")

    t1 = time.time()
    m = (t1 - t0) // 60
    s = (t1 - t0) % 60
    print("The Finished Time is {}m{}s".format(int(m), int(s)))

    return


if __name__ == '__main__':
    logger.info("Program evaluating started!")
    eval()
    logger.info("Done!")
