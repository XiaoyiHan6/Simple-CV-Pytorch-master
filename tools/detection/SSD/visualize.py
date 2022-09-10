import warnings

warnings.filterwarnings('ignore')

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import cv2
import time
import random
import logging
import argparse
import numpy as np
from data import *
from tqdm import tqdm
import torch.nn.parallel
from models.detection.SSD import SSD
from utils.get_logger import get_logger
from data.voc0712 import VOC_CLASSES as labelmap
from utils.augmentations.SSDAugmentations import BaseTransform

devkit_path = os.path.join(results_path, 'SSD')


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Testing')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset',
                        type=str,
                        default='VOC',
                        choices=['VOC', 'COCO'],
                        help='Dataset type, must be one of VOC or COCO.')
    parser.add_argument('--dataset_root',
                        type=str,
                        default=VOC_ROOT,
                        choices=[COCO_ROOT, VOC_ROOT],
                        help='Path to COCO or VOC directory')
    parser.add_argument('--model',
                        type=str,
                        default='ssd',
                        help='Testing Model')
    parser.add_argument('--training',
                        type=str,
                        default=False,
                        help='Model is training or testing')
    parser.add_argument('--evaluate',
                        type=str,
                        default=config.detection_evaluate,
                        help='Checkpoint state_dict file to evaluate training from')
    parser.add_argument('--save_folder',
                        type=str,
                        default=config.checkpoint_path,
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log_folder',
                        type=str,
                        default=config.log,
                        help='Log Folder')
    parser.add_argument('--log_name',
                        type=str,
                        default=config.detection_test_log,
                        help='Log Name')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to train model')
    parser.add_argument('--pretrained',
                        type=str,
                        default=False,
                        help='Models was pretrained')
    parser.add_argument('--thresh',
                        type=float,
                        default=0.6,
                        help='Models thresh')
    return parser.parse_args()


args = parse_args()


def write_test_results(dataset, model, transform, img_size=300):
    devkitpath = os.path.join(devkit_path, 'VOC')
    filename = os.path.join(devkitpath, 'visualize.txt')
    if os.path.exists(filename):
        os.remove(filename)

    if not os.path.exists(devkitpath):
        os.mkdir(devkitpath)

    # 8. Forward
    with tqdm(total=len(dataset.ids)) as pbar:
        for img_ind, info in enumerate(dataset.ids):
            # h,w,c
            img = dataset.load_image(img_ind)
            annot = dataset.load_annots(img_ind)

            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

            with open(filename, mode='a') as f:
                f.write('\nGROUND TRUTH FOR: ' + info[1] + '\n')
                for a in annot:
                    f.write('label: ' + labelmap[int(a[-1])] + ' ' + str(a[-1]) +
                            ' || coords: ' + ' || '.join(str(i) for i in a[:-1]) + '\n')

            if torch.cuda.is_available():
                # c,h,w
                img = torch.from_numpy(transform(img)).permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
            else:
                img = torch.from_numpy(transform(img)).permute(2, 0, 1).float().unsqueeze(dim=0)

            output = model(img)
            pred_num = 0
            for i in range(len(output)):
                boxes, scores, labels = output[i]
                boxes = boxes.cpu()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                labels -= 1
                results = []
                gt_labels_copy = list(set(annot[:, 4]))
                for j, a in enumerate(gt_labels_copy):
                    x = 0
                    num_copy = list(annot[:, 4]).count(a)
                    if a in labels:
                        for num_copy_i in range(num_copy):
                            try:
                                out = list(labels).index(a, x)
                            except Exception as e:
                                continue
                            if out in results:
                                continue
                            else:
                                results.append(out)
                                x = out + 1
                    else:
                        continue
                if results is []:
                    break
                else:
                    for _, result in enumerate(results):
                        if pred_num == 0:
                            with open(filename, mode='a') as f:
                                f.write('PREDICTIONS: ' + '\n')
                        score = scores[result]
                        boxes[result] /= img_size
                        label_name = labelmap[labels[result]]
                        coord = (boxes[result] * scale).numpy()
                        pred_num += 1
                        with open(filename, mode='a') as f:
                            f.write(str(pred_num) + ' label: '
                                    + label_name + ' ' + str(labels[result]) + ' || ' + 'score: ' +
                                    str(score) + ' || ' + 'coord: ' + ' || '.join(
                                str(c) for c in coord) + '\n')

            pbar.update(1)
    pbar.close()
    return


def Visualized(dataset, model, transform, img_size=300):
    # shuffle
    random.shuffle(dataset.ids)
    info = dataset.ids[1]
    img = dataset.load_image(1)
    annot = dataset.load_annots(1)
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    for i, a in enumerate(annot):
        xmin = int(a[0])
        ymin = int(a[1])
        xmax = int(a[2])
        ymax = int(a[3])
        label_a = str(int(a[4]))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, label_a, (xmin, ymin), font, 1, (0, 0, 255), 1)

    # h,w,c

    if torch.cuda.is_available():
        # c,h,w
        image = torch.from_numpy(transform(img)).permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
    else:
        image = torch.from_numpy(transform(img)).permute(2, 0, 1).float().unsqueeze(dim=0)

    output = model(image)
    for i in range(len(output)):
        boxes, scores, labels = output[i]
        boxes = boxes.cpu()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        labels -= 1
        results = []
        gt_labels_copy = list(set(annot[:, 4]))
        for j, a in enumerate(gt_labels_copy):
            x = 0
            num_copy = list(annot[:, 4]).count(a)
            if a in labels:
                for num_copy_i in range(num_copy):
                    try:
                        out = list(labels).index(a, x)
                    except Exception as e:
                        continue
                    if out in results:
                        continue
                    else:
                        results.append(out)
                        x = out + 1
            else:
                continue
        if results is []:
            break
        else:
            for _, result in enumerate(results):
                score = scores[result]
                boxes[result] /= img_size
                coord = (boxes[result] * scale).numpy()
                text = str(labels[result]) + ' | ' + str(np.round(score, 2))
                cv2.rectangle(img, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (255, 0, 255), 2)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(img, text, (int(coord[2]), int(coord[3])), font, 1, (255, 0, 0), 1)

        if args.dataset == 'VOC':
            filename = os.path.join(devkit_path, 'VOC', info[1] + '_VOC.jpg')
        elif args.dataset == 'COCO':
            filename = os.path.join(devkit_path, 'COCO', info + '_COCO.jpg')
        cv2.imwrite(filename, img)
    return


# 1. Torch choose cuda or cpu
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but you aren't using it" +
              "\n You can set the parameter of cuda to True.")
        torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(devkit_path):
    os.mkdir(devkit_path)

# 2. Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)


def test():
    # 3. Ready dataset
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset COCO')
        elif args.dataset_root is None:
            raise ValueError("WARNING: Using default COCO dataset, but " +
                             "--dataset_root was not specified.")

        dataset_test = CocoDetection(args.dataset_root, set_name='val2017')

    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset VOC')
        elif args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on VOC')

        dataset_test = VocDetection(args.dataset_root, image_sets=[('2007', 'test')])

    else:
        raise ValueError('Dataset type not understood (must be voc or coco), exiting.')

    # 4. Define to train mode
    model = SSD(version=args.dataset,
                training=args.training,
                batch_norm=False)

    print("Using model ssd...")

    if args.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    # 5. Loading model
    if args.evaluate:
        other, ext = os.path.splitext(args.evaluate)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            model_evaluate_load = os.path.join(args.save_folder, args.evaluate)
            model.load_state_dict(torch.load(model_evaluate_load))
        else:
            print('Sorry only .pth and .pkl files supported.')
    elif args.evaluate is None:
        print("Sorry, you should load weights! ")

    model.eval()

    # 6. print
    logger.info(f"args - {args}")

    # 7. Test
    with torch.no_grad():
        t0 = time.time()
        if args.dataset == 'VOC':
            write_test_results(dataset_test, model, transform=BaseTransform(), img_size=300)
        Visualized(dataset_test, model, transform=BaseTransform(), img_size=300)
        t1 = time.time()
        m = (t1 - t0) // 60
        s = (t1 - t0) % 60
    print("It took a total of {}m{}s to complete the testing.".format(int(m), int(s)))
    return


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    logger.info("Program started")
    test()
    logger.info("Done!")
