import logging
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from data import *
from models import resnet18_retinanet, resnet34_retinanet, \
    resnet50_retinanet, resnet101_retinanet, resnet152_retinanet
from torchvision import transforms
import torch.nn.parallel
import time
from utils.augmentations import Normalize, Resize
from utils.get_logger import get_logger
import cv2
import random

devkit_path = results_path


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch detection Testing')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset',
                        type=str,
                        default='COCO',
                        choices=['VOC', 'COCO'],
                        help='Dataset type, must be one of VOC or COCO.')
    parser.add_argument('--dataset_root',
                        type=str,
                        default=COCO_ROOT,
                        choices=[COCO_ROOT, VOC_ROOT],
                        help='Path to COCO or VOC directory')
    parser.add_argument('--basenet',
                        type=str,
                        default='ResNet',
                        help='Pretrained base model')
    parser.add_argument('--depth',
                        type=int,
                        default=50,
                        help='Backbone depth, including ResNet of 18, 34, 50, 101, 152')
    parser.add_argument('--training',
                        type=str,
                        default=False,
                        help='Flie is training or testing')
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
                        default=0.3,
                        help='Thresh')

    return parser.parse_args()


args = parse_args()


def write_test_results(dataset, model):
    filename = os.path.join(devkit_path, 'test.txt')
    if os.path.exists(filename):
        os.remove(filename)

    # 8. Forward
    for img_ind, info in enumerate(dataset.ids):
        print('Testing image {:d}/{:d}...'.format(img_ind + 1, len(dataset.ids)))
        data = dataset[img_ind]
        # h,w,c
        img = data['img']
        annot = data['annot']
        scale = data['scale']
        annot[:, 0:4] /= scale

        if torch.cuda.is_available():
            # c,h,1
            img = img.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
        else:
            img = img.permute(2, 0, 1).float().unsqueeze(dim=0)

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: ' + info[1] + '\n')
            for a in annot:
                f.write('label: ' + ' || '.join(str(i.numpy()) for i in a) + '\n')

        pred_score, pred_label, pred_bbox = model(img)

        pred_score = pred_score.cpu()
        pred_label = pred_label.cpu()
        pred_bbox = pred_bbox.cpu()
        pred_bbox /= scale

        pred_num = 0
        for index in range(len(pred_score)):
            if pred_score[index] >= args.thresh:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: ' + '\n')
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num) + ' label: '
                            + str((pred_label[index]).numpy()) + ' || ' + 'score: ' +
                            str((pred_score[index]).numpy()) + ' || ' + 'coords: ' + ' || '.join(
                        str(c.numpy()) for c in pred_bbox[index]) + '\n')


def Visualized(dataset, model):
    # shuffle
    random.shuffle(dataset.ids)
    info = dataset.ids[1]
    data = dataset[1]
    # h,w,c
    img = data['img']
    annot = data['annot']
    scale = data['scale']

    annot[:, 0:4] /= scale

    if torch.cuda.is_available():
        # c,h,w
        img = img.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
    else:
        img = img.permute(2, 0, 1).float().unsqueeze(dim=0)

    pred_score, pred_label, pred_bbox = model(img)

    pred_score = pred_score.cpu()
    pred_label = pred_label.cpu()
    pred_bbox = pred_bbox.cpu()
    pred_bbox /= scale
    if args.dataset == 'VOC':
        image_root = os.path.join(info[0], 'JPEGImages', info[1] + '.jpg')
    elif args.dataset == 'COCO':
        info = str(info).zfill(12)
        image_root = os.path.join(COCO_ROOT, 'test2017', info + ".jpg")

    image = cv2.imread(image_root)
    for i, a in enumerate(annot):
        xmin = int(a[0].numpy())
        ymin = int(a[1].numpy())
        xmax = int(a[2].numpy())
        ymax = int(a[3].numpy())
        label_a = str(int(a[4].numpy()))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, label_a, (xmin + 2, ymin + 2), font, 1, (0, 255, 0), 1)
    for i, pred_s in enumerate(pred_score):
        if pred_s >= args.thresh:
            xmin = int(pred_bbox[i][0].numpy())
            ymin = int(pred_bbox[i][1].numpy())
            xmax = int(pred_bbox[i][2].numpy())
            ymax = int(pred_bbox[i][3].numpy())
            label_b = str(int(pred_label[i].numpy()))
            score_b = str(round(float(pred_score[i].numpy()), 2))
            text = label_b + ' | ' + score_b
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
            cv2.putText(image, text, (xmax - 2, ymax - 2), font, 1, (250, 0, 255), 1)
    if args.dataset == 'VOC':
        filename = os.path.join(devkit_path, info[1] + '_VOC.jpg')
    elif args.dataset == 'COCO':
        filename = os.path.join(devkit_path, info + '_COCO.jpg')
    cv2.imwrite(filename, image)


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

if os.path.exists(args.save_folder) is None:
    os.mkdir(args.save_folder)

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

        dataset_test = CocoDetection(args.dataset_root, set_name='test2017',
                                     transform=transforms.Compose([Normalize(), Resize()]))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset VOC')
        elif args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on VOC')

        dataset_test = VocDetection(args.dataset_root, image_sets=[('2007', 'test')],
                                    transform=transforms.Compose([Normalize(), Resize()]))
    else:
        raise ValueError('Dataset type not understood (must be voc or coco), exiting.')

    if args.dataset == 'VOC':
        num_class = 20
    elif args.dataset == 'COCO':
        num_class = 80

    # 4. Define to train mode
    if args.basenet == 'ResNet':
        if args.depth == 18:
            model = resnet18_retinanet(num_classes=num_class,
                                       pretrained=args.pretrained,
                                       training=args.training)
        elif args.depth == 34:
            model = resnet34_retinanet(num_classes=num_class,
                                       pretrained=args.pretrained,
                                       training=args.training)
        elif args.depth == 50:
            model = resnet50_retinanet(num_classes=num_class,
                                       pretrained=args.pretrained,
                                       training=args.training)  # False means the models is not trained
        elif args.depth == 101:
            model = resnet101_retinanet(num_classes=num_class,
                                        pretrained=args.pretrained,
                                        training=args.training)
        elif args.depth == 152:
            model = resnet152_retinanet(num_classes=num_class,
                                        pretrained=args.pretrained,
                                        training=args.training)
        else:
            raise ValueError('Unsupported model depth!')

        print("Using model retinanet...")


    else:
        raise ValueError('Unsupported model type!')

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
            write_test_results(dataset_test, model)
        Visualized(dataset_test, model)
        t1 = time.time()
        m = (t1 - t0) // 60
        s = (t1 - t0) % 60
        print("It took a total of {}m{}s to complete the testing.".format(int(m), int(s)))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    logger.info("Program started")
    test()
    logger.info("Done!")
