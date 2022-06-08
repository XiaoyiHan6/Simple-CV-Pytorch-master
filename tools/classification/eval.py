import logging
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from data import *
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch
import torch.nn.parallel
import time
from utils.get_logger import get_logger
from utils.AverageMeter import AverageMeter
from utils.accuracy import accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Detection Evaluation')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset',
                        type=str,
                        default='ImageNet2012',
                        choices=['ImageNet2012', 'ImageNetmini', 'CIFAR10', 'CIFAR100'],
                        help='ImageNet2012, ImageNetmini, CIFAR10, CIFAR100')
    parser.add_argument('--dataset_root',
                        type=str,
                        default=ImageNet2012_Eval_ROOT,
                        help='Dataset root directory path')
    parser.add_argument('--basenet',
                        type=str,
                        default='resnet',
                        help='Pretrained base model')
    parser.add_argument('--depth',
                        type=int,
                        default=50,
                        help='Backbone depth, including ResNet of 18, 34, 50, 101, 152')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size for training')
    parser.add_argument('--evaluate',
                        type=str,
                        default=config.classification_evaluate,
                        help='Checkpoint state_dict file to evaluate training from')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of workers user in dataloading')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to eval model')
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
                        default=config.classification_eval_log,
                        help='Log Name')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='image size, like ImageNet:224, cifar:32')
    parser.add_argument('--pretrained',
                        type=str,
                        default=False,
                        help='Models was pretrained')

    return parser.parse_args()


args = parse_args()

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


def eval():
    # 3. Ready dataset
    if args.dataset == 'ImageNet2012':
        if args.dataset_root == ImageNetmini_Eval_ROOT or args.dataset_root == CIFAR_ROOT:
            raise ValueError("Must specify dataset_root if specifying dataset ImageNet2012")
        elif os.path.exists(ImageNet2012_Eval_ROOT) is None:
            raise ValueError("WARNING: Using default ImageNet2012 dataset_root because " +
                             "--dataset_root was not specified.")

        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_root,
            transform=torchvision.transforms.Compose([
                transforms.Resize((args.image_size,
                                   args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))

    elif args.dataset == 'ImageNetmini':
        if args.dataset_root == ImageNet2012_Eval_ROOT or args.dataset_root == CIFAR_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset ImageNetmini')
        elif args.dataset_root is None:
            raise ValueError("Must provide --dataset_root when on ImageNetmini")

        # Need to modify transform
        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset_root,
            transform=torchvision.transforms.Compose([
                transforms.Resize((args.image_size,
                                   args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))

    elif args.dataset == 'CIFAR10':
        if args.dataset_root == ImageNet2012_Eval_ROOT or args.dataset_root == ImageNetmini_Eval_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset CIFAR')
        elif args.dataset_root is None:
            raise ValueError("Must provide --dataset_root when training on CIFAR")

        dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_root, train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()]))

    elif args.dataset == 'CIFAR100':
        if args.dataset_root == ImageNet2012_Eval_ROOT or args.dataset_root == ImageNetmini_Eval_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset CIFAR')
        elif args.dataset_root is None:
            raise ValueError("Must provide --dataset_root when training on CIFAR")

        dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_root, train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()]))

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers,
                                             pin_memory=False, generator=torch.Generator(device='cuda'))

    top1 = AverageMeter()
    top5 = AverageMeter()

    # 4. Define to mode
    if args.basenet == 'resnet':
        if args.depth == 18:
            model = torchvision.models.resnet18(pretrained=args.pretrained)
        elif args.depth == 34:
            model = torchvision.models.resnet34(pretrained=args.pretrained)
        elif args.depth == 50:
            model = torchvision.models.resnet50(pretrained=args.pretrained)  # False means the models is not trained
        elif args.depth == 101:
            model = torchvision.models.resnet101(pretrained=args.pretrained)
        elif args.depth == 152:
            model = torchvision.models.resnet152(pretrained=args.pretrained)
        else:
            raise ValueError('Unsupported model depth!')
    else:
        raise ValueError('Unsupported model type!')

    if args.cuda:
        if torch.cuda.is_available():
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
        raise ValueError("Sorry, you should load weights! ")

    model.eval()

    # 6. Length
    iter_size = len(dataset) // args.batch_size
    print("len(dataset): {}, iter_size: {}".format(len(dataset), iter_size))
    logger.info(f"args - {args}")
    t0 = time.time()
    iter = 0

    # 7. Test
    with torch.no_grad():
        # 8. Load test data
        for data in dataloader:
            iter += 1
            images, targets = data
            if args.cuda:
                images, targets = images.cuda(), targets.cuda()

            # 9. Forward
            outputs = model(images)

            # 10. measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            logger.info(
                f"iter: {iter}, top1 acc: {acc1.item():.2f}%, top5 acc: {acc5.item():.2f}%. ")

        t1 = time.time()
        m = (t1 - t0) // 60
        s = (t1 - t0) % 60
        print("It took a total of {}m{}s to complete the evaluating.".format(int(m), int(s)))
    return top1.avg, top5.avg


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    logger.info("Program started")
    top1, top5 = eval()
    print("top1 acc: {}, top5 acc: {}".format(top1, top5))
    logger.info("Done!")
