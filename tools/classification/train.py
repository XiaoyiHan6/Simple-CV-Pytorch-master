import os
import logging
import argparse
import warnings

warnings.filterwarnings('ignore')

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import time
import torch
from data import *
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torchvision import transforms
from utils.accuracy import accuracy
from torch.utils.data import DataLoader
from utils.get_logger import get_logger
from models.classification.lenet5 import lenet5
from models.classification.alexnet import alexnet
from utils.AverageMeter import AverageMeter
from torch.cuda.amp import autocast, GradScaler
from models.classification.mobilenet_v3 import MobileNet_v3
from models.classification.googlenet import GoogLeNet
from models.classification.vgg import vgg11, vgg13, vgg16, vgg19
from models.classification.mobilenet_v2 import MobileNet_v2
from models.classification.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, \
    resnext101_32x8d
from models.classification.shufflenet import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, \
    shufflenet_v2_x2_0


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch classification Training')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset',
                        type=str,
                        default='ImageNet',
                        choices=['ImageNet', 'CIFAR'],
                        help='ImageNet, CIFAR')
    parser.add_argument('--dataset_root',
                        type=str,
                        default=ImageNet_Train_ROOT,
                        choices=[ImageNet_Train_ROOT, CIFAR_ROOT],
                        help='Dataset root directory path')
    parser.add_argument('--basenet',
                        type=str,
                        default='shufflenet',
                        choices=['resnet', 'vgg', 'lenet', 'alexnet', 'googlenet', 'mobilenet', 'resnext',
                                 'shufflenet'],
                        help='Pretrained base model')
    parser.add_argument('--depth',
                        type=int,
                        default=5,
                        help='BaseNet depth, including: LeNet of 5, AlexNet of 0, VGG of 11, 13, 16, 19, ResNet of 18, 34, 50, 101, 152, ResNeXt of 50, 101, GoogLeNet of 0, MobileNet of 2, 3, ShuffleNet of 5, 10, 15, 20')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size for training')
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of workers user in dataloading')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to train model')
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=1,
                        help='Gradient acumulation steps')
    parser.add_argument('--save_folder',
                        type=str,
                        default=config.checkpoint_path,
                        help='Directory for saving checkpoint models')
    parser.add_argument('--tensorboard',
                        type=str,
                        default=False,
                        help='Use tensorboard for loss visualization')
    parser.add_argument('--log_folder',
                        type=str,
                        default=config.log,
                        help='Log Folder')
    parser.add_argument('--log_name',
                        type=str,
                        default=config.classification_train_log,
                        help='Log Name')
    parser.add_argument('--tensorboard_log',
                        type=str,
                        default=config.tensorboard_log,
                        help='Use tensorboard for loss visualization')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=30,
                        help='Number of epochs')
    parser.add_argument('--num_classes',
                        type=int,
                        default=1000,
                        help='the number classes, like ImageNet:1000, cifar:10')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='image size, like ImageNet:224, cifar:32')
    parser.add_argument('--pretrained',
                        type=str,
                        default=True,
                        help='Models was pretrained')
    parser.add_argument('--init_weights',
                        type=str,
                        default=False,
                        help='Init Weights')
    parser.add_argument('--patience',
                        type=int,
                        default=2,
                        help='patience of ReduceLROnPlateau')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='Momentum value for optim')

    return parser.parse_args()


args = parse_args()

# 1. Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)

# 2. Torch choose cuda or cpu
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but you aren't using it" +
              "\n You can set the parameter of cuda to True.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    # 3. Create SummaryWriter
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # tensorboard  loss
        writer = SummaryWriter(args.tensorboard_log)
    # vgg16, alexnet and lenet5 need to resize image_size, because of fc.
    if args.basenet == 'vgg' or args.basenet == 'alexnet' or args.basenet == 'googlenet':
        args.image_size = 224
    elif args.basenet == 'lenet':
        args.image_size = 32

    # 4. Ready dataset
    if args.dataset == 'ImageNet':
        if args.dataset_root == CIFAR_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset ImageNet2012.')

        elif os.path.exists(ImageNet_Train_ROOT) is None:
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

    elif args.dataset == 'CIFAR':
        if args.dataset_root == ImageNet_Train_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset CIFAR10.')

        elif args.dataset_root is None:
            raise ValueError("Must provide --dataset_root when training on CIFAR10.")

        dataset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=True,
                                               transform=torchvision.transforms.Compose([
                                                   transforms.Resize((args.image_size,
                                                                      args.image_size)),
                                                   torchvision.transforms.ToTensor()]))
    else:
        raise ValueError('Dataset type not understood (must be ImageNet or CIFAR), exiting.')

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers,
                                             pin_memory=False, generator=torch.Generator(device='cuda'))

    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    # 5. Define train model

    # Unfortunately, LeNet5 and AlexNet don't provide pretrianed Model.
    if args.basenet == 'lenet':
        if args.depth == 5:
            model = lenet5(num_classes=args.num_classes,
                           init_weights=args.init_weights)
        else:
            raise ValueError('Unsupported LeNet depth!')

    elif args.basenet == 'alexnet':
        if args.depth == 0:
            model = alexnet(num_classes=args.num_classes,
                            init_weights=args.init_weights)
        else:
            raise ValueError('Unsupported AlexNet depth!')

    elif args.basenet == 'googlenet':
        if args.depth == 0:
            model = GoogLeNet(num_classes=args.num_classes,
                              pretrained=args.pretrained,
                              aux_logits=True,
                              init_weights=args.init_weights)
        else:
            raise ValueError('Unsupported GoogLeNet depth!')

    elif args.basenet == 'vgg':
        if args.depth == 11:
            model = vgg11(pretrained=args.pretrained,
                          num_classes=args.num_classes,
                          init_weights=args.init_weights)
        elif args.depth == 13:
            model = vgg13(pretrained=args.pretrained,
                          num_classes=args.num_classes,
                          init_weights=args.init_weights)
        elif args.depth == 16:
            model = vgg16(pretrained=args.pretrained,
                          num_classes=args.num_classes,
                          init_weights=args.init_weights)
        elif args.depth == 19:
            model = vgg19(pretrained=args.pretrained,
                          num_classes=args.num_classes,
                          init_weights=args.init_weights)
        else:
            raise ValueError('Unsupported VGG depth!')

    # Unfortunately for my resnet, there is no set init_weight, because I'm going to set object detection algorithm
    elif args.basenet == 'resnet':
        if args.depth == 18:
            model = resnet18(pretrained=args.pretrained,
                             num_classes=args.num_classes)
        elif args.depth == 34:
            model = resnet34(pretrained=args.pretrained,
                             num_classes=args.num_classes)
        elif args.depth == 50:
            model = resnet50(pretrained=args.pretrained,
                             num_classes=args.num_classes)  # False means the models was not trained
        elif args.depth == 101:
            model = resnet101(pretrained=args.pretrained,
                              num_classes=args.num_classes)
        elif args.depth == 152:
            model = resnet152(pretrained=args.pretrained,
                              num_classes=args.num_classes)
        else:
            raise ValueError('Unsupported ResNet depth!')

    elif args.basenet == 'resnext':
        if args.depth == 50:
            model = resnext50_32x4d(pretrained=args.pretrained,
                                    num_classes=args.num_classes)
        elif args.depth == 101:
            model = resnext101_32x8d(pretrained=args.pretrained,
                                     num_classes=args.num_classes)
        else:
            raise ValueError('Unsupported ResNeXt depth!')

    elif args.basenet == 'mobilenet':
        if args.depth == 2:
            model = MobileNet_v2(pretrained=args.pretrained,
                                 num_classes=args.num_classes,
                                 init_weights=args.init_weights)
        elif args.depth == 3:
            model = MobileNet_v3(pretrained=args.pretrained,
                                 num_classes=args.num_classes,
                                 init_weights=args.init_weights,
                                 type='small')
        else:
            raise ValueError('Unsupported MobileNet depth!')
    elif args.basenet == 'shufflenet':
        if args.depth == 5:
            model = shufflenet_v2_x0_5(pretrained=args.pretrained, num_classes=args.num_classes)
        elif args.depth == 10:
            model = shufflenet_v2_x1_0(pretrianed=args.pretrianed, num_classes=args.num_classes)
        elif args.depth == 15:
            model = shufflenet_v2_x1_5(pretrained=args.pretrained, num_classes=args.num_classes)
        elif args.depth == 20:
            model = shufflenet_v2_x2_0(pretrained=args.pretrained, num_classes=args.num_classes)
        else:
            raise ValueError('Unsupported ShuffleNet depth!')
    else:
        raise ValueError('Unsupported model type!')

    if args.cuda:
        if torch.cuda.is_available():
            model = model.cuda()
            model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    # 6. Loading weights
    if args.resume:
        other, ext = os.path.splitext(args.resume)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            model_load = os.path.join(args.save_folder, args.resume)
            model.load_state_dict(torch.load(model_load))
        else:
            print('Sorry only .pth and .pkl files supported.')
    if args.init_weights:
        # initialize newly added models' weights with xavier method
        if args.basenet == 'resnet' or args.basenet == 'shufflenet':
            print("There is no set init_weight, because I'm going to set object detection algorithm.")
        else:
            print("Initializing weights...")
    else:
        print("Not Initializing weights...")
    if args.pretrained:
        if args.basenet == 'lenet' or args.basenet == 'alexnet':
            print("There is no available pretrained model on the website. ")
        else:
            print("Models was pretrained...")
    else:
        print("Pretrained models is False...")

    model.train()

    iteration = 0

    # 7. Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.patience, verbose=True)
    scaler = GradScaler()

    # 8. Length
    iter_size = len(dataset) // args.batch_size
    print("len(dataset): {}, iter_size: {}".format(len(dataset), iter_size))
    logger.info(f"args - {args}")
    t0 = time.time()

    # 9. Create batch iterator
    for epoch in range(args.epochs):
        t1 = time.time()
        model.training = True
        torch.cuda.empty_cache()
        # 10. Load train data
        for data in dataloader:
            iteration += 1
            images, targets = data
            # 11. Backward
            optimizer.zero_grad()
            if args.cuda:
                images, targets = images.cuda(), targets.cuda()
                criterion = criterion.cuda()
            # 12. Forward
            with autocast():
                if args.basenet == 'googlenet':
                    outputs, aux2_output, aux1_output = model(images)
                    loss1 = criterion(outputs, targets)
                    loss_aux2 = criterion(aux2_output, targets)
                    loss_aux1 = criterion(aux1_output, targets)
                    loss = loss1 + loss_aux2 * 0.3 + loss_aux1 * 0.3
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    loss = loss / args.accumulation_steps

            if args.tensorboard:
                writer.add_scalar("train_classification_loss", loss.item(), iteration)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 13. Measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            losses.update(loss.item(), images.size(0))

            if iteration % 100 == 0:
                logger.info(
                    f"- epoch: {epoch},  iteration: {iteration}, lr: {optimizer.param_groups[0]['lr']}, "
                    f"top1 acc: {acc1.item():.2f}%, top5 acc: {acc5.item():.2f}%, "
                    f"loss: {loss.item():.3f}, (losses.avg): {losses.avg:3f} "
                )

        scheduler.step(losses.avg)

        t2 = time.time()
        h_time = (t2 - t1) // 3600
        m_time = ((t2 - t1) % 3600) // 60
        s_time = ((t2 - t1) % 3600) % 60
        print("epoch {} is finished, and the time is {}h{}min{}s".format(epoch, int(h_time), int(m_time), int(s_time)))

        # 14. Save train model
        if epoch != 0 and epoch % 10 == 0:
            print('Saving state, iter:', epoch)
            torch.save(model.state_dict(),
                       args.save_folder + '/' + args.dataset +
                       '_' + args.basenet + str(args.depth) + '_' + repr(epoch) + '.pth')
        torch.save(model.state_dict(),
                   args.save_folder + '/' + args.dataset + "_" + args.basenet + str(args.depth) + '.pth')

    if args.tensorboard:
        writer.close()

    t3 = time.time()
    h = (t3 - t0) // 3600
    m = ((t3 - t0) % 3600) // 60
    s = ((t3 - t0) % 3600) % 60
    print("The Finished Time is {}h{}m{}s".format(int(h), int(m), int(s)))
    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    logger.info("Program started")
    top1, top5, loss = train()
    print("top1 acc: {}, top5 acc: {}, loss:{}".format(top1, top5, loss))
    logger.info("Done!")
