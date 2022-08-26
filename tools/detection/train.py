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
import collections
import torch.optim as optim
from utils.Sampler import Sampler
from utils.collate import collate
from torchvision import transforms
from models.detection.SSD import SSD
from torch.utils.data import DataLoader
from utils.get_logger import get_logger
from torch.cuda.amp import autocast, GradScaler
from models.detection.RetinaNet import resnet18_retinanet, resnet34_retinanet, \
    resnet50_retinanet, resnet101_retinanet, resnet152_retinanet
from utils.augmentations import RetinaNetResize, RandomFlip, Normalize, SSDResize, SSDRandSampleCrop, \
    SSDToPercentCoords, SSDToAbsoluteCoords, SSDExpand

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch detection Training')
    parser.add_argument('--dataset',
                        type=str,
                        default='COCO',
                        choices=['COCO', 'VOC'],
                        help='Dataset type, must be one of VOC or COCO.')
    parser.add_argument('--dataset_root',
                        type=str,
                        default=COCO_ROOT,
                        choices=[COCO_ROOT, VOC_ROOT],
                        help='Path to COCO or VOC directory')
    parser.add_argument('--model',
                        type=str,
                        default='ssd',
                        choices=['retinanet', 'ssd'],
                        help='Training Model')
    parser.add_argument('--depth',
                        type=int,
                        default=0,
                        help='Model depth, including RetinaNet of 18, 34, 50, 101, 152, SSD of 0')
    parser.add_argument('--training',
                        type=str,
                        default=True,
                        help='Model is training or testing')
    parser.add_argument('--pretrained',
                        default=True,
                        type=str,
                        help='Models was pretrained')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers',
                        type=int,
                        default=2,
                        help='Number of workers user in dataloading')
    parser.add_argument('--tensorboard',
                        type=str,
                        default=False,
                        help='Use tensorboard for loss visualization')
    parser.add_argument('--tensorboard_log',
                        type=str,
                        default=config.tensorboard_log,
                        help='Use tensorboard for loss visualization')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to train model')
    parser.add_argument('--log_folder',
                        type=str,
                        default=config.log)
    parser.add_argument('--log_name',
                        type=str,
                        default=config.detection_train_log)
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--save_folder',
                        type=str,
                        default=config.checkpoint_path,
                        help='Directory for saving checkpoint models')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=4,
                        help='Number of epochs')

    return parser.parse_args()


args = parse_args()

# 1. Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)


def train():
    # 2. Create SummaryWriter
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # tensorboard loss
        writer = SummaryWriter(args.tensorboard_log)

    # 3. Create the data loaders
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset COCO')
        elif args.dataset_root is None:
            raise ValueError("WARNING: Using default COCO dataset, but " +
                             "--dataset_root was not specified.")
        if args.model == 'retinanet':
            dataset_train = CocoDetection(args.dataset_root, set_name='train2017',
                                          transform=transforms.Compose([
                                              Normalize(),
                                              RandomFlip(),
                                              RetinaNetResize()]))
        elif args.model == 'ssd':
            dataset_train = CocoDetection(args.dataset_root, set_name='train2017',
                                          transform=transforms.Compose([
                                              SSDToAbsoluteCoords(),
                                              SSDExpand(),
                                              SSDRandSampleCrop(),
                                              RandomFlip(),
                                              SSDToPercentCoords(),
                                              SSDResize(),
                                          ]))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset VOC')
        elif args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on VOC')
        if args.model == 'retinanet':
            dataset_train = VocDetection(args.dataset_root,
                                         transform=transforms.Compose([
                                             Normalize(),
                                             RandomFlip(),
                                             RetinaNetResize()]))
        elif args.model == 'ssd':
            dataset_train = VocDetection(args.dataset_root,
                                         transform=transforms.Compose([
                                             SSDToAbsoluteCoords(),
                                             SSDExpand(),
                                             SSDRandSampleCrop(),
                                             RandomFlip(),
                                             SSDToPercentCoords(),
                                             SSDResize(),
                                         ]))
    else:
        raise ValueError('Dataset type not understood (must be voc or coco), exiting.')

    sampler = Sampler(dataset_train, batch_size=args.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=args.num_workers, collate_fn=collate,
                                  batch_sampler=sampler)

    # 4. Create the model
    if args.model == 'retinanet':
        if args.depth == 18:
            model = resnet18_retinanet(num_classes=dataset_train.num_classes(),
                                       pretrained=args.pretrained,
                                       training=args.training)
        elif args.depth == 34:
            model = resnet34_retinanet(num_classes=dataset_train.num_classes(),
                                       pretrained=args.pretrained,
                                       training=args.training)
        elif args.depth == 50:
            model = resnet50_retinanet(num_classes=dataset_train.num_classes(),
                                       pretrained=args.pretrained,
                                       training=args.training)
        elif args.depth == 101:
            model = resnet101_retinanet(num_classes=dataset_train.num_classes(),
                                        pretrained=args.pretrained,
                                        training=args.training)
        elif args.depth == 152:
            model = resnet152_retinanet(num_classes=dataset_train.num_classes(),
                                        pretrained=args.pretrained,
                                        training=args.training)
        else:
            raise ValueError("Unsupported RetinaNet Model depth!")

        print("Using model retinanet...")
    elif args.model == 'ssd':
        if args.depth == 0:
            model = SSD(version=args.dataset,
                        training=args.training,
                        batch_norm=False)
        else:
            raise ValueError("Unsupported SSD Model depth!")
        print("Using model ssd...")

    else:
        raise ValueError('Unsupported model type!')

    if args.cuda:
        if torch.cuda.is_available():
            model = model.cuda()
            model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    if args.resume:
        other, ext = os.path.splitext(args.resume)
        if ext == '.pkl' or '.pth':
            model_load = os.path.join(args.save_folder, args.resume)
            model.load_state_dict(torch.load(model_load))
            print("Loading weights into state dict...")
        else:
            print("Sorry only .pth and .pkl files supported.")

    model.training = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler(enabled=True)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3], gamma=0.1)

    loss_hist = collections.deque(maxlen=500)

    model.train()
    model.module.freeze_bn()
    iter = 0

    iter_size = len(dataset_train) // args.batch_size
    print("len(dataset_train): {}, iter_size: {}".format(len(dataset_train), iter_size))
    logger.info(f"{args}")
    t0 = time.time()
    # 5. training
    for epoch_num in range(args.epochs):
        t1 = time.time()
        model.train()
        model.module.freeze_bn()

        epoch_loss = []

        for data in dataloader_train:
            try:
                iter += 1
                optimizer.zero_grad()
                imgs, annots, scales = data['img'], data['annot'], data['scale']
                if args.cuda:
                    if torch.cuda.is_available():
                        imgs = imgs.cuda().float()
                        annots = annots.cuda()
                else:
                    imgs = imgs.float()
                with autocast(enabled=True):
                    con_loss, loc_loss = model([imgs, annots])

                    con_loss = con_loss.mean()
                    loc_loss = loc_loss.mean()

                    loss = con_loss + loc_loss

                if bool(loss == 0):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                scaler.step(optimizer)

                scaler.update()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                if args.tensorboard:
                    writer.add_scalar("total_loss", np.mean(epoch_loss), iter)

                if iter % 100 == 0:
                    logger.info(
                        f"Epoch: {epoch_num} | Iteration: {iter}  | lr: {optimizer.param_groups[0]['lr']} | "
                        f"Classification loss: {float(con_loss):1.5f} | "
                        f"Regression loss: {float(loc_loss):1.5f} | Loss: {loss:1.5f} | "
                        f"np.mean(loss_hist): {np.mean(loss_hist):1.5f} | np.mean(total_loss): {np.mean(epoch_loss):1.5f}")

                del con_loss
                del loc_loss
            except Exception as e:
                print(e)
                continue

        # scheduler.step(np.mean(epoch_loss))
        scheduler.step()
        t2 = time.time()
        h_time = (t2 - t1) // 3600
        m_time = ((t2 - t1) % 3600) // 60
        s_time = ((t2 - t1) % 3600) % 60

        print(
            "epoch {} is finished, and the time is {}h{}m{}s".format(epoch_num, int(h_time), int(m_time), int(s_time)))

        if epoch_num % 1 == 0:
            print('Saving state, iter: ', epoch_num)
            torch.save(model.state_dict(),
                       args.save_folder + '/' + args.dataset +
                       '_' + args.model + str(args.depth) +
                       '_' + repr(epoch_num) + '.pth')

    torch.save(model.state_dict(), args.save_folder + '/' +
               args.dataset + '_' + args.model +
               str(args.depth) + '.pth')

    if args.tensorboard:
        writer.close()

    t3 = time.time()
    h = (t3 - t0) // 3600
    m = ((t3 - t1) % 3600) // 60
    s = ((t3 - t1) % 3600) % 60
    print("The Finished Time is {}h{}m{}s".format(int(h), int(m), int(s)))

    return


if __name__ == '__main__':
    logger.info("Program training started!")
    train()
    logger.info("Done!")
