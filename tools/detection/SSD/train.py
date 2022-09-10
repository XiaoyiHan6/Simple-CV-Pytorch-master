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
from tools.detection.SSD.eval.VOC.voc_eval import evaluate_voc
from utils.Sampler import Sampler
from tools.detection.SSD.eval.COCO.coco_eval import evaluate_coco
from models.detection.SSD import SSD
from utils.collate import ssd_collate
from torch.utils.data import DataLoader
from utils.get_logger import get_logger
from torch.cuda.amp import autocast, GradScaler
from utils.augmentations.SSDAugmentations import SSDAugmentation
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule

assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SSD Training')
    parser.add_argument('--dataset',
                        type=str,
                        default='VOC',
                        choices=['COCO', 'VOC'],
                        help='Dataset type, must be one of VOC or COCO.')
    parser.add_argument('--dataset_root',
                        type=str,
                        default=VOC_ROOT,
                        choices=[COCO_ROOT, VOC_ROOT],
                        help='Path to COCO or VOC directory')
    parser.add_argument('--model',
                        type=str,
                        default='ssd',
                        help='Training Model')
    parser.add_argument('--training',
                        type=str,
                        default=True,
                        help='Model is training or testing')
    parser.add_argument('--pretrained',
                        default='vgg16_reducedfc.pth',
                        type=str,
                        help='Pretrained base model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers',
                        type=int,
                        default=2,
                        help='Number of workers used in dataloading')
    parser.add_argument('--tensorboard',
                        type=str,
                        default=True,
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
                        default=3e-2,
                        help='initial learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=120,
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

        dataset_train = CocoDetection(args.dataset_root, set_name='train2017',
                                      transform=SSDAugmentation())
        dataset_val = CocoDetection(args.dataset_root, set_name='val2017')

    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset_root VOC')
        elif args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on VOC')

        dataset_train = VocDetection(args.dataset_root,
                                     transform=SSDAugmentation())
        dataset_val = VocDetection(args.dataset_root, image_sets=[('2007', 'test')])
    else:
        raise ValueError('Dataset type not understood (must be VOC or COCO), exiting.')

    sampler = Sampler(dataset_train, batch_size=args.batch_size, drop_last=True)

    dataloader_train = DataLoader(dataset_train, num_workers=args.num_workers, collate_fn=ssd_collate,
                                  batch_sampler=sampler)

    # 4. Create the model
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

    if args.resume:
        other, ext = os.path.splitext(args.resume)
        if ext == '.pkl' or '.pth':
            model_load = os.path.join(args.save_folder, args.resume)
            print("Resuming training, loading {}...".format(args.resume))
            model.load_state_dict(torch.load(model_load))
        else:
            raise ValueError("Sorry only .pth and .pkl files supported.")
    else:
        vgg_weights = os.path.join(args.save_folder, args.pretrained)
        print('Loading base betwork...')
        model.module.backbone.vgg.load_state_dict(torch.load(vgg_weights))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scaler = GradScaler(enabled=True)

    scheduler = WarmupCosineSchedule(optimizer, 5, 119)

    # scheduler = WarmupLinearSchedule(optimizer, 5, 119)
    # len(dataset_train): 16551, iter_size: 1034
    loss_hist = collections.deque(maxlen=1034)

    model.train()
    iter = 0

    iter_size = len(dataset_train) // args.batch_size
    print("len(dataset_train): {}, iter_size: {}".format(len(dataset_train), iter_size))
    logger.info(f"{args}")
    t0 = time.time()
    best_map = 0.0
    # 5. training
    for epoch_num in range(args.epochs):
        t1 = time.time()
        model.train()

        epoch_loss = []

        for data in dataloader_train:
            try:
                iter += 1
                optimizer.zero_grad()
                imgs, annots = data['img'], data['annot']
                if args.cuda:
                    if torch.cuda.is_available():
                        imgs = imgs.cuda().float()
                        annots = [ann.cuda() for ann in annots]
                else:
                    imgs = imgs.float()
                with autocast(enabled=True):
                    conf_loss, loc_loss = model([imgs, annots])

                conf_loss = conf_loss.mean()
                loc_loss = loc_loss.mean()

                loss = conf_loss + loc_loss

                if bool(loss == 0):
                    continue

                scaler.scale(loss).backward(retain_graph=True)

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
                        f"Classification loss: {float(conf_loss):1.3f} | "
                        f"Regression loss: {float(loc_loss):1.3f} | Loss: {loss:1.3f} | "
                        f"np.mean(loss_hist): {np.mean(loss_hist):1.3f} | np.mean(total_loss): {np.mean(epoch_loss):1.3f}")

                del conf_loss
                del loc_loss
            except Exception as e:
                print(e)
                continue
            # scheduler.step()

        scheduler.step()
        t2 = time.time()
        h_time = (t2 - t1) // 3600
        m_time = ((t2 - t1) % 3600) // 60
        s_time = ((t2 - t1) % 3600) % 60

        print(
            "epoch {} is finished, and the time is {}h{}m{}s".format(epoch_num, int(h_time), int(m_time), int(s_time)))
        if epoch_num > 20:
            print('Evaluation ssd...')
            t_eval_start = time.time()
            model.eval()
            model.training = False
            with torch.no_grad():
                if args.dataset == 'VOC':
                    aps, labelmap = evaluate_voc(dataset_val, model)
                    logger.info(f"Mean AP:{np.mean(aps):1.4f}")
                elif args.dataset == "COCO":
                    all_eval_result = evaluate_coco(dataset_val, model)
                    aps = all_eval_result[1]
                    logger.info(f"IoU=0.5, area=all, maxDets=100, mAP:{aps:1.4f}")

                else:
                    raise ValueError('Dataset type not understood (must be VOC or COCO), exiting.')

            if np.mean(aps) > best_map:
                print('Saving best mAP state, iter: ', epoch_num)
                torch.save(model.state_dict(),
                           args.save_folder + '/' + args.model + '_' +
                           args.dataset.lower() + '_' + 'best' + '.pth')
                best_map = np.mean(aps)
            t_eval_end = time.time()

            h_eval = (t_eval_end - t_eval_start) // 3600
            m_eval = ((t_eval_end - t_eval_start) % 3600) // 60
            s_eval = ((t_eval_end - t_eval_start) % 3600) % 60
            print(
                "Evaluation is finished, and the time is {}h{}m{}s".format(int(h_eval), int(m_eval), int(s_eval)))
            model.training = True
            model.train()

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
