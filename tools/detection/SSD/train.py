import os
import time
import torch
import logging
import numpy as np
import torch.nn as nn
from utils.path import MEANS
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from utils.get_logger import get_logger
from utils.scheduler import get_scheduler
from utils.optimizer import get_optimizer
from models.detection.SSD.ssd import build_ssd
from tools.detection.SSD.eval_voc import eval_voc
from tools.detection.SSD.eval_coco import eval_coco
from models.detection.SSD.box_head.loss import MultiBoxLoss
from models.detection.SSD.utils.augmentations import BaseTransform
from options.detection.SSD.train_options import args, cfg, dataloader_train, dataset_train, dataset_val, iter_size

assert torch.__version__.split('.')[0] == '1'
print('SSD train.py CUDA available: {}'.format(torch.cuda.is_available()))

get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # tensorboard loss
        writer = SummaryWriter(args.tensorboard_log)

    ssd_net = build_ssd(phase='train', cfg=cfg)
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
        net = net.cuda()

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    elif args.pretrained:
        vgg_weights = torch.load(args.save_folder + "/" + args.pretrained)
        print('Loading pretrained network {}...'.format(args.pretrained))
        ssd_net.vgg.load_state_dict(vgg_weights)
    else:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = get_optimizer(cfg=cfg, module=net)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)

    criterion = MultiBoxLoss(num_classes=cfg['DATA']['NUM_CLASSES'],
                             overlap_thresh=cfg['TRAIN']['MATCH_THRESH'],
                             neg_pos=cfg['TRAIN']['NEG_POS'],
                             neg_overlap=cfg['TRAIN']['NEG_THRESH'],
                             use_gpu=args.cuda)

    net.train()
    iter = 0
    # loss counters
    loc_loss = 0
    conf_loss = 0

    print('Loading the length of train dataset: {}, iter_size: {}.'.format(len(dataset_train), iter_size))

    logger.info(f"args - {args}")

    step_index = 0
    best_map = 0.0

    t_start = time.time()
    # create batch iterator
    for epoch_num in range(cfg['OPTIMIZE']['EPOCH']):
        t_epoch_start = time.time()
        for data in dataloader_train:
            if iter in cfg['OPTIMIZE']['LR_STEP']:
                step_index += 1
                # if use adjust_learning_rate, code will use XX.forward()
                # if use the others scheduler, code will use XX.step()
                scheduler.forward(optimizer, step_index)

            # load train data
            try:
                iter += 1
                optimizer.zero_grad()
                images, targets = data
                if args.cuda:
                    images = images.cuda()
                    targets = [ann.cuda() for ann in targets]
                else:
                    targets = [ann for ann in targets]
                # forward
                out = net(images)
                # backprop
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()
                if args.tensorboard:
                    writer.add_scalar("loss", loss.item(), iter)

                if iter % 100 == 0:
                    logger.info(f"Epoch: {str(epoch_num + 1)} | Iter: {iter} | "
                                f"lr: {optimizer.param_groups[0]['lr']} | "
                                f"Classification loss {float(loss_c.item()):1.3f} | "
                                f"Regression loss {float(loss_l.item()):1.3f} | "
                                f"Loss: {loss.item():1.3f}.")

                del loss_l
                del loss_c
            except Exception as e:
                print(e)
                continue
        # scheduler.step()
        t_epoch_end = time.time()
        h_epoch = (t_epoch_end - t_epoch_start) // 3600
        m_epoch = ((t_epoch_end - t_epoch_start) % 3600) // 60
        s_epoch = ((t_epoch_end - t_epoch_start) % 3600) % 60
        print("epoch {} is finished, and the time is {}h{}m{}s.".format(str(epoch_num + 1), int(h_epoch), int(m_epoch),
                                                                        int(s_epoch)))
        torch.save(ssd_net.state_dict(),
                   args.save_folder + '/' +
                   cfg['MODEL']['NAME'].lower() + "_" +
                   cfg['DATA']['NAME'].lower() + '.pth')

        if (epoch_num + 1) > 30:
            print('Evaluation ssd...')
            t_eval_start = time.time()
            net_eval = build_ssd(phase='test', cfg=cfg)
            net_eval.load_state_dict(
                torch.load(args.save_folder + "/" +
                           cfg['MODEL']['NAME'].lower() + "_" +
                           cfg['DATA']['NAME'].lower() + '.pth'))
            net_eval.eval()
            if args.cuda:
                net_eval = net_eval.cuda()
                cudnn.benchmark = True
            with torch.no_grad():
                if cfg['DATA']['NAME'] == 'VOC':
                    aps = eval_voc(dataset_val, net_eval, BaseTransform(cfg['DATA']['SIZE'], MEANS), args.cuda)
                    maps = np.mean(aps)
                elif cfg['DATA']['NAME'] == 'COCO':
                    aps = eval_coco(dataset_val, net_eval, BaseTransform(cfg['DATA']['SIZE'], MEANS), args.cuda)
                    maps = aps[1]
            logger.info(f"IoU=0.5, Mean AP = {maps:.3f}")
            if maps > best_map:
                print("Saving best mAP state, epoch: {} | iter: {}".format(str(epoch_num + 1), iter))
                torch.save(ssd_net.state_dict(), args.save_folder + '/' +
                           cfg['MODEL']['NAME'].lower() + "_" +
                           cfg['DATA']['NAME'].lower() + "_best.pth")
                best_map = maps
            t_eval_end = time.time()
            h_eval = (t_eval_end - t_eval_start) // 3600
            m_eval = ((t_eval_end - t_eval_start) % 3600) // 60
            s_eval = ((t_eval_end - t_eval_start) % 3600) % 60
            print(
                "Evaluation is finished, and the time is {}h{}m{}s.".format(int(h_eval), int(m_eval), int(s_eval)))

    if args.tensorboard:
        writer.close()
    t_end = time.time()
    h = (t_end - t_start) // 3600
    m = ((t_end - t_start) % 3600) // 60
    s = ((t_end - t_start) % 3600) % 60
    print("The Program Finished Time is {}h{}m{}s.".format(int(h), int(m), int(s)))
    return


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    logger.info("Program training started!")
    train()
    logger.info("Done!")
