import os
import sys
import time
import torch
import logging
import numpy as np
import collections
from utils.get_logger import get_logger
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from tools.detection.RetinaNet.eval_voc import eval_voc
from tools.detection.RetinaNet.eval_coco import eval_coco
from options.detection.RetinaNet.train_options import args, cfg, dataset_train, \
    dataset_val, dataloader_train, iter_size, retinanet, retinanet_eval

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

assert torch.__version__.split('.')[0] == '1'
print('RetinaNet train.py CUDA available: {}'.format(torch.cuda.is_available()))

# Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)

if __name__ == '__main__':
    logger.info("Program training started!")

    # Create SummaryWriter
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        # tensorboard loss
        writer = SummaryWriter(args.tensorboard_log)

    # Create the model
    print("Using model retinanet...")
    if args.cuda and torch.cuda.is_available():
        model = retinanet.cuda()

    if args.resume:
        other, ext = os.path.splitext(args.resume)
        if ext == '.pkl' or '.pth':
            model_load = os.path.join(args.save_folder, args.resume)
            model.load_state_dict(torch.load(model_load))
            print("Loading weights into state dict...")
        else:
            print("Sorry only .pth and .pkl files supported.")

    if args.cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(retinanet)

    model.training = True
    optimizer = get_optimizer(cfg=cfg, module=model)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)
    loss_hist = collections.deque(maxlen=500)

    print("len(dataset_train): {}, iter_size: {}".format(len(dataset_train), iter_size))
    logger.info(f"args - {args}")

    iter = 0
    best_map = 0.0
    t_start = time.time()
    # training
    for epoch_num in range(cfg['OPTIMIZE']['EPOCH']):
        t_epoch_start = time.time()
        model.train()
        model.module.freeze_bn()
        epoch_loss = []
        for data in dataloader_train:
            try:
                iter += 1
                optimizer.zero_grad()
                imgs, annots, scales = data['img'], data['annot'], data['scale']
                if args.cuda and torch.cuda.is_available():
                    imgs = imgs.cuda().float()
                    annots = annots.cuda()
                else:
                    imgs = imgs.float()

                con_loss, reg_loss = model([imgs, annots])

                con_loss = con_loss.mean()
                reg_loss = reg_loss.mean()

                loss = con_loss + reg_loss

                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                if args.tensorboard:
                    writer.add_scalar("total_loss", np.mean(epoch_loss), iter)

                if iter % 100 == 0:
                    logger.info(
                        f"Epoch: {str(epoch_num + 1)} | Iteration: {iter}  | lr: {optimizer.param_groups[0]['lr']} | "
                        f"Classification loss: {float(con_loss):1.5f} | "
                        f"Regression loss: {float(reg_loss):1.5f} | Loss: {loss:1.5f} | "
                        f"np.mean(loss_hist): {np.mean(loss_hist):1.5f} | np.mean(total_loss): {np.mean(epoch_loss):1.5f}")

                del con_loss
                del reg_loss
            except Exception as e:
                print(e)
                continue

        # ReduceLROnPlateau
        # scheduler.step(np.mean(epoch_loss))
        scheduler.step()
        t_epoch_end = time.time()
        h_epoch = (t_epoch_end - t_epoch_start) // 3600
        m_epoch = ((t_epoch_end - t_epoch_start) % 3600) // 60
        s_epoch = ((t_epoch_end - t_epoch_start) % 3600) % 60
        print(
            "epoch {} is finished, and the time is {}h{}m{}s".format(epoch_num + 1, int(h_epoch), int(m_epoch),
                                                                     int(s_epoch)))
        torch.save(model.module.state_dict(),
                   args.save_folder + '/' + cfg['MODEL']['NAME'].lower()
                   + '_' + cfg['MODEL']['BACKBONE']['NAME'].lower()
                   + str(cfg['MODEL']['BACKBONE']['DEPTH'])
                   + '_' + cfg['DATA']['NAME'].lower() + '.pth')

        if (epoch_num + 1) > 15:
            print('Evaluation RetinaNet...')
            t_eval_start = time.time()
            retinanet_eval.load_state_dict(
                torch.load(args.save_folder + '/' + cfg['MODEL']['NAME'].lower()
                           + '_' + cfg['MODEL']['BACKBONE']['NAME'].lower()
                           + str(cfg['MODEL']['BACKBONE']['DEPTH'])
                           + '_' + cfg['DATA']['NAME'].lower() + '.pth'))

            retinanet_eval.eval()
            if args.cuda:
                retinanet_eval = retinanet_eval.cuda()
            with torch.no_grad():
                maps = 0.0
                if cfg['DATA']['NAME'] == 'VOC':
                    print("waiting eval VOC, model RetinaNet...")
                    aps = eval_voc(dataset_val, retinanet_eval)
                    maps = np.mean(aps)
                elif cfg['DATA']['NAME'] == 'COCO':
                    print("waiting eval COCO, model RetinaNet...")
                    aps = eval_coco(dataset_val, retinanet_eval)
                    maps = aps[1]
            logger.info(f"IoU=0.5, Mean AP = {maps:.3f}")
            if maps > best_map:
                print("Saving best mAP state, epoch: {} | iter: {}".format(str(epoch_num + 1), iter))
                torch.save(model.module.state_dict(), args.save_folder + '/' + cfg['MODEL']['NAME'].lower()
                           + '_' + cfg['MODEL']['BACKBONE']['NAME'].lower()
                           + str(cfg['MODEL']['BACKBONE']['DEPTH'])
                           + '_' + cfg['DATA']['NAME'].lower() + '_best.pth')
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
    print("The Program Finished Time is {}h{}m{}s".format(int(h), int(m), int(s)))

    logger.info("Done!")
