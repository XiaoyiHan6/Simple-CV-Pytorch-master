from __future__ import print_function
import os
import cv2
import time
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from utils.path import MEANS, COLORS
from utils.get_logger import get_logger
from models.detection.SSD.ssd import build_ssd
from data.detection.SSD.voc0712 import VOC_CLASSES as labelmap
from models.detection.SSD.utils.augmentations import BaseTransform
from options.detection.SSD.test_options import args, cfg, dataset_test

devkit_path = os.path.join(args.Results, 'SSD')

assert torch.__version__.split('.')[0] == '1'
print('SSD visualize.py CUDA available: {}'.format(torch.cuda.is_available()))

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# 1. Mkdir checkpoints
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# 2. Mkdir results
if not os.path.exists(devkit_path):
    os.mkdir(devkit_path)

# 3. Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)


def write_eval_test_results(dataset, model, transform):
    filename = os.path.join(devkit_path, 'VOC/visualize.txt')
    if os.path.exists(filename):
        os.remove(filename)

    # Forward
    with tqdm(total=len(dataset)) as pbar:
        for i in range(len(dataset)):
            # h,w,c
            img = dataset.pull_image(i)
            # scale each detection back up to the image
            scale = torch.Tensor([img.shape[1], img.shape[0],
                                  img.shape[1], img.shape[0]])
            img_id, annotation = dataset.pull_anno(i)
            img = torch.from_numpy(transform(img)[0]).permute(2, 0, 1).unsqueeze(dim=0)
            with open(filename, mode='a') as f:
                f.write('\nGROUND TRUTH FOR: ' + img_id + '\n')
                for annot in annotation:
                    f.write('label: ' + labelmap[int(annot[4])] + ' ' + str(annot[4]) +
                            ' || coords: ' + ' || '.join(str(i) for i in annot[0:4]) + '\n')
            if args.cuda:
                img = img.cuda()

            outputs = model(img)
            detections = outputs.data
            pred_num = 0
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= args.visual_threshold:
                    if pred_num == 0:
                        with open(filename, mode='a') as f:
                            f.write('PREDICTIONS: ' + '\n')
                    score = detections[0, i, j, 0].cpu().numpy()
                    label_name = labelmap[i - 1]
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    pred_num += 1
                    with open(filename, mode='a') as f:
                        f.write(str(pred_num) + ' || label: ' + str(label_name) + ' ' + str(i - 1) + ' || ' +
                                'score: ' + str(score) + ' || coord: ' + ' || '.join(
                            str(c) for c in coords) + '\n')
                    j += 1
            pbar.update(1)
    pbar.close()
    return


def Visualized(dataset, model, transform):
    # shuffle
    rand = random.randint(1, 4952)
    img = dataset.pull_image(rand)
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    img_id = dataset.ids[rand]
    annotation = dataset.pull_item(rand)[1]
    for annot in annotation:
        xmin = int(annot[0] * img.shape[1])
        ymin = int(annot[1] * img.shape[0])
        xmax = int(annot[2] * img.shape[1])
        ymax = int(annot[3] * img.shape[0])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), COLORS[0], 2)
        label = str(int(annot[4]))
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, label, (xmin, int(ymin + 10)), font, 1, COLORS[1], 1)

    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1).unsqueeze(dim=0)
    if args.cuda:
        x = x.cuda()
    outputs = model(x)
    detections = outputs.data
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= args.visual_threshold:
            score = detections[0, i, j, 0].cpu().numpy()
            # label_name = labelmap[i - 1]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            # coords = (pt[0], pt[1], pt[2], pt[3])
            text = str(i - 1) + ' | ' + str(np.round(score, 2))
            cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])),
                          COLORS[2], 2)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img, text, (int(pt[2] - 30), int(pt[3])), font, 1, COLORS[3], 1)
            j += 1
    if cfg['DATA']['NAME'] == 'VOC':
        filename = os.path.join(devkit_path, str(img_id[1]) + '_voc.jpg')
    elif cfg['DATA']['NAME'] == 'COCO':
        filename = os.path.join(devkit_path, str(img_id) + '_coco.jpg')
    cv2.imwrite(filename, img)
    return


def test():
    # 4. Load Net
    net = build_ssd(phase='test', cfg=cfg)
    net.load_state_dict(torch.load(args.save_folder + "/" + args.evaluate))
    net.eval()

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # 5. print
    logger.info(f"args - {args}")

    # 6. Test
    with torch.no_grad():
        t0 = time.time()
        if cfg['DATA']['NAME'] == 'VOC':
            write_eval_test_results(dataset_test, net,
                                    transform=BaseTransform(size=cfg['DATA']['SIZE'], mean=MEANS))
        Visualized(dataset_test, net, transform=BaseTransform(size=cfg['DATA']['SIZE'], mean=MEANS))
        t1 = time.time()
        m = (t1 - t0) // 60
        s = (t1 - t0) % 60
    print("It took a total of {}m{}s to complete the testing.".format(int(m), int(s)))
    return


if __name__ == '__main__':
    logger.info("Visualization Program started")
    test()
    logger.info("Done!")
