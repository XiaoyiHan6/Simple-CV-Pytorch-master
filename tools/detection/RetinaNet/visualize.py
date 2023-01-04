import os
import cv2
import time
import random
import logging
import torch.nn.parallel
from utils.path import COLORS
from utils.get_logger import get_logger
from options.detection.RetinaNet.test_options import args, cfg, dataset_test, COCO_ROOT, model

assert torch.__version__.split('.')[0] == '1'
print('RetinaNet visualize.py CUDA available: {}'.format(torch.cuda.is_available()))

# Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)

devkit_path = os.path.join(args.Results, 'RetinaNet')


def Visualized(dataset, model):
    # shuffle
    random.shuffle(dataset.image_ids)
    info = dataset.image_ids[1]
    data = dataset[1]
    # h,w,c
    img, annotation, scale = data['img'], data['annot'], data['scale']

    if cfg['DATA']['NAME'] == 'COCO':
        path = os.path.join(COCO_ROOT, 'val2017', str(info).zfill(12) + ".jpg")
    elif cfg['DATA']['NAME'] == 'VOC':
        path = os.path.join(info[0], 'JPEGImages', info[1] + ".jpg")

    cv_img = cv2.imread(path)
    for annot in enumerate(annotation):
        xmin = int(annot[1][0] / scale)
        ymin = int(annot[1][1] / scale)
        xmax = int(annot[1][2] / scale)
        ymax = int(annot[1][3] / scale)
        label = str(int(annot[1][4]))
        cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), COLORS[0], 2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(cv_img, label, (xmin, int(ymin) + 10), font, 1, COLORS[1], 1)

    if args.cuda and torch.cuda.is_available():
        # h,w,c -> c,h,w
        img = torch.from_numpy(img).permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
    else:
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(dim=0)

    pred_score, pred_label, pred_bbox = model(img)

    pred_score = pred_score.cpu()
    pred_label = pred_label.cpu()
    pred_bbox = pred_bbox.cpu()
    pred_bbox /= scale

    for i, pred_s in enumerate(pred_score):
        if pred_s > args.visual_threshold:
            xmin = int(pred_bbox[i][0].numpy())
            ymin = int(pred_bbox[i][1].numpy())
            xmax = int(pred_bbox[i][2].numpy())
            ymax = int(pred_bbox[i][3].numpy())
            label = str(int(pred_label[i].numpy()))
            score = str(round(float(pred_score[i].numpy()), 2))
            text = label + ' | ' + score
            cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), COLORS[2], 2)
            cv2.putText(cv_img, text, (xmin, ymax), font, 1, COLORS[3], 1)
    if cfg['DATA']['NAME'] == 'VOC':
        filename = os.path.join(devkit_path, info[1] + '_VOC.jpg')
    elif cfg['DATA']['NAME'] == 'COCO':
        filename = os.path.join(devkit_path, str(info) + '_COCO.jpg')
    cv2.imwrite(filename, cv_img)


if __name__ == '__main__':
    logger.info("Visualization Program started")

    if args.cuda and torch.cuda.is_available():
        model = model.cuda()

    if args.evaluate:
        other, ext = os.path.splitext(args.evaluate)
        if ext == '.pkl' or 'pth':
            model_load = os.path.join(args.save_folder, args.evaluate)
            model.load_state_dict(torch.load(model_load))
            print("Loading weights into state dict...")
        else:
            print('Sorry only .pth and .pkl files supported.')
    elif args.evaluate is None:
        print("Sorry, you should load weights! ")

    if args.cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    logger.info(f"{args}")

    # interference
    model.training = False
    model.eval()
    model.module.freeze_bn()

    with torch.no_grad():
        t_start = time.time()
        Visualized(dataset_test, model)
        t_end = time.time()
        m = (t_end - t_start) // 60
        s = (t_end - t_start) % 60
        print("It took a total of {}m{}s to complete the testing.".format(int(m), int(s)))

    logger.info("Done!")
