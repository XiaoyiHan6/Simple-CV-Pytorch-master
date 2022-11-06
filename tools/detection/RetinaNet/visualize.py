import warnings
import os
import sys
import cv2
import time
import random
import logging
import torch.nn.parallel
from utils.get_logger import get_logger
from options.detection.RetinaNet.test_options import args, cfg, dataset_test, model

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

devkit_path = results_path


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
        thresh = pred_score[0]
        for index in range(len(pred_score)):
            if pred_score[index] > thresh:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: ' + '\n')
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num) + ' label: '
                            + str((pred_label[index]).numpy()) + ' || ' + 'score: ' +
                            str((pred_score[index]).numpy()) + ' || ' + 'coords: ' + ' || '.join(
                        str(c.numpy()) for c in pred_bbox[index]) + '\n')
                thresh = pred_score[index]


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
        image_root = os.path.join(COCO_ROOT, 'val2017', info + ".jpg")

    image = cv2.imread(image_root)
    for i, a in enumerate(annot):
        xmin = int(a[0].numpy())
        ymin = int(a[1].numpy())
        xmax = int(a[2].numpy())
        ymax = int(a[3].numpy())
        label_a = str(int(a[4].numpy()))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, label_a, (xmin, ymin), font, 1, (0, 0, 255), 1)

    thresh = pred_score[0]
    for i, pred_s in enumerate(pred_score):
        if pred_s > thresh:
            xmin = int(pred_bbox[i][0].numpy())
            ymin = int(pred_bbox[i][1].numpy())
            xmax = int(pred_bbox[i][2].numpy())
            ymax = int(pred_bbox[i][3].numpy())
            label_b = str(int(pred_label[i].numpy()))
            score_b = str(round(float(pred_score[i].numpy()), 2))
            text = label_b + ' | ' + score_b
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
            cv2.putText(image, text, (xmin, ymax), font, 1, (255, 0, 0), 1)
            thresh = pred_s
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

    # 4. Define to train mode

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
