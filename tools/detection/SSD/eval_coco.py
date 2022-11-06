from __future__ import print_function
import os
import json
import time
import torch
import logging
from tqdm import tqdm
from utils.path import MEANS
import torch.backends.cudnn as cudnn
from utils.get_logger import get_logger
from pycocotools.cocoeval import COCOeval
from models.detection.SSD.ssd import build_ssd
from models.detection.SSD.utils.augmentations import BaseTransform
from options.detection.SSD.eval_options import args, cfg, dataset_eval

assert torch.__version__.split('.')[0] == '1'
print('SSD eval_coco.py CUDA available: {}'.format(torch.cuda.is_available()))

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)


def get_output_dir(name, phase):
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def eval_coco(dataset, net, transform, cuda):
    num_images = len(dataset)
    net.eval()

    with torch.no_grad():
        # start collecting results
        results = []
        with tqdm(total=num_images)as pbar:
            for i in range(num_images):
                img = dataset.pull_image(i)

                h, w, _ = img.shape
                x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1).unsqueeze(dim=0)
                # run network
                if cuda:
                    x = x.cuda()
                detections = net(x).data
                for j in range(1, detections.size(1)):
                    dets = detections[0, j, :]
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.size(0) == 0:
                        continue
                    boxes = dets[:, 1:]
                    boxes[:, 0] *= w
                    boxes[:, 2] *= w
                    boxes[:, 1] *= h
                    boxes[:, 3] *= h
                    scores = dets[:, 0].cpu().numpy()
                    boxes = boxes.cpu().numpy()
                    # compute predicted labels and scores
                    # for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    for id, box in enumerate(boxes):
                        # append detection for each positively labeled class
                        image_result = {
                            'image_id': dataset.ids[i],
                            'category_id': dataset.label_to_coco_label(int(j - 1)),
                            'score': float(scores[id]),
                            'bbox': [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                            # boxes(xmin,ymin,xmax,ymax) change to (x,y,w,h) (MS COCO standard)
                        }

                        # append detection to results
                        results.append(image_result)

                pbar.update(1)
        pbar.close()
    if not len(results):
        return
    devkit_path = os.path.join(args.Results, 'SSD')
    devkit_path = get_output_dir(devkit_path, 'COCO')
    # write output
    with open('{}/{}_bbox_results.json'.format(devkit_path, cfg['DATA']['NAME'].lower()), 'w') as f:
        json.dump(results, f)

    # load results in COCO evaluation tool
    coco_gt = dataset.coco
    coco_pred = coco_gt.loadRes('{}/{}_bbox_results.json'.format(devkit_path, cfg['DATA']['NAME'].lower()))

    # run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    all_eval_result = coco_eval.stats
    net.train()

    return all_eval_result


if __name__ == '__main__':
    logger.info("Evaluation COCO Dataset Program started")
    # load net
    net = build_ssd(phase='test', cfg=cfg)  # initialize SSD

    net.load_state_dict(torch.load(args.save_folder + "/" + args.evaluate))
    net.eval()
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    t_start = time.time()
    coco_eval = eval_coco(dataset_eval, net, transform=BaseTransform(cfg['DATA']['SIZE'], MEANS), cuda=args.cuda)
    t_end = time.time()
    if coco_eval is not None:
        logger.info(
            f"\nIoU=0.5:0.95, area=all, maxDets=100, mAP:{coco_eval[0]:.3f}\n"
            f"IoU=0.5, area=all, maxDets=100, mAP:{coco_eval[1]:.3f}\n"
            f"IoU=0.75, area=all, maxDets=100, mAP:{coco_eval[2]:.3f}\n"
            f"IoU=0.5:0.95, area=small, maxDets=100, mAP:{coco_eval[3]:.3f}\n"
            f"IoU=0.5:0.95, area=medium, maxDets=100, mAP:{coco_eval[4]:.3f}\n"
            f"IoU=0.5:0.95, area=large, maxDets=100, mAP:{coco_eval[5]:.3f}\n"
            f"IoU=0.5:0.95, area=all, maxDets=1, mAR:{coco_eval[6]:.3f}\n"
            f"IoU=0.5:0.95, area=all, maxDets=10, mAR:{coco_eval[7]:.3f}\n"
            f"IoU=0.5:0.95, area=all, maxDets=100, mAR:{coco_eval[8]:.3f}\n"
            f"IoU=0.5:0.95, area=small, maxDets=100, mAR:{coco_eval[9]:.3f}\n"
            f"IoU=0.5:0.95, area=medium, maxDets=100, mAR:{coco_eval[10]:.3f}\n"
            f"IoU=0.5:0.95, area=large, maxDets=100, mAR:{coco_eval[11]:.3f}."
        )
    m = (t_end - t_start) // 60
    s = (t_end - t_start) % 60
    print("It took a total of {}m{}s to complete the evaluation.".format(int(m), int(s)))
    logger.info("Done!")
