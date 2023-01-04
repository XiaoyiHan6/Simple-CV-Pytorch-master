import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import time
import torch
import logging
from tqdm import tqdm
from utils.get_logger import get_logger
from pycocotools.cocoeval import COCOeval
from options.detection.RetinaNet.eval_options import args, cfg, dataset_eval, model

assert torch.__version__.split('.')[0] == '1'
print('RetinaNet eval_coco.py CUDA available: {}'.format(torch.cuda.is_available()))

# Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)


def get_output_dir(name, phase):
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def eval_coco(dataset, model, threshold=0.05):
    num_images = len(dataset)
    model.eval()
    with torch.no_grad():
        # start collecting results
        results = []
        image_ids = []
        with tqdm(total=num_images) as pbar:
            for i in range(num_images):
                data = dataset[i]
                img, scale = torch.from_numpy(data['img']), data['scale']
                if args.cuda and torch.cuda.is_available():
                    scores, labels, boxes = model(img.permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
                else:
                    scores, labels, boxes = model(img.permute(2, 0, 1).float().unsqueeze(dim=0))
                scores = scores.cpu()
                labels = labels.cpu()
                boxes = boxes.cpu()

                # correct boxes for image scale
                boxes /= scale
                if boxes.shape[0] > 0:
                    # change to (x,y,w,h) (MS COCO standard)
                    boxes[:, 2] -= boxes[:, 0]
                    boxes[:, 3] -= boxes[:, 1]

                    # compute predicted labels and scores
                    for box_id in range(boxes.shape[0]):
                        score = float(scores[box_id])
                        label = int(labels[box_id])
                        box = boxes[box_id, :]

                        # scores are sorted, so we can break
                        if score < threshold:
                            break
                        # append detection for each positively labeled class
                        img_result = {
                            'image_id': dataset.image_ids[i],
                            'category_id': dataset.label_to_coco_label(label),
                            'score': float(score),
                            'bbox': box.tolist(),
                        }
                        # append detection to results
                        results.append(img_result)

                # append image to list of processed images
                image_ids.append(dataset.image_ids[i])

                # print progress
                pbar.update(1)
        pbar.close()
        if not len(results):
            return
        # write output
        devkit_path = get_output_dir(args.Results, 'RetinaNet')
        devkit_path = get_output_dir(devkit_path, 'COCO')
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
        model.train()

        return all_eval_result


if __name__ == '__main__':
    logger.info("Program evaluating started!")

    if args.cuda and torch.cuda.is_available():
        model = model.cuda()

    if args.evaluate:
        other, ext = os.path.splitext(args.evaluate)
        if ext == '.pkl' or '.pth':
            model_load = os.path.join(args.save_folder, args.evaluate)
            model.load_state_dict(torch.load(model_load))
            print("Loading weights into state dict...")
        else:
            print("Sorry only .pth and .pkl files supported.")

    elif args.evaluate is None:
        print("Sorry, you should load weights!")

    if args.cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    logger.info(f"{args}")

    t_start = time.time()

    # interference
    model.training = False
    model.eval()
    model.module.freeze_bn()
    coco_eval = eval_coco(dataset_eval, model)
    t_end = time.time()

    if coco_eval is not None:
        logger.info(
            f"\nIoU=0.5:0.95, area=all, maxDets=100, mAP:{coco_eval[0]:.3f}\n"
            f"IoU=0.5, area=all, maxDets=100, mAP:{coco_eval[1]:.3f}\n"
            f"IoU=0.75, area=all, maxDets=100, mAP:{coco_eval[2]:.3f}\n"
            f"IoU=0.5:0.95, area=small, maxDets=100, mAP:{coco_eval[3]:.3f}\n"
            f"IoU=0.5:0.95, area=medium,maxDets=100,mAP:{coco_eval[4]:.3f}\n"
            f"IoU=0.5:0.95,area=large,maxDets=100,mAP:{coco_eval[5]:.3f}\n"
            f"IoU=0.5:0.95,area=all,maxDets=1,mAR:{coco_eval[6]:.3f}\n"
            f"IoU=0.5:0.95,area=all,maxDets=10,mAR:{coco_eval[7]:.3f}\n"
            f"IoU=0.5:0.95,area=all,maxDets=100,mAR:{coco_eval[8]:.3f}\n"
            f"IoU=0.5:0.95,area=small,maxDets=100,mAR:{coco_eval[9]:.3f}\n"
            f"IoU=0.5:0.95,area=medium,maxDets=100,mAR:{coco_eval[10]:.3f}\n"
            f"IoU=0.5:0.95,area=large,maxDets=100,mAR:{coco_eval[11]:.3f}."
        )

    m = (t_end - t_start) // 60
    s = (t_end - t_start) % 60
    print("The Finished Time is {}m{}s".format(int(m), int(s)))

    logger.info("Done!")
