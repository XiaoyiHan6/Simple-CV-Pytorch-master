"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""
from __future__ import print_function
import os
import sys
import time
import torch
import pickle
import logging
import numpy as np
from tqdm import tqdm
from utils.get_logger import get_logger
from data.detection.RetinaNet.voc import VOC_CLASSES as labelmap
from options.detection.RetinaNet.eval_options import args, cfg, dataset_eval, VOC_ROOT, model

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

assert torch.__version__.split('.')[0] == '1'
print('RetinaNet eval_voc.py CUDA available: {}'.format(torch.cuda.is_available()))

# Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)

# Path
annopath = os.path.join(VOC_ROOT, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(VOC_ROOT, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(VOC_ROOT, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.Results
set_type = 'test'


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # /data/PycharmProject/Simple-CV-Pytorch-master/results/RetinaNet/VOC/det/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'RetinaNet/VOC/det')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(results):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for idx in range(len(results)):
                dets = results[idx]
                # the VOCdevkit expects 1-based indices
                if dets['label'] == cls_ind:
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(dets['image_id'][1], dets['score'],
                                   dets['bbox'][0] + 1, dets['bbox'][1] + 1,
                                   dets['bbox'][2] + 1, dets['bbox'][3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    # /data/PycharmProject/Simple-CV-Pytorch-master-master/results/RetinaNet/VOC/annot_cache (annotations_cache)
    cachedir = os.path.join(devkit_path, 'RetinaNet/VOC')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imgsetpath.format(set_type), cls, cachedir,
            ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    return aps


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    # /data/PycharmProject/Simple-CV-Pytorch-master/results/RetinaNet/VOC/annots.pkl
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        with tqdm(total=len(imagenames)) as pbar:
            for i, imagename in enumerate(imagenames):
                recs[imagename] = parse_rec(annopath % (imagename))
                pbar.update(1)
        pbar.close()
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def eval_voc(dataset, model, threshold=0.05):
    num_images = len(dataset)
    model.eval()

    output_dir = get_output_dir(devkit_path, 'RetinaNet/VOC/annot_cache')
    det_file = os.path.join(output_dir, '../detections.pkl')
    with torch.no_grad():
        # start collecting results
        results = []
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
                    # compute predicted labels and scores
                    for box_id in range(boxes.shape[0]):
                        score = float(scores[box_id])
                        label = int(labels[box_id])
                        box = boxes[box_id, :]

                        # scores are sorted, so we can break
                        if score < threshold:
                            break
                        # (image_id,label,score,x1,y1,x2,y2)
                        img_result = {
                            'image_id': dataset.image_ids[i],
                            'label': label,
                            'score': float(score),
                            'bbox': box.tolist(),
                        }
                        # append detection to results
                        results.append(img_result)

                pbar.update(1)
    pbar.close()
    if not len(results):
        return

    with open(det_file, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    aps = evaluate_detections(results, output_dir, dataset)
    return aps


def evaluate_detections(results, output_dir, dataset):
    write_voc_results_file(results)
    aps = do_python_eval(output_dir)
    return aps


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
        aps = eval_voc(dataset_eval, model)
        if aps is not None:
            for i, cls in enumerate(labelmap):
                logger.info(f"AP for {cls} = {float(aps[i]):.4f}")
            logger.info(f"Mean AP = {np.mean(aps):.4f}")
        t_end = time.time()
        m = (t_end - t_start) // 60
        s = (t_end - t_start) % 60
        print("The Finished Time is {}m{}s".format(int(m), int(s)))

    logger.info("Done!")
