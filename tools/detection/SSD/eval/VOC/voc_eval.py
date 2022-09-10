""" SSD VOC EVAL"""
import os
import sys
import time

import torch
import numpy as np
from tqdm import tqdm

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from data import *
from data.config import results_path
from utils.augmentations.SSDAugmentations import BaseTransform

YEAR = '2007'
# annopath = /data/VOCdevkit/VOC2007/Annotations
annopath = os.path.join(VOC_ROOT, 'VOC' + YEAR, 'Annotations', '%s.xml')

# imgpath = /data/VOCdevkit/VOC2007/JPEGImages
imgpath = os.path.join(VOC_ROOT, 'VOC' + YEAR, 'JPEGImages', '%s.jpg')

# imgsetpath = /data/VOCdevkit/VOC2007/ImageSets
imgsetpath = os.path.join(VOC_ROOT, 'VOC' + YEAR, 'ImageSets', 'Main', '{:s}.txt')

# devkit_path = /data/PycharmProject/Simple-CV-Pytorch-master/result/SSD/VOC
devkit_path = os.path.join(results_path, 'SSD', '')

dataset_mean = (104, 117, 123)

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
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # /data/PycharmProject/Simple-CV-Pytorch-master/results/SSD/VOC/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = devkit_path
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(pred_scores, pred_boxes, pred_labels, info):
    for cls_ind, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'a+') as f:
            dets = []
            for i in range(len(pred_labels)):
                if pred_labels[i] == cls_ind:
                    det = []
                    det.append(info[1])
                    det.append(pred_scores[i])
                    for j in range(4):
                        det.append(pred_boxes[i][j])
                    dets += [det]
            if dets == []:
                continue
            dets = np.array(dets)
            # the VOCdevkit expects 1-based indices
            for k in range(len(dets)):
                f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                        format(dets[k][0], float(dets[k][1]),
                               float(dets[k][2]) + 1, float(dets[k][3]) + 1,
                               float(dets[k][4]) + 1, float(dets[k][5]) + 1))


def do_python_eval(use_07=True):
    # /data/PycharmProject/Simple-CV-Pytorch-master/results/SSD/VOC
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imgsetpath.format(set_type), cls,
            ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
    print('Mean AP={:.4f} '.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    return aps, labelmap


# recall, precision
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
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath % (imagename))
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
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


def write_voc_result(dataset, model, transform, img_size=300):
    model.eval()
    # delete previous detection results
    # /data/PycharmProject/Simple-CV-Pytorch-master/results/SSD/VOC/
    # print("Deleting previous detection results...")
    if not os.path.exists(devkit_path):
        os.mkdir(devkit_path)
    for cls_ind, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        if os.path.exists(filename):
            os.remove(filename)

    with tqdm(total=len(dataset.ids)) as pbar:
        for img_ind, info in enumerate(dataset.ids):
            pred_scores = []
            pred_boxes = []
            pred_labels = []

            img = dataset.load_image(img_ind)
            annot = dataset.load_annots(img_ind)

            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

            if torch.cuda.is_available():
                img = torch.from_numpy(transform(img)).permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
            else:
                img = torch.from_numpy(transform(img)).permute(2, 0, 1).float().unsqueeze(dim=0)

            output = model(img)
            for i in range(len(output)):
                boxes, scores, labels = output[i]
                boxes = boxes.cpu().data
                scores = scores.cpu().data.numpy()
                labels = labels.cpu().data.numpy()
                labels -= 1
                results = []
                gt_labels_copy = list(set(annot[:, 4]))
                for j, a in enumerate(gt_labels_copy):
                    x = 0
                    num_copy = list(annot[:, 4]).count(a)
                    if a in labels:
                        for num_copy_i in range(num_copy):
                            try:
                                out = list(labels).index(a, x)
                            except Exception as e:
                                continue
                            if out in results:
                                continue
                            else:
                                results.append(out)
                                x = out + 1
                    else:
                        continue
                if results is []:
                    break
                else:
                    for _, result in enumerate(results):
                        pred_scores.append(scores[result])
                        boxes[result] /= img_size
                        coord = (boxes[result] * scale).numpy()
                        pred_boxes.append(coord)
                        pred_labels.append(labels[result])

            write_voc_results_file(pred_scores, pred_boxes, pred_labels, info)
            pbar.update(1)
    print("Writing VOC results file...")


def evaluate_voc(dataset, model):
    write_voc_result(dataset, model, transform=BaseTransform())
    aps, labelmap = do_python_eval()
    return aps, labelmap
