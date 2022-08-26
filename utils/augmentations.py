import os
import sys
import torch
import random
import skimage
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from utils.iou import iou_numpy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class SSDExpand(object):
    def __init__(self, mean=(104, 117, 123), flip_prob=0.5):
        self.flip_prob = flip_prob
        self.mean = mean

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        if annots.shape[0] == 0:
            return sample
        if np.random.uniform(0, 1) > self.flip_prob:
            height, width, channels = img.shape
            ratio = random.uniform(1, 4)
            left = random.uniform(0, width * ratio - width)
            top = random.uniform(0, height * ratio - height)
            expand_image = np.zeros((int(height * ratio), int(width * ratio), channels), dtype=img.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height), int(left):int(left + width)] = img
            img = expand_image

            annots = annots.copy()
            annots[:, :2] += (int(left), int(top))
            annots[:, 2:4] += (int(left), int(top))
            sample['img'] = img
            sample['annot'] = annots
            return sample
        else:
            return sample


class SSDToAbsoluteCoords(object):
    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        if annots.shape[0] == 0:
            return sample
        height, width, _ = img.shape
        annots[:, 0] *= width
        annots[:, 2] *= width
        annots[:, 1] *= height
        annots[:, 3] *= height
        sample['img'] = img
        sample['annot'] = annots
        return sample


class SSDToPercentCoords(object):
    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        if annots.shape[0] == 0:
            return sample
        height, width, _ = img.shape
        annots[:, 0] /= width
        annots[:, 2] /= width
        annots[:, 1] /= height
        annots[:, 3] /= height
        sample['img'] = img
        sample['annot'] = annots
        return sample


class SSDRandSampleCrop(object):
    def __init__(self):
        super(SSDRandSampleCrop, self).__init__()
        self.sample_optins = (
            # using original input image
            None,
            # min_iou and max_iou
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        if annots.shape[0] == 0:
            return sample
        height, width, _ = img.shape
        while True:
            mode = random.choice(self.sample_optins)
            if mode is None:
                return sample
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max iter (50)
            for _ in range(50):
                current_img = img

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                if h / w < 0.5 or h / w > 2:
                    continue
                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)

                # rect.shape: [x1,y1,x2,y2]
                rect = [int(left), int(top), int(left + w), int(top + h)]
                overlap = []
                for i, annot in enumerate(annots):
                    overlap.append(iou_numpy(annot[:4], rect))
                overlap = np.array(overlap)
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_img = current_img[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (annots[:, :2] + annots[:, 2:4]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                if not mask.any():
                    continue

                # take only matching gt boxes
                current_annots = annots[mask, :].copy()
                # should we use the box left and top corner or the crop's
                current_annots[:, :2] = np.maximum(current_annots[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left top)
                current_annots[:, :2] -= rect[:2]

                current_annots[:, 2:4] = np.minimum(current_annots[:, 2:4], rect[2:])
                current_annots[:, 2:4] -= rect[:2]
                sample['img'] = current_img
                sample['annot'] = current_annots
                return sample


class SSDResize(object):
    def __init__(self):
        pass

    def __call__(self, sample, side=300):
        img, annots = sample['img'], sample['annot']

        height, width, depth = img.shape

        img = skimage.transform.resize(img, (side, side))
        annots[:, 0] = annots[:, 0] / width * 300
        annots[:, 2] = annots[:, 2] / width * 300
        annots[:, 1] = annots[:, 1] / height * 300
        annots[:, 3] = annots[:, 3] / height * 300
        scale = height / width

        return {'img': torch.from_numpy(img), 'annot': torch.from_numpy(annots), 'scale': scale}


# 3.
class RetinaNetResize(object):
    def __init__(self):
        pass

    def __call__(self, sample, min_side=608, max_side=1024):
        # annots = [x_min, y_min, x_max, y_max, id]
        # skimge shape[0]=height,shape[1]=width
        img, annots = sample['img'], sample['annot']
        height, width, depth = img.shape
        smallest_side = min(height, width)

        # rescale the image so the smallest side is min_length
        scale = min_side / smallest_side

        # check if the largest side(length) is now greater than max_length, which can happen
        # when images have a large aspect ratio
        largest_side = max(height, width)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        image = skimage.transform.resize(img, (int(round(height * scale)), int(round(width * scale))))
        # resize the image with the computed scale
        height, width, depth = image.shape

        pad_h = 32 - height % 32
        pad_w = 32 - width % 32

        new_image = np.zeros((height + pad_h, width + pad_w, depth)).astype(np.float32)

        new_image[:height, :width, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


# 2.
class RandomFlip(object):
    """HorizontalFlip"""

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        if annots.shape[0] == 0:
            return sample
        if np.random.uniform(0, 1) > self.flip_prob:
            width = img.shape[1]
            img = img[:, ::-1, :]

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            annots[:, 0] = width - x2
            annots[:, 2] = width - x1
        return {'img': img, 'annot': annots}


class RetinaNetRandomCrop(object):
    def __init__(self, crop_prob=0.5):
        self.crop_prob = crop_prob

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        if annots.shape[0] == 0:
            return img, annots

        if np.random.uniform(0, 1) > self.crop_prob:
            h, w, _ = img.shape
            max_bbox = np.concatenate([
                np.min(annots[:, 0:2], axis=0),
                np.max(annots[:, 2:4], axis=0)
            ], axis=-1)

            max_left_trans, max_up_trans = max_bbox[0], max_bbox[1]
            max_right_trans, max_down_trans = w - max_bbox[2], h - max_bbox[3]

            crop_xmin = max(
                0, int(max_bbox[0] - np.random.uniform(0, max_left_trans)))
            crop_ymin = max(
                0, int(max_bbox[1] - np.random.uniform(0, max_up_trans)))
            crop_xmax = max(
                w, int(max_bbox[2] + np.random.uniform(0, max_right_trans)))
            crop_ymax = max(
                h, int(max_bbox[3] + np.random.uniform(0, max_down_trans)))
            img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            annots[:, [0, 2]] = annots[:, [0, 2]] - crop_xmin
            annots[:, [1, 3]] = annots[:, [1, 3]] - crop_ymin
            h_new, w_new, _ = img.shape
        return {'img': img, 'annot': annots}


# 1.
class Normalize(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])
        pass

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        img = (img.astype(np.float32) - self.mean) / self.std
        return {'img': img, 'annot': annots}


class UnNormalize(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean

        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


if __name__ == '__main__':
    img_root = os.path.join(BASE_DIR, 'images', 'detection', '000002.jpg')
    img = skimage.io.imread(img_root)
    height, width, depth = img.shape
    print("height:{} , width:{}, depth:{}".format(height, width, depth))
    annot2 = np.array([[139.0, 200.0, 207.0, 301.0, 18.0]])
    annot1 = np.array([[48.0, 240.0, 195.0, 371.0, 11.0], [8.0, 12.0, 352.0, 498.0, 14.0]])
    annot3 = np.array([[123.0, 155.0, 215.0, 195.0, 17.0], [239.0, 156.0, 307.0, 205.0, 8.0]])
    sample = {'img': img, 'annot': annot2}
    skimage.io.imshow(img)
    for i, coord in enumerate(sample['annot']):
        a = plt.Rectangle((coord[0], coord[1]), coord[2] - coord[0],
                          coord[3] - coord[1], fill=False, edgecolor='r', linewidth=2)
        plt.gca().add_patch(a)
    plt.show()

    expand = SSDExpand()
    sample = expand(sample)
    img, annots = sample['img'], sample['annot']
    height, width, depth = img.shape
    skimage.io.imshow(img)
    for i, coord in enumerate(annots):
        print("expand coord:", coord)
        a = plt.Rectangle((coord[0], coord[1]), coord[2] - coord[0],
                          coord[3] - coord[1], fill=False, edgecolor='b', linewidth=2)
        plt.gca().add_patch(a)
    plt.show()
    print("expand new_height:{} , new_width:{}, new_depth:{}".format(height, width, depth))

    crop = SSDRandSampleCrop()
    sample = crop(sample)
    img, annots = sample['img'], sample['annot']
    height, width, depth = img.shape
    skimage.io.imshow(img)
    for i, coord in enumerate(annots):
        print("crop coord:", coord)
        a = plt.Rectangle((coord[0], coord[1]), coord[2] - coord[0],
                          coord[3] - coord[1], fill=False, edgecolor='b', linewidth=2)
        plt.gca().add_patch(a)
    plt.show()
    print("crop new_height:{} , new_width:{}, new_depth:{}".format(height, width, depth))

    randomflip = RandomFlip()
    sample = randomflip(sample)
    img, annots = sample['img'], sample['annot']
    height, width, depth = img.shape
    skimage.io.imshow(img)
    for i, coord in enumerate(annots):
        print("randomflip coord:", coord)
        a = plt.Rectangle((coord[0], coord[1]), coord[2] - coord[0],
                          coord[3] - coord[1], fill=False, edgecolor='g', linewidth=2)
        plt.gca().add_patch(a)
    plt.show()
    print("randomflip new_height:{} , new_width:{}, new_depth:{}".format(height, width, depth))

    resize = SSDResize()
    sample = resize(sample)
    img, annots, scale = sample['img'], sample['annot'], sample['scale']
    height, width, depth = img.shape
    skimage.io.imshow(img.numpy())
    for i, coord in enumerate(annots):
        print("resize coord:", coord)
        a = plt.Rectangle((coord[0], coord[1]), coord[2] - coord[0],
                          coord[3] - coord[1], fill=False, edgecolor='g', linewidth=2)
        plt.gca().add_patch(a)
    plt.show()
    print("resize new_height:{} , new_width:{}, new_depth:{}, scale:{}".format(height, width, depth, scale))
