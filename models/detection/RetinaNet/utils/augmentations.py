import os
import sys
import skimage
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)


# 3.
class Resize(object):
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
        return {'img': new_image, 'annot': annots, 'scale': scale}


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


class RandomCrop(object):
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

    resize = Resize()
    sample = resize(sample)
    img, annots = sample['img'], sample['annot']
    height, width, depth = img.shape
    skimage.io.imshow(img)
    for i, coord in enumerate(annots):
        print("resize coord:", coord)
        a = plt.Rectangle((coord[0], coord[1]), coord[2] - coord[0],
                          coord[3] - coord[1], fill=False, edgecolor='b', linewidth=2)
        plt.gca().add_patch(a)
    plt.show()
    print("resize new_height:{} , new_width:{}, new_depth:{}".format(height, width, depth))
