import sys
import skimage
import skimage.io
import numpy as np
import os.path as osp
from utils.path import VOC_path
from torch.utils.data import Dataset

VOC_CLASSES = (  # always index 0
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

VOC_ROOT = VOC_path

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class VocDetection(Dataset):
    def __init__(self, root_dir, set_name=[('2007', 'trainval'), ('2012', 'trainval')], transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.keep_difficult = False
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.image_ids = list()
        self.class_to_ind = dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        for (year, name) in set_name:
            rootpath = osp.join(self.root_dir, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.image_ids.append((rootpath, line.strip()))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        img_info = self.image_ids[image_index]
        path = osp.join(self._imgpath % img_info)
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        img_info = self.image_ids[image_index]
        path = osp.join(self._annopath % img_info)
        target = ET.parse(path).getroot()
        annots = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for pt in pts:
                cur_pt = float(bbox.find(pt).text)
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            annots += [bndbox]
        annots = np.array(annots)
        return annots

    def image_aspect_ratio(self, image_index):
        img = self.load_image(image_index)
        height, width, _ = img
        # w/h
        return float(width) / float(height)

    def num_classes(self):
        return 20


if __name__ == '__main__':
    voc = VocDetection(VOC_ROOT)
    print(voc.load_image(1).shape)
    print(voc.load_annotations(1))
