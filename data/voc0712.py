"""VOC Dataset Classes"""
# cv2 bgr -> rgb
import os
import sys
import cv2
import numpy as np
from utils.path import VOC_path
from torch.utils.data import Dataset

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = [  # always index 0
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
    'tvmonitor'
]

colors = [
    (39, 129, 113),
    (164, 80, 133),
    (83, 122, 114),
    (99, 81, 172),
    (95, 56, 104),
    (37, 84, 86),
    (14, 89, 122),
    (80, 7, 65),
    (10, 102, 25),
    (90, 185, 109),
    (106, 110, 132),
    (169, 158, 85),
    (188, 185, 26),
    (103, 1, 17),
    (82, 144, 81),
    (92, 7, 184),
    (49, 81, 155),
    (179, 177, 69),
    (93, 187, 158),
    (13, 39, 73),
]

# note: if you used our download scripts, this should be right
VOC_ROOT = VOC_path


class VocDetection(Dataset):
    def __init__(self,
                 root_dir,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None,
                 keep_difficult=False):
        self.root_dir = root_dir
        self.image_set = image_sets
        self.transform = transform
        self.categories = VOC_CLASSES

        self.category_id_to_voc_label = dict(
            zip(self.categories, range(len(self.categories))))

        self.voc_label_to_category_id = {
            i: category_id for i, category_id in
            self.category_id_to_voc_label.items()}

        self.keep_difficult = keep_difficult

        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = []
        for (year, name) in image_sets:
            # rootpath = /data/VOCdevkit/VOC2007
            rootpath = os.path.join(self.root_dir, 'VOC' + year)
            for line in open(
                    os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annots(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        sample['img'] = sample['img'][:, :, (2, 1, 0)]
        return sample

    def load_image(self, idx):
        img = cv2.imread(self._imgpath % self.ids[idx], 1)
        height, width, channels = img.shape
        return img

    def load_annots(self, idx):
        target = ET.parse(self._annopath % self.ids[idx]).getroot()
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
            label_idx = self.category_id_to_voc_label[name]
            bndbox.append(label_idx)
            annots += [bndbox]  # [xmin,ymin,xmax,ymax,label_ind]
            # img_id = target.find('filename').text[:-4]

        annots = np.array(annots)
        # format:[[x1, y1, x2, y2, label_ind], ... ]
        return annots.astype(np.float32)

    def find_category_id_from_voc_label(self, voc_label):
        return self.voc_label_to_category_id[voc_label]

    def image_aspect_ratio(self, idx):
        image = self.load_image(idx)
        # w/h
        return float(image.shape[1] / float(image.shape[0]))

    def num_classes(self):
        return 20


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from utils.collate import ssd_collate
    from utils.augmentations.SSDAugmentations import SSDAugmentation

    Data = VocDetection(VOC_ROOT, transform=SSDAugmentation())
    data_loader = DataLoader(Data,
                             batch_size=16,
                             num_workers=0,
                             shuffle=True,
                             pin_memory=True,
                             collate_fn=ssd_collate)
    print('the data length is : ', len(data_loader))
    for data in data_loader:
        img, annots = data['img'], data['annot']
        print("img.shape:", img.shape)
        print("annots:", annots)
