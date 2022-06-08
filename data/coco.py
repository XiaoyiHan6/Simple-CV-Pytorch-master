"""COCO Dataset Classes"""
import os
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from utils.path import COCO2017_path, COCO2014_path
from skimage import io, color

COCO_2014_ROOT = COCO2014_path
COCO_ROOT = COCO2017_path
COCO_CLASSES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
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
    (12, 50, 60),
    (16, 179, 33),
    (112, 69, 165),
    (15, 139, 63),
    (33, 191, 159),
    (182, 173, 32),
    (34, 113, 133),
    (90, 135, 34),
    (53, 34, 86),
    (141, 35, 190),
    (6, 171, 8),
    (118, 76, 112),
    (89, 60, 55),
    (15, 54, 88),
    (112, 75, 181),
    (42, 147, 38),
    (138, 52, 63),
    (128, 65, 149),
    (106, 103, 24),
    (168, 33, 45),
    (28, 136, 135),
    (86, 91, 108),
    (52, 11, 76),
    (142, 6, 189),
    (57, 81, 168),
    (55, 19, 148),
    (182, 101, 89),
    (44, 65, 179),
    (1, 33, 26),
    (122, 164, 26),
    (70, 63, 134),
    (137, 106, 82),
    (120, 118, 52),
    (129, 74, 42),
    (182, 147, 112),
    (22, 157, 50),
    (56, 50, 20),
    (2, 22, 177),
    (156, 100, 106),
    (21, 35, 42),
    (13, 8, 121),
    (142, 92, 28),
    (45, 118, 33),
    (105, 118, 30),
    (7, 185, 124),
    (46, 34, 146),
    (105, 184, 169),
    (22, 18, 5),
    (147, 71, 73),
    (181, 64, 91),
    (31, 39, 184),
    (164, 179, 33),
    (96, 50, 18),
    (95, 15, 106),
    (113, 68, 54),
    (136, 116, 112),
    (119, 139, 130),
    (31, 139, 34),
    (66, 6, 127),
    (62, 39, 2),
    (49, 99, 180),
    (49, 119, 155),
    (153, 50, 183),
    (125, 38, 3),
    (129, 87, 143),
    (49, 87, 40),
    (128, 62, 120),
    (73, 85, 148),
    (28, 144, 118),
    (29, 9, 24),
    (175, 45, 108),
    (81, 175, 64),
    (178, 19, 157),
    (74, 188, 190),
    (18, 114, 2),
    (62, 128, 96),
    (21, 3, 150),
    (0, 6, 95),
    (2, 20, 184),
    (122, 37, 185),
]


class CocoDetection(Dataset):
    def __init__(self,
                 root_dir,
                 set_name='train2017',
                 transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        # annotations
        self.coco = COCO(
            os.path.join(self.root_dir,
                         'annotations',
                         'instances_' + self.set_name + '.json'))

        self.ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        # load class name (name->label)
        self.cat_ids = self.coco.getCatIds()
        self.categories = self.coco.loadCats(self.cat_ids)

        self.categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}

        for c in self.categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
            params: idx
            return:  img,  annots, scale
        """
        # idx name
        # annots == targets
        img = self.load_image(idx)
        annot = self.load_annots(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, idx):

        # image_info: {'license': 3, 'file_name': '000000288174.jpg',
        #             'coco_url': 'http://images.cocodataset.org/train2017/000000288174.jpg',
        #             'height': 427, 'width': 640, 'date_captured': '2013-11-19 23:47:46',
        #             'flickr_url': 'http://farm6.staticflickr.com/5101/5651186170_9ff5af6e3e_z.jpg',
        #             'id': 288174}

        image_info = self.coco.loadImgs(self.ids[idx])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        # path:  /data/public/coco2017/train2017/000000288174.jpg
        img = io.imread(path)
        # height, width, channels = img.shape
        if len(img.shape) == 2:
            img = color.gray2rgb(img)
        # <class 'numpy.ndarray'>
        return img.astype(np.float32) / 255.0

    def load_annots(self, idx):
        # get ground truth annotations
        annot_ids = self.coco.getAnnIds(imgIds=self.ids[idx], iscrowd=False)
        # parse annotations
        annots = np.zeros((0, 5))

        if len(annot_ids) == 0:
            return annots

        # parse annotations
        coco_annots = self.coco.loadAnns(annot_ids)
        for idx, a in enumerate(coco_annots):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            annot = np.zeros((1, 5))
            annot[0, :4] = a['bbox']
            annot[0, 4] = self.coco_label_to_label(a['category_id'])
            annots = np.append(annots, annot, axis=0)

        # [x, y, w, h] -> [x1, y1, x2, y2]
        annots[:, 2] = annots[:, 0] + annots[:, 2]
        annots[:, 3] = annots[:, 1] + annots[:, 3]
        # annot = [x_min, y_min, x_max, y_max, id]
        return annots

    def coco_label_to_label(self, category_id):
        return self.coco_labels_inverse[category_id]

    def label_to_coco_label(self, coco_label):
        return self.coco_labels[coco_label]

    def num_classes(self):
        return 80

    def image_aspect_ratio(self, idx):
        image = self.coco.loadImgs(self.ids[idx])[0]
        return float(image['width']) / float(image['height'])
