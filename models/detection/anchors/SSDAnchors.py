import torch
from torch import nn
from math import sqrt as sqrt
import torch.nn.functional as F
from torch.cuda.amp import autocast
from itertools import product as product


class SSDAnchors(nn.Module):
    """
    anchors: 38x38x4+19x19x6+10x10x6+5x5x6+3x3x4+1x1x4=8732
    """

    def __init__(self, img_size=300, feature_maps=[38, 19, 10, 5, 3, 1],
                 steps=[8, 16, 32, 64, 100, 300],
                 aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                 clip=True, variance=[0.1, 0.2], version='VOC',
                 min_sizes=[30, 60, 111, 162, 213, 264],
                 max_sizes=[60, 111, 162, 213, 264, 315],
                 ):
        super(SSDAnchors, self).__init__()
        if img_size == None:
            self.img_size = 300
        else:
            self.img_size = img_size
        if feature_maps == None:
            self.feature_maps = [38, 19, 10, 5, 3, 1]
        else:
            self.feature_maps = feature_maps
        if steps == None:
            self.steps = [8, 16, 32, 64, 100, 300]
        else:
            self.steps = steps
        if aspect_ratios == None:
            self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        else:
            self.aspect_ratios = aspect_ratios
        if clip == None:
            self.clip = True
        else:
            self.clip = clip
        if variance == None:
            self.variance = [0.1, 0.2]
        else:
            self.variance = variance
        if version == 'VOC':
            self.version = version
            if min_sizes == None:
                self.min_sizes = [30, 60, 111, 162, 213, 264]
            else:
                self.min_sizes = min_sizes
            if max_sizes == [60, 111, 162, 213, 264, 315]:
                self.max_sizes = max_sizes
            else:
                self.max_sizes = [60, 111, 162, 213, 264, 315]
        elif version == 'COCO':
            self.version = version
            if min_sizes == [21, 45, 99, 153, 207, 261]:
                self.min_sizes = min_sizes
            else:
                self.min_sizes = [21, 45, 99, 153, 207, 261]
            if max_sizes == [45, 99, 153, 207, 261, 315]:
                self.max_sizes = max_sizes
            else:
                self.max_sizes = [45, 99, 153, 207, 261, 315]
        else:
            raise ValueError("Dataset type is error!")

    @autocast()
    def forward(self, x):
        mean = []

        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # feature_map of k-th
                f_k = float(self.img_size) / float(self.steps[k])

                # center
                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k

                # r==1, size = s_k
                s_k = self.min_sizes[k] / self.img_size
                mean += [cx, cy, s_k, s_k]

                # r==1, size = sqrt(s_k * s_(k+1))
                s_k_plus = self.max_sizes[k] / self.img_size
                s_k_prime = sqrt(s_k * s_k_plus)
                mean += [cx, cy, s_k_prime, s_k_prime]

                # ration != 1
                for r in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(r), s_k / sqrt(r)]
                    mean += [cx, cy, s_k / sqrt(r), s_k * sqrt(r)]
        # torch
        boxes = torch.tensor(mean).view(-1, 4)
        # norm [0,1]
        if self.clip:
            boxes.clamp_(max=1, min=0)
        # anchor boxes
        return boxes


if __name__ == "__main__":
    anchors = SSDAnchors(version='COCO')
    print(anchors.min_sizes)
    print(anchors.forward().shape)
    print(anchors.forward())
