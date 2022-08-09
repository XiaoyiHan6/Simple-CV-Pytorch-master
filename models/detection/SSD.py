import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F


class SSD(nn.Module):
    def __init__(self):
        super(SSD, self).__init__()

    @autocast()
    def forward(self, x):
        return x


backbone = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    '512': [],
}
neck = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
head = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [],
}
