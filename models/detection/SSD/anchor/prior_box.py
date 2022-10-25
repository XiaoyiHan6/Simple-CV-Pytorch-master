from __future__ import division
import torch
from math import sqrt as sqrt
from itertools import product as product


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['DATA']['SIZE']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['PRIOR_BOX']['ASPECT_RATIOS'])
        self.variance = cfg['PRIOR_BOX']['VARIANCE'] or [0.1]
        self.feature_maps = cfg['PRIOR_BOX']['FEATURE_MAPS']
        self.min_sizes = cfg['PRIOR_BOX']['MIN_SIZES']
        self.max_sizes = cfg['PRIOR_BOX']['MIN_SIZES']
        self.steps = cfg['PRIOR_BOX']['STEPS']
        self.aspect_ratios = cfg['PRIOR_BOX']['ASPECT_RATIOS']
        self.clip = cfg['PRIOR_BOX']['CLIP']
        self.version = cfg['DATA']['NAME']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    from options.detection.SSD.train_options import cfg

    box = PriorBox(cfg)
    print('Priors box shape:', box.forward().shape)
    print('Priors box:\n', box.forward())
