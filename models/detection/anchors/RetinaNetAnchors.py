import torch
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast


# anchors = [x_min, y_min, x_max, x_max]

class RetinaNetAnchors(nn.Module):
    def __init__(self,
                 pyramid_levels=None,
                 strides=None,
                 sizes=None,
                 ratios=None,
                 scales=None):
        super(RetinaNetAnchors, self).__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    @autocast()
    def forward(self, image):
        """
        generate anchors
        """
        # (B, C, W, H)
        image_shape = np.array(image.shape[2:])
        # (W and H) of feature map : [org/8, org/16, org/32, org/64, org/128]
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            # the center of anchors is (0, 0), and generate the  information about 9 anchors,forms:(x1, y1, x2, y2)
            anchors = generate_anchors(base_size=self.sizes[idx],
                                       ratios=self.ratios,
                                       scales=self.scales)

            shifted_anchors = shift(image_shapes[idx],
                                    self.strides[idx],
                                    anchors)

            # shifted_anchors are added in all_anchors
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        # np.expand_dims extend all_anchors
        all_anchors = np.expand_dims(all_anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchors (reference) windows by enumerating aspect ratios x
    scales w.r.t. a reference window.
    """
    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        # based on the reference of the base_size, like the operation of uniformization
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    #  compute the total number of anchors
    num_anchors = len(ratios) * len(scales)

    # initialize output anchors (9, 4), 4 indicate location
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    # np.tile (a, (2, 3)):(the x-axis of a)  copy twice, and (the y-axis of a) copy three times
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]
    # areas = [1024,1625,2580, 1024,1625,2580, 1024,1625,2580]

    # correct for ratios
    # W = (W*H/ratio)^(1/2)
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    # H = W * ratio
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # [0, 0, W, H] -> [-W/2, -H/2, W/2, H/2]

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size.

    Args:
        shape: Shape to shift the anchors over.
        stride: Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    """
    x = np.array([0, 1, 2])
    y = np.array([0, 1])
    
    X, Y = np.meshgrid(x,y)
    print(X)
    print(Y)
    
    [[0 1 2]
     [0 1 2]]
    [[0 0 0]
     [1 1 1]] 
    """

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # a.ravel(): flatten
    # np.vastack: row
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


if __name__ == "__main__":
    # org 128*128 -> C1 64*64 -> C2 32*32
    # -> C3 P3 16*16 -> C4 P4 8*8 -> C5 P5 4*4 -> C6 P6 2*2 -> C7 P7 1*1
    # (16*16+8*8+4*4+2*2+1*1)*9=(256+64+16+4+1)*9=341*9=3069
    image = np.random.rand(1, 1, 128, 128)
    anchors = RetinaNetAnchors()
    anchors = anchors(image)
    print(anchors.shape)
    # torch.Size([1, 3069, 4])
