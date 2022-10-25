import torch
import numpy as np


def collate(data):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    imgs = [d['img'] for d in data]
    targets = [d['annot'] for d in data]
    scales = [d['scale'] for d in data]

    # skimage shape[0]=h,shape[1]=w
    # B, H, W, 3-> ... -> B, 3, W, H
    heights = [int(img.shape[0]) for img in imgs]
    widths = [int(img.shape[1]) for img in imgs]
    batch_size = len(imgs)

    max_height = np.array(heights).max()
    max_width = np.array(widths).max()
    # B, H, W, 3
    padded_imgs = np.zeros((batch_size, max_height, max_width, 3),
                           dtype=np.float32)

    for i, img in enumerate(imgs):
        padded_imgs[i, :img.shape[0], :img.shape[1], :] = img
    padded_imgs = torch.from_numpy(padded_imgs)
    # B, 3, H, W
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    num_annots = max(target.shape[0] for target in targets)
    if num_annots > 0:
        annots = np.ones((len(targets), num_annots, 5), dtype=np.float32) * (-1)
        for i, target in enumerate(targets):
            if target.shape[0] > 0:
                annots[i, :target.shape[0], :] = target
    else:
        annots = np.ones((len(targets), 1, 5), dtype=np.float32) * (-1)

    annots = torch.from_numpy(annots)

    scales = np.array(scales, dtype=np.float32)
    scales = torch.from_numpy(scales)

    return {'img': padded_imgs, 'annot': annots, 'scale': scales}
