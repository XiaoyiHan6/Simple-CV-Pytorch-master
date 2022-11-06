import os
import torch
import torch.nn as nn
from models.detection.SSD.utils.l2norm import L2Norm
from models.detection.SSD.box_head.inference import Detect
from models.detection.SSD.anchor.prior_box import PriorBox
from models.detection.SSD.backbone.vgg import vgg, add_extras
from models.detection.SSD.box_head.box_predictor import multibox


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, cfg, base, extras, head, batch_norm=False):
        super(SSD, self).__init__()
        self.phase = phase
        self.cfg = cfg
        self.bn = batch_norm
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.size = cfg['DATA']['SIZE']
        self.num_classes = cfg['DATA']['NUM_CLASSES']

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == "test":
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes=self.num_classes,
                                 top_k=cfg['TEST']['TOP_K'],
                                 conf_thresh=cfg['TEST']['CONF_THRESH'],
                                 nms_thresh=cfg['TEST']['NMS_THRESH'])

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu

        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if self.bn:
                if k % 6 == 5:
                    sources.append(x)
            else:
                if k % 4 == 3:
                    sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect.forward(
                loc.view(loc.size(0), -1, 4),  # loc preds [batch, num_boxes, 4] -> [16, 8732, 4]
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds [batch, num_boxes, num_classes]
                self.priors.type(type(x.data))  # default boxes [8732, 4]
            )
        else:  # train
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}


def build_ssd(phase, cfg):
    size = cfg['DATA']['SIZE']
    num_classes = cfg['DATA']['NUM_CLASSES']
    batch_norm = cfg['MODEL']['BATCH_NORM']
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(batch_norm), num_classes, batch_norm)
    return SSD(phase, cfg, base_, extras_, head_, batch_norm)


if __name__ == "__main__":
    from options.detection.SSD.train_options import cfg

    print("train")
    # train
    ssd = build_ssd(phase='train', cfg=cfg)
    x = torch.randn(16, 3, 300, 300)
    y = ssd(x)
    print("Loc    shape: ", y[0].shape)
    print("Conf   shape: ", y[1].shape)
    print("Priors shape: ", y[2].shape)

    print("---------------------------------------")

    print("test")
    # test
    ssd_test = build_ssd(phase='test', cfg=cfg)
    input = torch.randn(16, 3, 300, 300)
    out = ssd_test(input)
    print("out.shape:", out.shape)
