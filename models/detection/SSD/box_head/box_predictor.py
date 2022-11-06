from torch import nn


def multibox(vgg, extra, num_classes, batch_norm=False):
    loc1 = nn.Conv2d(in_channels=vgg[21].out_channels, out_channels=4 * 4, kernel_size=3, padding=1)
    loc2 = nn.Conv2d(in_channels=vgg[-2].out_channels, out_channels=6 * 4, kernel_size=3, padding=1)

    conf1 = nn.Conv2d(in_channels=vgg[21].out_channels, out_channels=4 * num_classes, kernel_size=3, padding=1)
    conf2 = nn.Conv2d(in_channels=vgg[-2].out_channels, out_channels=6 * num_classes, kernel_size=3, padding=1)
    if batch_norm:
        loc3 = nn.Conv2d(in_channels=extra[3].out_channels, out_channels=6 * 4, kernel_size=3, padding=1)
        loc4 = nn.Conv2d(in_channels=extra[9].out_channels, out_channels=6 * 4, kernel_size=3, padding=1)
        loc5 = nn.Conv2d(in_channels=extra[15].out_channels, out_channels=4 * 4, kernel_size=3, padding=1)
        loc6 = nn.Conv2d(in_channels=extra[21].out_channels, out_channels=4 * 4, kernel_size=3, padding=1)

        conf3 = nn.Conv2d(in_channels=extra[3].out_channels, out_channels=6 * num_classes, kernel_size=3, padding=1)
        conf4 = nn.Conv2d(in_channels=extra[9].out_channels, out_channels=6 * num_classes, kernel_size=3, padding=1)
        conf5 = nn.Conv2d(in_channels=extra[15].out_channels, out_channels=4 * num_classes, kernel_size=3, padding=1)
        conf6 = nn.Conv2d(in_channels=extra[21].out_channels, out_channels=4 * num_classes, kernel_size=3, padding=1)
    else:

        loc3 = nn.Conv2d(in_channels=extra[2].out_channels, out_channels=6 * 4, kernel_size=3, padding=1)
        loc4 = nn.Conv2d(in_channels=extra[6].out_channels, out_channels=6 * 4, kernel_size=3, padding=1)
        loc5 = nn.Conv2d(in_channels=extra[10].out_channels, out_channels=4 * 4, kernel_size=3, padding=1)
        loc6 = nn.Conv2d(in_channels=extra[14].out_channels, out_channels=4 * 4, kernel_size=3, padding=1)

        conf3 = nn.Conv2d(in_channels=extra[2].out_channels, out_channels=6 * num_classes, kernel_size=3, padding=1)
        conf4 = nn.Conv2d(in_channels=extra[6].out_channels, out_channels=6 * num_classes, kernel_size=3, padding=1)
        conf5 = nn.Conv2d(in_channels=extra[10].out_channels, out_channels=4 * num_classes, kernel_size=3, padding=1)
        conf6 = nn.Conv2d(in_channels=extra[14].out_channels, out_channels=4 * num_classes, kernel_size=3, padding=1)

    loc_layers = [loc1, loc2, loc3, loc4, loc5, loc6]
    conf_layers = [conf1, conf2, conf3, conf4, conf5, conf6]
    return vgg, extra, (loc_layers, conf_layers)


if __name__ == '__main__':
    from models.detection.SSD.backbone.vgg import vgg, add_extras

    base = {
        '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
        '512': [],
    }

    vgg, extra, (l, c) = multibox(vgg(base['300'], 3),
                                  add_extras(True), 21, True)
    print("loc layers:")
    print(nn.Sequential(*l))
    print('---------------------------')
    print("conf layers:")
    print(nn.Sequential(*c))
