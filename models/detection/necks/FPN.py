import torch.nn as nn
import torch.nn.functional as F
import torch


class FPN(nn.Module):
    def __init__(self, C3_inplanes, C4_inplanes, C5_inplanes, planes=256):
        super(FPN, self).__init__()
        # planes = 256 channels
        self.P3_1 = nn.Conv2d(C3_inplanes, planes, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.P4_1 = nn.Conv2d(C4_inplanes, planes, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.P5_1 = nn.Conv2d(C5_inplanes, planes, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.P6 = nn.Conv2d(C5_inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.P7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1))

    def forward(self, inputs):
        [C3, C4, C5] = inputs
        P5 = self.P5_1(C5)
        P4 = self.P4_1(C4)
        P4 = F.interpolate(P5, size=(P4.shape[2], P4.shape[3]),
                           mode='nearest') + P4
        P3 = self.P3_1(C3)
        P3 = F.interpolate(P4, size=(P3.shape[2], P3.shape[3]),
                           mode='nearest') + P3
        P6 = self.P6(C5)
        P7 = self.P7(P6)

        P5 = self.P5_2(P5)
        P4 = self.P4_2(P4)
        P3 = self.P3_2(P3)

        del C3, C4, C5
        return [P3, P4, P5, P6, P7]


if __name__ == "__main__":
    # Img size 672*640 -> C1 168*160 -> C2 168*160
    # -> C3 84*80 -> C4 42*40 -> C5 21*20
    # -> P3 84*80 -> P4 42*40 -> P5 21*20 -> P6 11*10 -> P7 6*5
    C3 = torch.randn([2, 128 * 4, 84, 80])
    C4 = torch.randn([2, 256 * 4, 42, 40])
    C5 = torch.randn([2, 512 * 4, 21, 20])

    model = FPN(128 * 4, 256 * 4, 512 * 4, 256)
    out = model([C3, C4, C5])
    for i in range(len(out)):
        print(out[i].shape)
