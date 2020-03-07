# encoding: utf-8
# @author kwp
# @created 2020-3-6

import torch
import torch.nn as nn
import torch.nn.functional as F


# https://blog.csdn.net/qq_27261889/article/details/86304061
# https://blog.csdn.net/qq_36401512/article/details/88663352
# https://blog.csdn.net/lucky_kai/article/details/95348349

class PaddingLayer(nn.Module):
    """
    没有padding层，定义一个
    """
    def __init__(self, padding):
        super(PaddingLayer, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding, mode="replicate")


class UNet(nn.Module):
    def __init__(self, in_channels=1, classes=2):
        super(UNet, self).__init__()
        self._init_contract_blocks(in_channels)
        self._init_expansive_blocks(classes)

    def forward(self, x):
        # Contract
        c1 = self.contract_block1(x)
        m1 = self.max_pool1(c1)

        c2 = self.contract_block2(m1)
        m2 = self.max_pool2(c2)

        c3 = self.contract_block3(m2)
        m3 = self.max_pool3(c3)

        c4 = self.contract_block4(m3)
        m4 = self.max_pool4(c4)

        c5 = self.contract_block5(m4)
        # Expand
        u1 = self.up_conv1(c5)
        e1 = self.expansive_block1(self._concat_tensor(u1,c4))

        u2 = self.up_conv2(e1)
        e2 = self.expansive_block2(self._concat_tensor(u2,c3))

        u3 = self.up_conv3(e2)
        e3 = self.expansive_block3(self._concat_tensor(u3, c2))

        u4 = self.up_conv4(e3)
        e4 = self.expansive_block4(self._concat_tensor(u4, c1))

        x = self.output(e4)
        return x

    def _init_contract_blocks(self, in_channels):
        # 定义收缩部分
        self.contract_block1 = self._g_block(in_channels, 64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        for i in range(2, 5): # from 64 to 1024
            setattr(self, "contract_block{}".format(i), self._g_block(16 * (2 ** i), 2 * 16 * (2 ** i)))
            setattr(self, "max_pool{}".format(i), nn.MaxPool2d(kernel_size=2, stride=2))

        self.contract_block5 = self._g_block(512, 1024)   # bottom layers

    def _init_expansive_blocks(self, classes):
        for i in range(1, 5): # from 1024 to 128
            setattr(self, "up_conv{}".format(i), self._g_up_conv(2 * 1024 // (2 ** i)))
            setattr(self, "expansive_block{}".format(i), self._g_block(2 * 1024 // (2 ** i), 1024 // (2 ** i)))

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=classes, kernel_size=1),
            nn.Softmax2d()
        ) if classes != 1 else nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=classes, kernel_size=1),
            nn.Sigmoid()
        )

    def _g_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _g_up_conv(self, in_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            PaddingLayer([1,0,1,0]),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2),

            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True),
        )

    def _g_max_pool(self):
        return nn.MaxPool2d(kernel_size=2, stride=2)

    def _concat_tensor(self, tensor1, tensor2):
        diffY = torch.tensor([tensor2.size()[2] - tensor1.size()[2]])
        diffX = torch.tensor([tensor2.size()[3] - tensor1.size()[3]])
        tensor1 = F.pad(tensor1, [diffX // 2, diffX - diffX // 2,  diffY // 2, diffY - diffY // 2])
        return torch.cat([tensor2, tensor1], dim=1)


if __name__ == '__main__':
    net = UNet(in_channels=1, classes=2)
    print(net)
    data = torch.randn(size=(1, 1, 572, 572))
    out = net(data)
    print(out)