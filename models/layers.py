
# ---------------------------------------------------------------------------------------第二个，，，，上面的不要，，，1，9------------------------------
import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    """ 像素注意力 (Pixel Attention) """

    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.pa(x)


class CALayer(nn.Module):
    """ 通道注意力 (Channel Attention) """

    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        return x * self.ca(y)


# =========================================================
# [修正] 对应开题报告中的 "FFA Block" (背景分支)
# 之前的名字叫 StandardBlock，现在改名为 FFABlock
# 并在内部强化了 "特征融合" (Feature Fusion) 的概念
# =========================================================
class FFABlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(FFABlock, self).__init__()

        # 特征提取路
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

        # 注意力路 (Feature Fusion Attention)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        # 局部残差路径
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)

        # 注意力融合
        # 这符合 FFA-Net 的核心思想：利用注意力机制融合特征
        res = self.calayer(res)
        res = self.palayer(res)

        # 全局残差
        res += x
        return res


# =========================================================
# [保持] 对应开题报告中的 "方向感知卷积" (雨纹分支)
# =========================================================
class RainBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(RainBlock, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)

        # Direction Conv (方向感知)
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1))
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0))
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x

        # Direction Perception
        h = self.conv_h(res)
        v = self.conv_v(res)
        cat = torch.cat([h, v], dim=1)
        res = self.fusion(cat)

        res = self.calayer(res)
        res = self.palayer(res)

        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks, block_type):
        super(Group, self).__init__()
        modules = [block_type(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res