
import torch
import torch.nn as nn
from models.layers import default_conv, Group, FFABlock, RainBlock


# === 【新增】用于消融实验的 Baseline 普通残差块 ===
class StandardBlock(nn.Module):
    """
    当关闭 FFA 或 DAC 时，网络退化为使用这个最普通的残差块。
    这才能证明你的 FFA 和 DAC 是真有用的！
    """

    def __init__(self, conv, dim, kernel_size, **kwargs):
        super(StandardBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size)

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        return x + res


class CAM(nn.Module):
    # ... (你的 CAM 代码保持完全不变) ...
    def __init__(self, n_feats):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_bg = nn.Sequential(
            nn.Linear(n_feats, n_feats // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats // 8, n_feats, bias=False),
            nn.Sigmoid()
        )
        self.fc_rain = nn.Sequential(
            nn.Linear(n_feats, n_feats // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats // 8, n_feats, bias=False),
            nn.Sigmoid()
        )

    def forward(self, bg_feat, rain_feat):
        b, c, _, _ = bg_feat.size()
        y_bg = self.avg_pool(bg_feat).view(b, c)
        y_rain = self.avg_pool(rain_feat).view(b, c)

        att_bg = self.fc_bg(y_bg).view(b, c, 1, 1)
        att_rain = self.fc_rain(y_rain).view(b, c, 1, 1)

        sum_att = att_bg + att_rain + 1e-6
        norm_att_bg = att_bg / sum_att
        norm_att_rain = att_rain / sum_att

        return bg_feat * norm_att_bg, rain_feat * norm_att_rain


class DBDCNet(nn.Module):
    # === 【核心修改 1】加入三个消融开关，默认全部开启 ===
    def __init__(self, gps=3, blocks=4, conv=default_conv,
                 use_ffa=True, use_dac=True, use_cam=True):
        super(DBDCNet, self).__init__()
        self.gps = gps
        self.dim = 64
        self.use_cam = use_cam
        kernel_size = 3

        self.head = nn.Sequential(
            conv(3, self.dim, kernel_size),
            conv(self.dim, self.dim, kernel_size)
        )

        # === 【核心修改 2】根据开关决定背景分支的 Block ===
        bg_block = FFABlock if use_ffa else StandardBlock
        self.bg_branch = nn.ModuleList([
            Group(conv, self.dim, kernel_size, blocks=blocks, block_type=bg_block)
            for _ in range(gps)
        ])

        # === 【核心修改 3】根据开关决定雨纹分支的 Block ===
        rain_block = RainBlock if use_dac else StandardBlock
        self.rain_branch = nn.ModuleList([
            Group(conv, self.dim, kernel_size, blocks=blocks, block_type=rain_block)
            for _ in range(gps)
        ])

        # 如果开启 CAM 才初始化，节省内存
        if self.use_cam:
            self.cams = nn.ModuleList([
                CAM(self.dim) for _ in range(gps)
            ])

        self.fusion_bg = conv(self.dim * gps, self.dim, 1)
        self.fusion_rain = conv(self.dim * gps, self.dim, 1)

        self.tail_bg = nn.Sequential(
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        )
        self.tail_rain = nn.Sequential(
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        )

    def forward(self, x):
        x_feat = self.head(x)
        bg_feats = []
        rain_feats = []
        curr_bg = x_feat
        curr_rain = x_feat

        for i in range(self.gps):
            curr_bg = self.bg_branch[i](curr_bg)
            curr_rain = self.rain_branch[i](curr_rain)

            # === 【核心修改 4】如果 CAM 关闭，则跳过特征解耦交互 ===
            if self.use_cam:
                curr_bg, curr_rain = self.cams[i](curr_bg, curr_rain)

            bg_feats.append(curr_bg)
            rain_feats.append(curr_rain)

        out_bg_feat = self.fusion_bg(torch.cat(bg_feats, dim=1))
        out_rain_feat = self.fusion_rain(torch.cat(rain_feats, dim=1))

        pred_bg = self.tail_bg(out_bg_feat)
        pred_rain = self.tail_rain(out_rain_feat)

        return pred_bg, pred_rain