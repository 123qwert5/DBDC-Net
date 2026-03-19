import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 的平滑变体)
    相比 L1 Loss，它对异常值更鲁棒，能帮你在 PSNR 上多榨出 0.1~0.2 dB。
    """

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


class DBDCLoss(nn.Module):
    # === 【核心修改 1】新增 use_cycle 开关，默认 True 保证你的完全体不受影响 ===
    def __init__(self, use_cycle=True):
        super(DBDCLoss, self).__init__()
        self.criterion = CharbonnierLoss()
        self.use_cycle = use_cycle

    def forward(self, pred_bg, pred_rain, gt_bg, gt_rain, input_rain):
        """
        [严格适配 train.py 的 5 参数接口]
        """

        # 1. 基础重构损失 (显式监督)
        loss_bg = self.criterion(pred_bg, gt_bg)
        loss_rain = self.criterion(pred_rain, gt_rain)

        # 先将基础的两个损失加起来
        loss_total = loss_bg + loss_rain

        # 2. === [核心修改 2] 只有当 use_cycle 为 True 时，才计算并加上物理约束 ===
        if self.use_cycle:
            # 物理约束: O = B + R
            reconstructed_input = pred_bg + pred_rain
            loss_cycle = self.criterion(reconstructed_input, input_rain)
            # 加上 0.5 权重的 Cycle Loss
            loss_total = loss_total + 0.5 * loss_cycle

        return loss_total