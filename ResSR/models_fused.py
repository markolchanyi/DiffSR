import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, num_filters, kernel_size, bias=True, bn=True, act=nn.ELU(alpha=1.0, inplace=False), res_scale=0.1):
        super(ResBlock, self).__init__()
        pad = (kernel_size // 2)
        m = []
        for i in range(2):
            m.append(nn.Conv3d(num_filters, num_filters, kernel_size, bias=bias, padding=pad))
            if bn:
                m.append(nn.BatchNorm3d(num_filters))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.se = SEBlock(num_filters)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = self.se(res)
        x = x + res
        return x


class SRmodel(nn.Module):
    def __init__(self,
                 num_filters,
                 num_residual_blocks,
                 kernel_size,
                 use_global_residual=False,
                 num_filters_l0=32,
                 num_residual_blocks_l0=3,
                 l0_channels=1,
                 l1plus_channels=27):
        super().__init__()
        pad = (kernel_size // 2)

        self.l0_channels = l0_channels
        self.l1plus_channels = l1plus_channels
        self.use_global_residual = use_global_residual

        # l=0 HEAD
        m_head_l0 = [nn.Conv3d(self.l0_channels, num_filters_l0, kernel_size, padding=pad)]

        # l=0 BODY (Keep ReLu)
        m_body_l0 = []
        for _ in range(num_residual_blocks_l0):
            m_body_l0.append(ResBlock(num_filters_l0, kernel_size, act=nn.ReLU(inplace=False)))
        m_body_l0.append(nn.Conv3d(num_filters_l0, num_filters_l0, kernel_size, padding=pad))

        # l≥1 HEAD
        m_head_l1plus = [nn.Conv3d(self.l1plus_channels, num_filters, kernel_size, padding=pad)]

        # l≥1 BODY
        m_body_l1plus = []
        for _ in range(num_residual_blocks):
            m_body_l1plus.append(ResBlock(num_filters, kernel_size, act=nn.ELU(alpha=1.0, inplace=False)))
        m_body_l1plus.append(nn.Conv3d(num_filters, num_filters, kernel_size, padding=pad))

        # Fusion: Combine l=0 and l≥1 streams
        m_fusion = [
            nn.Conv3d(num_filters_l0 + num_filters, num_filters, kernel_size=1),
            nn.ELU(alpha=1.0, inplace=True)
        ]

        # Tail: Final output to 28 channels
        m_tail = [nn.Conv3d(num_filters, 28, kernel_size, padding=pad)]

        self.head_l0 = nn.Sequential(*m_head_l0)
        self.body_l0 = nn.Sequential(*m_body_l0)

        self.head_l1plus = nn.Sequential(*m_head_l1plus)
        self.body_l1plus = nn.Sequential(*m_body_l1plus)

        self.fusion = nn.Sequential(*m_fusion)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # Split input
        x_l0 = x[:, 0:1, ...]     # l=0 channel
        x_l1plus = x[:, 1:, ...]  # l≥1 channels

        # Process l=0 stream
        x_l0_head = self.head_l0(x_l0)
        x_l0_body = self.body_l0(x_l0_head)
        x_l0_out = x_l0_head + x_l0_body

        # Process l≥1 stream
        x_l1plus_head = self.head_l1plus(x_l1plus)
        x_l1plus_body = self.body_l1plus(x_l1plus_head)
        x_l1plus_out = x_l1plus_head + x_l1plus_body

        # Fuse
        fused = torch.cat([x_l0_out, x_l1plus_out], dim=1)
        fused = self.fusion(fused)

        # Tail
        x_out = self.tail(fused)

        # Global residual
        if self.use_global_residual:
            x_out = x_out + x

        return x_out

