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
                 num_filters_nonang=32,
                 num_residual_blocks_nonang=3,
                 nonang_channels=2,
                 l1plus_channels=27):
        super().__init__()
        pad = (kernel_size // 2)

        self.nonang_channels = nonang_channels
        self.l1plus_channels = l1plus_channels
        self.use_global_residual = use_global_residual

        # l=0 HEAD
        m_head_nonang = [nn.Conv3d(self.nonang_channels, num_filters_nonang, kernel_size, padding=pad)]

        # l=0 BODY (Keep ReLu)
        m_body_nonang = []
        for _ in range(num_residual_blocks_nonang):
            m_body_nonang.append(ResBlock(num_filters_nonang, kernel_size, act=nn.ReLU(inplace=False)))
        m_body_nonang.append(nn.Conv3d(num_filters_nonang, num_filters_nonang, kernel_size, padding=pad))

        # l≥1 HEAD
        m_head_l1plus = [nn.Conv3d(self.l1plus_channels, num_filters, kernel_size, padding=pad)]

        # l≥1 BODY
        m_body_l1plus = []
        for _ in range(num_residual_blocks):
            m_body_l1plus.append(ResBlock(num_filters, kernel_size, act=nn.ELU(alpha=1.0, inplace=False)))
        m_body_l1plus.append(nn.Conv3d(num_filters, num_filters, kernel_size, padding=pad))

        # Fusion: Combine l=0 and l≥1 streams
        m_fusion = [
            nn.Conv3d(num_filters_nonang + num_filters, num_filters, kernel_size=1),
            nn.ELU(alpha=1.0, inplace=True)
        ]

        # Tail: Final output to 29 channels
        m_tail = [nn.Conv3d(num_filters, 29, kernel_size, padding=pad)]

        self.head_nonang = nn.Sequential(*m_head_nonang)
        self.body_nonang = nn.Sequential(*m_body_nonang)

        self.head_l1plus = nn.Sequential(*m_head_l1plus)
        self.body_l1plus = nn.Sequential(*m_body_l1plus)

        self.fusion = nn.Sequential(*m_fusion)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # Split input
        x_nonang = x[:, 0:2, ...]     # l=0 and lowb channel
        x_l1plus = x[:, 2:, ...]  # l≥1 channels

        # Process l=0/lowb stream
        x_nonang_head = self.head_nonang(x_nonang)
        x_nonang_body = self.body_nonang(x_nonang_head)
        x_nonang_out = x_nonang_head + x_nonang_body

        # Process l≥1 stream
        x_l1plus_head = self.head_l1plus(x_l1plus)
        x_l1plus_body = self.body_l1plus(x_l1plus_head)
        x_l1plus_out = x_l1plus_head + x_l1plus_body

        # Fuse
        fused = torch.cat([x_nonang_out, x_l1plus_out], dim=1)
        fused = self.fusion(fused)

        # Tail
        x_out = self.tail(fused)

        # Global residual
        if self.use_global_residual:
            x_out = x_out + x

        return x_out

