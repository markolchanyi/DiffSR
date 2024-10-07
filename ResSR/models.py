import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, num_filters, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=0.1):
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
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        x = x + res
        return x

# Inspired by EDSR
class SRmodel(nn.Module):
    def __init__(self, num_filters, num_residual_blocks, kernel_size, use_global_residual=False):
        super().__init__()

        pad = (kernel_size // 2)

        m_head = [nn.Conv3d(1, num_filters, kernel_size,  padding=pad)]

        m_body = []
        for _ in range(num_residual_blocks):
            m_body.append(ResBlock(num_filters, kernel_size))
        m_body.append(nn.Conv3d(num_filters, num_filters, kernel_size,  padding=pad))


        # define tail module
        m_tail = [
            nn.Conv3d(num_filters, 1, kernel_size, padding=pad)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.use_global_residual = use_global_residual


    def forward(self, x):

        x2 = self.head(x)
        res = self.body(x2)
        x3 = x2 + res
        x4 = self.tail(x3)
        if self.use_global_residual:
            x5 = x4 + x
        else:
            x5 = x4

        return x5


