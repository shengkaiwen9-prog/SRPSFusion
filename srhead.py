import torch
import torch.nn as nn

def pixelshuffle(in_c, out_c, upscale=2):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c * (upscale ** 2), 3, 1, 1),
        nn.PixelShuffle(upscale)
    )

class SRHead(nn.Module):
    """2× 上采样，IMDN 轻量残差，仅 0.4 M 参数"""
    def __init__(self, in_ch=1, nf=32):
        super().__init__()
        self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
        body = []
        for _ in range(4):
            body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)
        self.up   = pixelshuffle(nf, nf, 2)
        self.tail = nn.Conv2d(nf, 1, 3, 1, 1)

    def forward(self, x):
        res = self.head(x)
        x   = self.body(res) + res
        x   = self.up(x)
        return self.tail(x)