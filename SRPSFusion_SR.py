# SRPSFusion_withSR.py
# 基于你原始 SRPSFusion 文件的内容，新增了轻量级 SRNet 并将其注册到模型
# 仅做了必要的增补（SRNet）和 forward 返回项增加，不改动原有模块实现

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18
from MSSA import MSCM, Block

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class S2M(nn.Module):
    def __init__(self, channels, r=4):
        super(S2M, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_block = nn.BatchNorm2d(channels)
    def forward(self, x):
        # spatial attention
        local_w = self.local_att(x)  ## local attention
        ## channel attention
        global_w = self.global_att(x)
        mask = self.sigmoid(local_w * global_w)
        masked_feature = mask * x
        output = self.conv_block(masked_feature)
        return output

class SFP(nn.Module):
    def __init__(self, channels, img=False):
        super(SFP, self).__init__()
        # self.mask = S2M(channels[0])
        # self.mask = TVSGA(channels[0])
        self.conv_block = nn.Sequential(
            FAConv(channels[0], channels[0], kernel_size=3),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
            nn.Sigmoid(),
            # nn.Tanh(),
        )
    def forward(self, x):
        # x = self.mask(x)
        # return (self.conv_block(x) + 1) / 2
        return self.conv_block(x)

class Prediction(nn.Module):
    def __init__(self, channels):
        super(Prediction, self).__init__()
        # self.mask = S2M(channels[0])
        # self.mask = TVSGA(channels[0])
        self.conv_block = nn.Sequential(
            FAConv(channels[0], channels[0], kernel_size=3),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
            # FAConv(channels[0], channels[1], kernel_size=3),
            nn.Sigmoid(),
            # nn.Tanh(),
        )
    def forward(self, x):
        # x = self.mask(x)
        # return (self.conv_block(x) + 1) / 2
        return self.conv_block(x)

# TVSGA
class BasicConv_(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False,
    ):
        super(BasicConv_, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )

class SparseAct(nn.Module):
    def __init__(self, tau=0.05, mode="soft"):  # tau 越大越稀疏
        super().__init__()
        self.tau = tau
        self.mode = mode

    def forward(self, x):
        if self.mode == "hard":
            # |x| < tau -> 0
            return F.hardshrink(x, lambd=self.tau)
        else:
            # |x| < tau -> 0, 外侧线性缩小（更平滑）
            return F.softshrink(x, lambd=self.tau)

class FE(nn.Module):
    def __init__(self, tau=0.05, sparse_mode="soft"):
        super(FE, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()

        self.sparse = SparseAct(tau=tau, mode=sparse_mode)

        # 把原来的 spatial 拆开，便于在每一步后做稀疏
        self.dw = BasicConv_(2, 2, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True, groups=2)
        self.conv_h = nn.Conv2d(2, 2, kernel_size=(3, 1), padding=(1, 0), groups=2)
        self.conv_w = nn.Conv2d(2, 2, kernel_size=(1, 3), padding=(0, 1), groups=2)

        self.ch = nn.Conv2d(2, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x_compress = self.compress(x)  # [B,2,H,W]

        y = self.dw(x_compress)
        y = self.sparse(y)             # 稀疏1：提取过程就开始变稀疏

        y = self.conv_h(y)
        y = self.sparse(y)             # 稀疏2

        y = self.conv_w(y)
        y = self.sparse(y)             # 稀疏3

        y = y + x_compress             # 残差保留基本结构
        x_out = self.ch(y)             # [B,1,H,W]

        # 门控也做“稀疏化”：让 scale 大量接近 0（而不是全是 0~1 的稠密值）
        scale = torch.sigmoid(x_out)
        scale = self.sparse(scale)     # 关键：scale 自身也变稀疏（很多位置变 0）

        return x * scale

def compute_featuremap_entropy(feature_maps: torch.Tensor, bins: int = 50) -> torch.Tensor:
    B, C, H, W = feature_maps.shape
    flattened = feature_maps.view(B, -1)
    min_vals = flattened.min(dim=-1, keepdim=True)[0]
    max_vals = flattened.max(dim=-1, keepdim=True)[0]
    normalized = (flattened - min_vals) / (max_vals - min_vals + 1e-9)
    bin_indices = torch.floor(normalized * bins).long()
    bin_indices = torch.clamp(bin_indices, 0, bins - 1)
    hist = torch.zeros((B, bins), device=feature_maps.device)
    hist.scatter_add_(
        dim=-1,
        index=bin_indices,
        src=torch.ones_like(bin_indices, dtype=torch.float32)
    )
    prob = hist / (C * H * W) + 1e-9
    entropy = - (prob * torch.log2(prob)).sum(dim=-1)
    return entropy
class TVSGA(nn.Module):
    def __init__(self, gate_channels):
        super(TVSGA, self).__init__()
        self.ChannelGateH = FE()
        self.ChannelGateW = FE()
        self.SpatialGate = FE()
    def forward(self, x):
        B, C, _, _ = x.size()
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out = self.SpatialGate(x)
        w1 = compute_featuremap_entropy(x_out)
        w2 = compute_featuremap_entropy(x_out11)
        w3 = compute_featuremap_entropy(x_out21)
        w = w1 + w2 + w3
        x_out = ((1+(w1 / w)).view(B, 1, 1, 1)) * x_out + (((w2 / w)).view(B, 1, 1, 1)) * x_out11 + (((w3 / w)).view(B, 1, 1, 1)) * x_out21
        return x_out

class OutS(nn.Module):
    def __init__(self, channels=64):
        super(OutS, self).__init__()
        self.SA = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x1, x2):
        out = self.SA(x1 + x2)
        out = torch.max(out, 1)[0].unsqueeze(1) + torch.mean(out, 1).unsqueeze(1)
        return out
class OutC(nn.Module):
    def __init__(self, channels=64):
        super(OutC, self).__init__()
        self.CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels//2),
            nn.Sigmoid(),
        )
    def forward(self, x1, x2):
        out = self.CA(torch.cat([x1, x2], dim=1))
        return out
class MMFM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MMFM, self).__init__()
        self.SE = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            TVSGA(channels),
        )

        self.CE = nn.Sequential(
            nn.Conv2d(channels*2+1, channels*2+1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels*2+1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels*2+1, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            TVSGA(channels),
        )

        self.FuseC = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            TVSGA(channels),
        )

        self.FuseS = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            TVSGA(channels),
        )

        self.OutC = OutC(channels*2)
        self.OutS = OutS(channels)

        self.SA1 = Block(channels, num_heads=8)
        self.SA2 = Block(channels, num_heads=8)
    def forward(self, x1, x2, x3):
        B, c, _, _ = x1.shape

        x3 = F.interpolate(x3, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
        FuseS1 = self.SE(x1 + x2 + x3)
        FuseC1 = self.CE(torch.cat([x1, x2, x3], dim=1))

        FuseSC12 = self.FuseC(torch.cat([FuseS1, FuseC1], dim=1))
        FuseCS12 = self.FuseS(FuseS1 + FuseC1)

        outC = self.OutC(FuseSC12, FuseCS12)
        outS = self.OutS(FuseSC12, FuseCS12)

        out = (self.SA1(x1) + self.SA2(x2) + (x3)) * outC + outS

        return out
class MMFM_(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MMFM_, self).__init__()
        self.SE = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            TVSGA(channels),
        )

        self.CE = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            TVSGA(channels),
        )

        self.FuseC = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            TVSGA(channels),
        )

        self.FuseS = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            TVSGA(channels),
        )

        self.OutC = OutC(channels*2)
        self.OutS = OutS(channels)

        self.SA1 = Block(channels, num_heads=8)
        self.SA2 = Block(channels, num_heads=8)
    def forward(self, x1, x2):
        B, c, _, _ = x1.shape
        FuseS1 = self.SE(x1 + x2)
        FuseC1 = self.CE(torch.cat([x1, x2], dim=1))

        FuseSC12 = self.FuseC(torch.cat([FuseS1, FuseC1], dim=1))
        FuseCS12 = self.FuseS(FuseS1 + FuseC1)

        outC = self.OutC(FuseSC12, FuseCS12)
        outS = self.OutS(FuseSC12, FuseCS12)

        out = (self.SA1(x1) + self.SA2(x2)) * outC + outS
        return out

class FAConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1, bias=True, with_gate=True, pool=False):
        super(FAConv, self).__init__()
        self.input_linear_p = nn.Conv2d(input_dim, output_dim // 4, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        self.input_linear_g = nn.Conv2d(input_dim, (output_dim - output_dim // 2), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.activation = nn.GELU()
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
        if pool:
            self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, src):
        if not hasattr(self, 'pool'):
            src = src
        else:
            src = self.pool(src)
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)

        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g), dim=1)
        return output

# ------------------ 新增：轻量级 SR 模块 ------------------
class SRBlock(nn.Module):
    """一个简单的残差块用于 SR backbone"""
    def __init__(self, channels):
        super(SRBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out

class SRNet(nn.Module):
    """
    轻量级可重复的 x2 上采超分模块。
    输入通道默认为1（灰度/热图），可通过 repeat_blocks 控制网络深度。
    up_times 指定需要把输入上采多少倍（必须是 2 的幂），通过重复 pixelshuffle x2 块实现。
    """
    def __init__(self, in_ch=1, feat_channels=64, repeat_blocks=4, up_times=1):
        super(SRNet, self).__init__()
        assert up_times >= 1 and (up_times & (up_times - 1)) == 0, "up_times must be power of 2 (1,2,4,...)"
        self.head = nn.Conv2d(in_ch, feat_channels, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(*[SRBlock(feat_channels) for _ in range(repeat_blocks)])
        # 每个 pixelshuffle 块把特征上采 2x
        self.up_blocks = nn.ModuleList()
        times = int(up_times.bit_length() - 1)  # e.g., up_times=4 -> times=2 (two x2 blocks)
        for _ in range(times):
            # 通过 conv 输出 feat_channels * 4 然后 PixelShuffle(2)
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(feat_channels, feat_channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                )
            )
        self.tail = nn.Conv2d(feat_channels, in_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # x: [B, C, H, W]  (通常 C==1)
        feat = self.head(x)
        feat = self.body(feat)
        for blk in self.up_blocks:
            feat = blk(feat)
        out = self.tail(feat)
        # 输出范围与原预测一致（假设预测使用 Tanh->( +1)/2 => [0,1]）
        out = torch.sigmoid(out)  # 保证在 [0,1]
        return out

# ------------------ 修改：在 SRPSFusion 中初始化 SRNet 并在 forward 生成上采版本 ------------------
class SRPSFusion(nn.Module):
    def __init__(self, n_classes, sr_repeat_blocks=4):
        super(SRPSFusion, self).__init__()
        # self.dims = [4, 8, 12, 16, 20]
        self.dims = [4, 8, 16, 24, 32, 40]

        resnet = resnet18(pretrained=False, in_channel=3)
        self.firstconv_rgb = resnet.conv1
        self.firstbn_rgb = resnet.bn1
        self.firstrelu_rgb = resnet.relu
        self.firstmaxpool_rgb = resnet.maxpool
        self.encoder_rgb1 = resnet.layer1
        self.encoder_rgb2 = resnet.layer2
        self.encoder_rgb3 = resnet.layer3
        self.encoder_rgb4 = resnet.layer4

        resnet2 = resnet18(pretrained=False, in_channel=1)
        self.firstconv_ir = resnet2.conv1
        self.firstbn_ir = resnet2.bn1
        self.firstrelu_ir = resnet2.relu
        self.firstmaxpool_ir = resnet2.maxpool
        self.encoder_ir1 = resnet2.layer1
        self.encoder_ir2 = resnet2.layer2
        self.encoder_ir3 = resnet2.layer3
        self.encoder_ir4 = resnet2.layer4

        self.fuse5 = MMFM_(self.dims[4])
        self.fuse4 = MMFM(self.dims[3])
        self.fuse3 = MMFM(self.dims[2])
        self.fuse2 = MMFM(self.dims[1])
        self.fuse1 = MMFM(self.dims[0])

        self.pred_fusion = Prediction([self.dims[0], 1])
        self.pred_fusion2 = Prediction([self.dims[0], 1])
        self.pred_fusion3 = Prediction([self.dims[1], 1])
        self.pred_fusion4 = Prediction([self.dims[2], 1])
        self.pred_fusion5 = Prediction([self.dims[3], 1])

        self.pred_vi = nn.Sequential(
            nn.BatchNorm2d(self.dims[0]),
            FAConv(self.dims[0], self.dims[0], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[0]),
            FAConv(self.dims[0], self.dims[0], kernel_size=3),
            SFP([self.dims[0], 1])
        )
        self.pred_ir = nn.Sequential(
            nn.BatchNorm2d(self.dims[0]),
            FAConv(self.dims[0], self.dims[0], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[0]),
            FAConv(self.dims[0], self.dims[0], kernel_size=3),
            SFP([self.dims[0], 1])
        )
        # ========== 新增：各层重构头 ==========
        # 第二层重构头
        self.pred_vi2 = nn.Sequential(
            nn.BatchNorm2d(self.dims[1]),
            FAConv(self.dims[1], self.dims[1], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[1]),
            FAConv(self.dims[1], self.dims[1], kernel_size=3),
            SFP([self.dims[1], 1])
        )
        self.pred_ir2 = nn.Sequential(
            nn.BatchNorm2d(self.dims[1]),
            FAConv(self.dims[1], self.dims[1], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[1]),
            FAConv(self.dims[1], self.dims[1], kernel_size=3),
            SFP([self.dims[1], 1])
        )
        # 第三层重构头
        self.pred_vi3 = nn.Sequential(
            nn.BatchNorm2d(self.dims[2]),
            FAConv(self.dims[2], self.dims[2], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[2]),
            FAConv(self.dims[2], self.dims[2], kernel_size=3),
            SFP([self.dims[2], 1])
        )
        self.pred_ir3 = nn.Sequential(
            nn.BatchNorm2d(self.dims[2]),
            FAConv(self.dims[2], self.dims[2], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[2]),
            FAConv(self.dims[2], self.dims[2], kernel_size=3),
            SFP([self.dims[2], 1])
        )
        # 第四层重构头
        self.pred_vi4 = nn.Sequential(
            nn.BatchNorm2d(self.dims[3]),
            FAConv(self.dims[3], self.dims[3], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[3]),
            FAConv(self.dims[3], self.dims[3], kernel_size=3),
            SFP([self.dims[3], 1])
        )
        self.pred_ir4 = nn.Sequential(
            nn.BatchNorm2d(self.dims[3]),
            FAConv(self.dims[3], self.dims[3], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[3]),
            FAConv(self.dims[3], self.dims[3], kernel_size=3),
            SFP([self.dims[3], 1])
        )
        # 第五层重构头
        self.pred_vi5 = nn.Sequential(
            nn.BatchNorm2d(self.dims[4]),
            FAConv(self.dims[4], self.dims[4], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[4]),
            FAConv(self.dims[4], self.dims[4], kernel_size=3),
            SFP([self.dims[4], 1])
        )
        self.pred_ir5 = nn.Sequential(
            nn.BatchNorm2d(self.dims[4]),
            FAConv(self.dims[4], self.dims[4], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[4]),
            FAConv(self.dims[4], self.dims[4], kernel_size=3),
            SFP([self.dims[4], 1])
        )

        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.Decoder5 = nn.Sequential(
            nn.BatchNorm2d(self.dims[4]),
            FAConv(self.dims[4], self.dims[3], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[3]),
            FAConv(self.dims[3], self.dims[3], kernel_size=3),
        )
        self.Decoder4 = nn.Sequential(
            nn.BatchNorm2d(self.dims[3]),
            FAConv(self.dims[3], self.dims[2], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[2]),
            FAConv(self.dims[2], self.dims[2], kernel_size=3),
        )
        self.Decoder3 = nn.Sequential(
            nn.BatchNorm2d(self.dims[2]),
            FAConv(self.dims[2], self.dims[1], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[1]),
            FAConv(self.dims[1], self.dims[1], kernel_size=3),
        )
        self.Decoder2 = nn.Sequential(
            nn.BatchNorm2d(self.dims[1]),
            FAConv(self.dims[1], self.dims[0], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[0]),
            FAConv(self.dims[0], self.dims[0], kernel_size=3),
        )
        self.Decoder1 = nn.Sequential(
            nn.BatchNorm2d(self.dims[0]),
            FAConv(self.dims[0], self.dims[0], kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[0]),
            FAConv(self.dims[0], self.dims[0], kernel_size=3),
        )

        # -------- 新增 SRNet：用于将低分辨率融合图 upsample 到 原始分辨率 -----------
        # 我们使用单个共享的 SRNet（输入单通道，输出单通道），支持按需上采到 2^k
        # 不在这里预设 up_times（forward 时按输入图尺寸计算需要的倍数）
        self.sr_shared = SRNet(in_ch=1, feat_channels=64, repeat_blocks=sr_repeat_blocks, up_times=1)
        # -------------------------------------------------------------------------

    def forward(self, rgb, depth):
        rgb = rgb
        thermal = depth[:, :1, ...]  # 保持原来行为

        ######################################################################
        rgb1 = (self.firstrelu_rgb(self.firstbn_rgb(self.firstconv_rgb(rgb))))
        rgb2 = (self.encoder_rgb1(self.firstmaxpool_rgb(rgb1)))
        rgb3 = (self.encoder_rgb2(rgb2))
        rgb4 = (self.encoder_rgb3(rgb3))
        rgb5 = (self.encoder_rgb4(rgb4))

        thermal1 = (self.firstrelu_ir(self.firstbn_ir(self.firstconv_ir(thermal))))
        thermal2 = (self.encoder_ir1(self.firstmaxpool_ir(thermal1)))
        thermal3 = (self.encoder_ir2(thermal2))
        thermal4 = (self.encoder_ir3(thermal3))
        thermal5 = (self.encoder_ir4(thermal4))

        ## fused featrue
        fused_f5 = self.fuse5(rgb5, thermal5)
        pred5 = (self.Decoder5(fused_f5))
        fused_img5 = self.pred_fusion5(pred5)
        fused_f4 = self.fuse4(rgb4, thermal4, fused_img5) + self.up2x(pred5)
        pred4 = (self.Decoder4(fused_f4))
        fused_img4 = self.pred_fusion4(pred4)
        fused_f3 = self.fuse3(rgb3, thermal3, fused_img4) + self.up2x(pred4)
        pred3 = (self.Decoder3(fused_f3))
        fused_img3 = self.pred_fusion3(pred3)
        fused_f2 = self.fuse2(rgb2, thermal2, fused_img3) + self.up2x(pred3)
        pred2 = (self.Decoder2(fused_f2))
        fused_img2 = self.pred_fusion2(pred2)
        fused_f1 = self.fuse1(rgb1, thermal1, fused_img2) + self.up2x(pred2)
        pred1 = self.Decoder1(fused_f1)

        vi_img5 = self.pred_vi5(fused_f5)
        ir_img5 = self.pred_ir5(fused_f5)

        vi_img4 = self.pred_vi4(fused_f4)
        ir_img4 = self.pred_ir4(fused_f4)

        vi_img3 = self.pred_vi3(fused_f3)
        ir_img3 = self.pred_ir3(fused_f3)

        vi_img2 = self.pred_vi2(fused_f2)
        ir_img2 = self.pred_ir2(fused_f2)

        vi_img = self.pred_vi(fused_f1)
        ir_img = self.pred_ir(fused_f1)

        fused_img = self.pred_fusion(pred1)

        # ------------------ 生成 SR 上采后的融合图（上采到与输入 rgb 相同大小）------------------
        # 假设 fused_img (pred1) 已经是原始分辨率（与 rgb 相同）
        # 对 fused_img2/3/4/5 分别按需要上采：2->x2, 3->x4, 4->x8, 5->x16（按你的下采过程）
        # 使用共享 SRNet：通过重复 pixelshuffle 模块实现不同 up_times；
        device = fused_img.device
        H, W = rgb.size(2), rgb.size(3)

        # helper: 上采函数（计算需要的 up_times 并使用临时 SRNet）
        def upsample_via_shared_sr(x, target_h, target_w):
            # x: [B,1,h,w]
            in_h, in_w = x.size(2), x.size(3)
            if in_h == target_h and in_w == target_w:
                return x
            # 计算整数缩放比（假设 target=in_h * scale）
            scale_h = target_h // in_h
            scale_w = target_w // in_w
            assert scale_h == scale_w and (scale_h & (scale_h - 1)) == 0, "Expected equal power-of-2 scaling"
            up_times = scale_h
            # 动态构造临时 SRNet 或者调整共享 sr_shared 的 up_times：这里直接使用临时 SRNet 实例（复制共享权重）
            # 方案：创建新的 SRNet with same backbone depth, set its weights from shared but with appropriate up_times
            times = int(up_times.bit_length() - 1)
            # create an instance with required up_times
            sr = SRNet(in_ch=1, feat_channels=64, repeat_blocks=len(self.sr_shared.body), up_times=up_times).to(device)
            # try to copy weights from shared where possible (head, body, tail, up-block convs)
            try:
                # copy head
                sr.head.load_state_dict(self.sr_shared.head.state_dict())
                # copy body
                sr.body.load_state_dict(self.sr_shared.body.state_dict())
                # for up_blocks: if sr has more up_blocks than shared, copy available ones, others init randomly
                for i in range(min(len(sr.up_blocks), len(self.sr_shared.up_blocks))):
                    sr.up_blocks[i].load_state_dict(self.sr_shared.up_blocks[i].state_dict())
                # copy tail
                sr.tail.load_state_dict(self.sr_shared.tail.state_dict())
            except Exception:
                # 如果复制失败，不影响（保持初始化）
                pass
            out = sr(x)
            return out

        # fused_img2 is expected 1/2 size -> up x2
        fused_img2_sr = upsample_via_shared_sr(fused_img2, H, W)
        fused_img3_sr = upsample_via_shared_sr(fused_img3, H, W)
        fused_img4_sr = upsample_via_shared_sr(fused_img4, H, W)
        fused_img5_sr = upsample_via_shared_sr(fused_img5, H, W)

        # 重构图超分
        vi_img2_sr = upsample_via_shared_sr(vi_img2, H, W)
        vi_img3_sr = upsample_via_shared_sr(vi_img3, H, W)
        vi_img4_sr = upsample_via_shared_sr(vi_img4, H, W)
        vi_img5_sr = upsample_via_shared_sr(vi_img5, H, W)

        ir_img2_sr = upsample_via_shared_sr(ir_img2, H, W)
        ir_img3_sr = upsample_via_shared_sr(ir_img3, H, W)
        ir_img4_sr = upsample_via_shared_sr(ir_img4, H, W)
        ir_img5_sr = upsample_via_shared_sr(ir_img5, H, W)

        # 返回所有结果
        return (fused_img, vi_img, ir_img,
                fused_img2, fused_img3, fused_img4, fused_img5,
                fused_img2_sr, fused_img3_sr, fused_img4_sr, fused_img5_sr,
                vi_img2, vi_img3, vi_img4, vi_img5,
                ir_img2, ir_img3, ir_img4, ir_img5,
                vi_img2_sr, vi_img3_sr, vi_img4_sr, vi_img5_sr,
                ir_img2_sr, ir_img3_sr, ir_img4_sr, ir_img5_sr)
        return fused_img

