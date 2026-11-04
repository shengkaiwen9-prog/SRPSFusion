import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class mssa(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.temperature = nn.Parameter(torch.ones(num_heads // 4, 1, 1))

        self.v1 = nn.Linear(dim // 4, dim // 4, bias=qkv_bias)
        self.v2 = nn.Linear(dim // 4, dim // 4, bias=qkv_bias)
        self.v3 = nn.Linear(dim // 4, dim // 4, bias=qkv_bias)
        self.v4 = nn.Linear(dim // 4, dim // 4, bias=qkv_bias)
        self.qk1 = nn.Linear(dim // 4, dim, bias=qkv_bias)
        self.qk2 = nn.Linear(dim // 4, dim, bias=qkv_bias)
        self.qk3 = nn.Linear(dim // 4, dim, bias=qkv_bias)
        self.qk4 = nn.Linear(dim // 4, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool11 = nn.AdaptiveAvgPool2d(1)
            self.pool12 = nn.AdaptiveMaxPool2d(1)
            self.sr1 = nn.Conv2d(dim // 4, dim // 4, kernel_size=1, stride=1)
            self.pool21 = nn.AdaptiveAvgPool2d(3)
            self.pool22 = nn.AdaptiveMaxPool2d(3)
            self.sr2 = nn.Conv2d(dim // 4, dim // 4, kernel_size=1, stride=1)
            self.pool31 = nn.AdaptiveAvgPool2d(5)
            self.pool32 = nn.AdaptiveMaxPool2d(5)
            self.sr3 = nn.Conv2d(dim // 4, dim // 4, kernel_size=1, stride=1)
            self.pool41 = nn.AdaptiveAvgPool2d(7)
            self.pool42 = nn.AdaptiveMaxPool2d(7)
            self.sr4 = nn.Conv2d(dim // 4, dim // 4, kernel_size=1, stride=1)
            self.norm1 = nn.LayerNorm(dim // 4)
            self.norm2 = nn.LayerNorm(dim // 4)
            self.norm3 = nn.LayerNorm(dim // 4)
            self.norm4 = nn.LayerNorm(dim // 4)
            self.act = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=2)
        v1 = self.v1(x1).reshape(B, N, self.num_heads // 4, C // self.num_heads).permute(0, 2, 1, 3)
        v2 = self.v2(x2).reshape(B, N, self.num_heads // 4, C // self.num_heads).permute(0, 2, 1, 3)
        v3 = self.v3(x3).reshape(B, N, self.num_heads // 4, C // self.num_heads).permute(0, 2, 1, 3)
        v4 = self.v4(x4).reshape(B, N, self.num_heads // 4, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C1 = x1.shape
        x1_ = x1.permute(0, 2, 1).reshape(B, C1, H, W)
        x1_ = self.sr1((0.5 * self.pool11(x1_) + 0.5 * self.pool12(x1_))).reshape(B, C1, -1).permute(0, 2, 1)
        x1_ = self.norm1(x1_)
        x1_ = self.act(x1_)
        qk1 = self.qk1(x1_).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1 = qk1[0], qk1[1]
        q1 = q1.transpose(-2, -1)
        k1 = k1.transpose(-2, -1)
        v1 = v1.transpose(-2, -1)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        x1 = (attn1 @ v1).permute(0, 3, 1, 2).reshape(B, N, C // 4)

        B, N, C2 = x2.shape
        x2_ = (x2 + x1).permute(0, 2, 1).reshape(B, C2, H, W)
        x2_ = self.sr2((0.5 * self.pool21(x2_) + 0.5 * self.pool22(x2_))).reshape(B, C2, -1).permute(0, 2, 1)
        x2_ = self.norm2(x2_)
        x2_ = self.act(x2_)
        qk2 = self.qk2(x2_).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2 = qk2[0], qk2[1]
        q2 = q2.transpose(-2, -1)
        k2 = k2.transpose(-2, -1)
        v2 = v2.transpose(-2, -1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        x2 = (attn2 @ v2).permute(0, 3, 1, 2).reshape(B, N, C // 4)

        B, N, C1 = x3.shape
        x3_ = (x3 + x2).permute(0, 2, 1).reshape(B, C1, H, W)
        x3_ = self.sr3((0.5 * self.pool31(x3_) + 0.5 * self.pool32(x3_))).reshape(B, C1, -1).permute(0, 2, 1)
        x3_ = self.norm3(x3_)
        x3_ = self.act(x3_)
        qk3 = self.qk3(x3_).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q3, k3 = qk3[0], qk3[1]
        q3 = q3.transpose(-2, -1)
        k3 = k3.transpose(-2, -1)
        v3 = v3.transpose(-2, -1)
        q3 = torch.nn.functional.normalize(q3, dim=-1)
        k3 = torch.nn.functional.normalize(k3, dim=-1)
        attn3 = (q3 @ k3.transpose(-2, -1)) * self.temperature
        attn3 = attn3.softmax(dim=-1)
        attn3 = self.attn_drop(attn3)
        x3 = (attn3 @ v3).permute(0, 3, 1, 2).reshape(B, N, C // 4)

        B, N, C2 = x4.shape
        x4_ = (x4 + x3).permute(0, 2, 1).reshape(B, C2, H, W)
        x4_ = self.sr4((0.5 * self.pool41(x4_) + 0.5 * self.pool42(x4_))).reshape(B, C2, -1).permute(0, 2, 1)
        x4_ = self.norm4(x4_)
        x4_ = self.act(x4_)
        qk4 = self.qk4(x4_).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q4, k4 = qk4[0], qk4[1]
        q4 = q4.transpose(-2, -1)
        k4 = k4.transpose(-2, -1)
        v4 = v4.transpose(-2, -1)
        q4 = torch.nn.functional.normalize(q4, dim=-1)
        k4 = torch.nn.functional.normalize(k4, dim=-1)
        attn4 = (q4 @ k4.transpose(-2, -1)) * self.temperature
        attn4 = attn4.softmax(dim=-1)
        attn4 = self.attn_drop(attn4)
        x4 = (attn4 @ v4).permute(0, 3, 1, 2).reshape(B, N, C // 4)

        x = torch.cat([x1, x2, x3, x4], dim=2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP_(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm = LayerNorm_(dim, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv2d(dim, dim // mlp_ratio, 1)
        self.pos = nn.Conv2d(dim // mlp_ratio, dim // mlp_ratio, 3, padding=1, groups=dim // mlp_ratio)
        self.fc2 = nn.Conv2d(dim // mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x

class LayerNorm_(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class SpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size, expand_ratio=2):
        super().__init__()
        self.norm = LayerNorm_(dim, eps=1e-6, data_format="channels_first")
        self.att1 = nn.Sequential(
                nn.Conv2d(dim//4, dim//4, 1),
                nn.GELU(),
                nn.Conv2d(dim//4, dim//4, kernel_size=5, padding=2, groups=dim//4)
        )
        self.att2 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, kernel_size=7, padding=3, groups=dim // 4)
        )
        self.att3 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, kernel_size=9, padding=4, groups=dim // 4)
        )
        self.att4 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, kernel_size=11, padding=5, groups=dim // 4)
        )
        self.v = nn.Conv2d(dim, dim // 4, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = self.att1(x1) * self.v(x)
        x2 = self.att2(x2+x1) * self.v(x)
        x3 = self.att3(x3+x2) * self.v(x)
        x4 = self.att4(x4+x3) * self.v(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, kernel_size=3, num_heads=8, window_size=14, mlp_ratio=4, drop_path=0.):
        super().__init__()
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP_(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class lrra(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.v = nn.Linear(dim, dim, bias=bias)
        self.qk = nn.Linear(dim, dim * 2, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AdaptiveAvgPool2d(7)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
    def forward(self, x, H, W):
        B, N, C = x.shape
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)
        qk = self.qk(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = in_features
        hidden_features = in_features // 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MSCM(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, use_pos_emb=True,
                 num_heads=8, qkv_bias=True, attn_drop=0., drop=0., scales=1, qk_scale=None, sr_ratio=1, linear=True):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.mssa = mssa(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                         proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.MLP = Mlp(dim)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = x + self.drop_path(self.gamma * self.mssa(self.norm1(x), H, W))
        x = x + self.MLP(self.norm2(x))
        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x
