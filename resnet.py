import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv3x3_(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=out_planes, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LKDWConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(LKDWConv, self).__init__()
        self.Conv7x7 = nn.Conv2d(in_ch, out_ch, kernel_size=11, stride=stride, padding=5, groups=in_ch)
        self.ConvH1 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(1, 0), groups=1)
        self.ConvW1 = nn.Conv2d(out_ch, out_ch, kernel_size=(1, 3), padding=(0, 1), groups=1)
        self.ConvH2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(1, 0), groups=1)
        self.ConvW2 = nn.Conv2d(out_ch, out_ch, kernel_size=(1, 3), padding=(0, 1), groups=1)
        self.Conv = nn.Conv2d(out_ch * 2, out_ch, 1, padding=0)

    def forward(self, x):
        # print(x.shape)
        x_ = self.Conv7x7(x)
        x_h = self.ConvH1(x_)
        x_hw = self.ConvW1(x_h)
        x_w = self.ConvW2(x_)
        x_wh = self.ConvH2(x_w)
        out = self.Conv(torch.cat((x_hw, x_wh), dim=1))
        return out


def dsa_conv(in_channels, out_channels, stride=1, padding=3, dilation=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                  groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class ASPP(nn.Module):
    def __init__(self, in_channel, depth=256, ratio=1 / 4, out_channel=256):
        super(ASPP, self).__init__()
        self.in_channel = in_channel
        self.ratio = ratio
        out_channel = int(in_channel * ratio)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim

        self.conv = nn.Conv2d(in_channel, depth, 1, 1)

        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        # 普通空洞卷积
        # self.atrous_block3 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        # self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        # self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        # self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        # self.atrous_block24 = nn.Conv2d(in_channel, depth, 3, 1, padding=24, dilation=24)
        # 深度空洞可分离卷积
        self.ads_block3 = dsa_conv(in_channel, depth, 1, 3, 3)
        self.ads_block6 = dsa_conv(in_channel, depth, 1, 6, 6)
        self.ads_block12 = dsa_conv(in_channel, depth, 1, 12, 12)
        self.ads_block18 = dsa_conv(in_channel, depth, 1, 18, 18)
        self.ads_block24 = dsa_conv(in_channel, depth, 1, 24, 24)
        # self.ads_block30 = dsa_conv(in_channel, depth, 1, 30, 30)
        # self.ads_block36 = dsa_conv(in_channel, depth, 1, 36, 36)

        self.conv_1x1_output = nn.Conv2d(depth * 7, depth, 1, 1)
        # 平均池化+Transform
        # self.avg_pool = nn.AdaptiveMaxPool2d(1)     #最大
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均
        self.channel_conv = nn.Sequential(
            nn.Conv2d(depth, out_channel, kernel_size=1),
            nn.LayerNorm([out_channel, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            # nn.RReLU(inplace=True),   #新激活
            # nn.GELU(),
            nn.Conv2d(out_channel, in_channel, kernel_size=1))

    def forward(self, x):
        y = x
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        # image_features = F.upsample(image_features, size=size, mode='bilinear')
        image_features = F.interpolate(image_features, size=x.size()[2:], mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)
        # 普通空洞卷积
        # atrous_block3 = self.atrous_block3(x)
        # atrous_block6 = self.atrous_block6(x)
        # atrous_block12 = self.atrous_block12(x)
        # atrous_block18 = self.atrous_block18(x)
        # atrous_block24 = self.atrous_block24(x)
        # 深度空洞可分离卷积
        ads_block3 = self.ads_block3(x)
        ads_block6 = self.ads_block6(x)
        ads_block12 = self.ads_block12(x)
        ads_block18 = self.ads_block18(x)
        ads_block24 = self.ads_block24(x)
        # ads_block30 = self.ads_block30(x)
        # ads_block36 = self.ads_block36(x)

        # print(image_features.shape)

        # net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12,
        #                                       atrous_block18, atrous_block24], dim=1))
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, ads_block3, ads_block6, ads_block12,
                                              ads_block18, ads_block24], dim=1))
        net = self.avg_pool(net)
        # print(net.shape)
        # "+"形式
        net = self.channel_conv(net)
        # print(net.shape)
        net = y + net

        # “*”形式
        # net = torch.sigmoid(self.channel_conv(net))
        # net = net * net
        return net


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            # 全连接层
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x


class FANLayer_Conv(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, kernel_size=1, bias=True, with_gate=True, pool=False):
        super(FANLayer_Conv, self).__init__()
        self.input_linear_p = nn.Conv2d(input_dim, output_dim // 4, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, bias=bias)
        self.input_linear_g = nn.Conv2d(input_dim, (output_dim - output_dim // 2), kernel_size=kernel_size,
                                        stride=stride, padding=kernel_size // 2)
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=32, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = FANLayer_Conv(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # self.Attn = CBAM(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.Attn(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=32, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 32.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=32, replace_stride_with_dilation=None,
                 norm_layer=None, in_channel=3):
        super(ResNet, self).__init__()
        # self.conv_ = nn.Conv2d(1, 1, 3, padding=2)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 4
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.block = block
        self.groups = groups
        self.base_width = width_per_group

        # 224,224,3 -> 112,112,64
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 112,112,64 -> 56,56,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 56,56,64 -> 56,56,256
        self.layer1 = self._make_layer(block, 8, layers[0])

        # 56,56,256 -> 28,28,512
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        # 28,28,512 -> 14,14,1024
        self.layer3 = self._make_layer(block, 24, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # 14,14,1024 -> 7,7,2048
        self.layer4 = self._make_layer(block, 32, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 7,7,2048 -> 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 2048 -> num_classes
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Conv_block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def freeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = True


def resnet18(pretrained=False, progress=True, num_classes=2, in_channel=3):
    model = ResNet(BasicBlock, [1, 1, 1, 1], in_channel=in_channel)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes != 1000:
        model.fc = nn.Linear(256 * model.block.expansion, num_classes)
    return model


def resnet34(pretrained=False, progress=True, num_classes=2, in_channel=3):
    model = ResNet(BasicBlock, [3, 4, 6, 3], in_channel=in_channel)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet34'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes != 1000:
        model.fc = nn.Linear(256 * model.block.expansion, num_classes)
    return model


def resnet50(pretrained=False, progress=True, num_classes=2, in_channel=3):
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_channel=in_channel)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes != 1000:
        model.fc = nn.Linear(256 * model.block.expansion, num_classes)
    return model


def resnet101(pretrained=False, progress=True, num_classes=2, in_channel=3):
    model = ResNet(Bottleneck, [3, 4, 23, 3], in_channel=in_channel)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes != 1000:
        model.fc = nn.Linear(256 * model.block.expansion, num_classes)
    return model


def resnet152(pretrained=False, progress=True, num_classes=2, in_channel=3):
    model = ResNet(Bottleneck, [3, 8, 36, 3], in_channel=in_channel)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes != 1000:
        model.fc = nn.Linear(256 * model.block.expansion, num_classes)
    return model


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model = resnet18().cuda()
    model1 = resnet34().cuda()
    model2 = resnet50().cuda()
    model3 = resnet101().cuda()
    model4 = resnet152().cuda()

    # net.load_from(weights=np.load(config_vit.pretrained_path))
    # model = ViT_seg(num_classes=1).to('cuda')

    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('flops:', flops)
    print('params:', params)
    flops, params = get_model_complexity_info(model1, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('flops:', flops)
    print('params:', params)
    flops, params = get_model_complexity_info(model2, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('flops:', flops)
    print('params:', params)
    flops, params = get_model_complexity_info(model3, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('flops:', flops)
    print('params:', params)
    flops, params = get_model_complexity_info(model4, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('flops:', flops)
    print('params:', params)