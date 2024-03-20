import torch
import torch.nn as nn
from copy import deepcopy
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url
from spikingjelly.activation_based import layer, neuron, functional, surrogate

import torchvision.models as models
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model import common
import torch
import torch.nn.functional as F
import copy


__all__ = ['SpikingResNet', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101',
           'spiking_resnet152', 'spiking_resnext50_32x4d', 'spiking_resnext101_32x8d',
           'spiking_wide_resnet50_2', 'spiking_wide_resnet101_2']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size // 2), groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes, out_planes, kernel_size=1, stride=1,bias=False):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=bias)

def make_model(args, parent=False):

    return spiking_ResNet18_FLANC(args, pretrained=False, surrogate_function=surrogate.ATan(), detach_reset=True)

# input x 是什么？图片 or 网络参数（怎么得到x并传入到这个类的）
class conv_basis(nn.Module):
    # filter_bank滤波器组，in_channels输入通道数，basis_size卷积核的通道数，n_basis卷积核的数量，kernel_size每个卷积核的大小
    def __init__(self, filter_bank, in_channels, basis_size, n_basis, kernel_size, stride=1, bias=True):
        super(conv_basis, self).__init__()
        self.in_channels = in_channels
        self.n_basis = n_basis
        self.kernel_size = kernel_size
        self.basis_size = basis_size
        self.stride = stride
        self.group = in_channels // basis_size
        self.weight = filter_bank
        #self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(n_basis, basis_size, kernel_size, kernel_size)))
        self.bias = nn.Parameter(torch.zeros(n_basis)) if bias else None # 表示卷积操作是否加上偏置
        # self.conv1 = conv3x3(stride=self.stride, dilation=self.kernel_size//2)
        #print(stride)
        # self.__repr__()
        # self.conv2d_1 =F.conv2d(in_channel, out_channel,)
        # 设置为多步模式
        functional.set_step_mode(self, step_mode='m')


    # x是输入张量(4维)，返回一个Tensor
    def forward(self, x):
        # T = x.shape[0]   # 时间步单步
        # x_seq_step_by_step  = []  # 时间步单步
        if self.group == 1: # 如果为1表示没有分组，直接使用默认的 F.conv2d 函数对整个卷积核集合进行卷积
        #     for t in range(T):
        #         x = F.conv2d(input=x[t], weight=self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2)
        #         x_seq_step_by_step.append(x.unsqueeze(0))
        #     x_seq_step_by_step = torch.cat(x_seq_step_by_step, 0)
        #     # x = F.conv2d(input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2)
        #
            conv2d = layer.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=self.bias, stride=self.stride,
                                  padding=self.kernel_size//2, step_mode='m')  # 提前给定in 和 out 和 kernel_size，会生成一个weight
            conv2d.weight = self.weight   # 后赋值weight进行覆盖
            conv2d.padding = self.kernel_size//2
            conv2d.padding_mode = 'zeros'  # 走padding_mode 等于zeros的部分
            x = conv2d(x)

            # x = F.conv2d(input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2)  # 原本

            # x = functional.seq_to_ann_forward(x, x1)  # 师兄方法
        else:

            # for xi in torch.split(x, self.basis_size, dim=2):
            #     # x[10, 32, 64, 16, 16]
            #     for t in range(T):
            #        xi =  F.conv2d(input=xi[t], weight=self.weight, bias=self.bias, stride=self.stride, padding=self.kernel_size // 2)
            #        x_seq_step_by_step.append(xi.unsqueeze(0))
            #     x_seq_step_by_step = torch.cat(x_seq_step_by_step, 0)
            #
            # x = torch.cat(x_seq_step_by_step, dim=2)
            #
            # x = torch.cat([F.conv2d(input=xi, weight=self.weight, bias=self.bias, stride=self.stride,padding=self.kernel_size//2)
            #                for xi in torch.split(x, self.basis_size, dim=1)], dim=1)

            x_list = []
            for xi in torch.split(x, self.basis_size, dim=2):
                conv2d = layer.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=self.bias, stride=self.stride,
                                      padding=self.kernel_size // 2, step_mode='m')
                conv2d.weight = self.weight
                conv2d.padding = self.kernel_size // 2
                conv2d.padding_mode = 'zeros'
                x1 = conv2d(xi)
                # print(1, x1.shape)
                x_list.append(x1)
            x = torch.cat(x_list, dim=2)
            # print(2, x.shape)
            # [10, 256, 16, 16, 16]  没有合并起来，调整dim = 2
            # torch.Size([10, 32, 128, 16, 16])
            # torch.Size([10, 32, 32, 18, 18]) 修改后的结果
            # x = conv3x3()
            # x(input)
             # 把X拆开按照维度=1,然后组起来
        return x
 #todo：查看filter_bank，确定为何32×32会变成16×16（spikingresnet中多了个池化层，暂时不清楚为何多这个池化）


    def __repr__(self):
        s = 'Conv_basis(in_channels={}, basis_size={}, group={}, n_basis={}, kernel_size={}, out_channel={})'.format(
            self.in_channels, self.basis_size, self.group, self.n_basis, self.kernel_size, self.group * self.n_basis)
        return s

class DecomBlock(nn.Module):
    def __init__(self, filter_bank, in_channels, out_channels, n_basis, basis_size, kernel_size, stride=1, bias=False, conv=conv3x3, norm=common.default_norm, act=common.default_act, spiking_neuron: callable = None, **kwargs ):
        super(DecomBlock, self).__init__()
        group = in_channels // basis_size # 输入通道数比上卷积核的通道数得到通道分组数量
        modules = [conv_basis(filter_bank, in_channels, basis_size, n_basis, kernel_size, stride, bias)]
        #if norm is not None: modules.append(norm(group * n_basis))
        modules.append(conv(group * n_basis, out_channels, kernel_size=1, stride=1, bias=bias))
        # 将输入通道in_channels变成group*n_basis（通道分组数量×卷积核的数量）得到新的卷积层，将其添加到modules列表中
        self.conv = nn.Sequential(*modules)
        # 将多个神经网络层组合成一个顺序执行的模型
        # 设置为多步模式
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # print("1 DecomBlock", x.shape)
        x = self.conv(x)
        # print("2 DecomBlock", x.shape)
        return x
        # return self.conv(x)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, n_basis, basis_size, block_expansion=0, stride=1, downsample=False,
                 groups = 1, dilation=1, base_width=64, norm_layer=None, conv3x3=conv3x3, conv1x1=conv1x1,
                 spiking_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        self.th = 0.5
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d  # nn-> layer
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.filter_bank_1 = nn.Parameter(torch.empty(n_basis, basis_size, 3, 3))
        self.filter_bank_2 = nn.Parameter(torch.empty(n_basis, basis_size, 3, 3))

        X = torch.empty(n_basis*2, basis_size, 3, 3)
        torch.nn.init.orthogonal(X)
        self.filter_bank_1.data = copy.deepcopy(X[:n_basis,:])
        self.filter_bank_2.data = copy.deepcopy(X[n_basis:,:])


        self.conv1 = DecomBlock(self.filter_bank_1, inplanes, planes, n_basis, basis_size, kernel_size=3, stride=stride, bias=False)
        # 卷积层1
        self.bn1 = norm_layer(planes)
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        # 脉冲输出的神经元层1
        self.conv2 = DecomBlock(self.filter_bank_2, planes, planes, n_basis, basis_size, kernel_size=3, stride=1, bias=False)
        # 卷积层2
        self.bn2 = norm_layer(planes)
        self.sn2 = neuron.IFNode(v_threshold=self.th)
        # 脉冲输出的神经元层2
        if downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * block_expansion,
                        kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(planes * block_expansion),   # nn.BatchNorm2d(planes * block_expansion)修改为SNN适配的
            )
        else:
            self.downsample = None
        # self.downsample = None # 对downsample设置为none，spiking_resnet中设置的为none。补充，但是在这里需要用到downsample，因为输入和输出的通道数不一样，无法相加。
        self.stride = stride
        # 设置为多步模式
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # print("1 BasicBlock", x.shape)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)   # 进行求均值元素的总数应该是64而不是16，需要找到为什么经过out = self.conv1(x)输出通道还是64
        out = self.sn1(out)
        # print('BasicBlock基础块第一层：',out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn2(out)
        # print('BasicBlock基础块第二层：',out)

        return out

    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
# 没用到
class DefaultBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,  n_basis, basis_size, block_expansion=0, stride=1, downsample=False, conv3x3=common.default_conv, conv1x1=common.default_conv):
        super(DefaultBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * block_expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_expansion),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # 瓶颈层,用于在resnet中构建深度神经网络,暂时不用修改
    expansion = 4
    # 每个瓶颈层输出的特征图的通道数是输入特征图的4倍,为了使网络中的计算分不到更多的通道中,从而可以提升模型的性能

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn3(out)

        return out

class SpikingResNet(nn.Module):
    def __init__(self, T: int, block, layers, n_basis=1, basis_fract=1, net_fract=1, num_classes=10, conv3x3=conv3x3, conv1x1=conv1x1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, spiking_neuron: callable = None, **kwargs):
        # groups = 1 是spiking resnet中的
        super(SpikingResNet, self).__init__()
        # 步长
        self.T = T
        self.th = 0.5

        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.basis_fract = basis_fract
        self.net_fract = net_fract
        self.n_basis = n_basis
        self.conv3x3 = conv3x3
        self.conv1x1 = conv1x1

        self.base_width = width_per_group
        # self.conv1 = conv3x3(3, 64, kernel_size=3, stride=1, bias=False)  # resnet_flanc
        # self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                           bias=False)  # spiking_resnet
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                  bias=False)  #按照spiking_resnetpadding如果设置为3，
        self.bn1 = norm_layer(self.inplanes)

        self.sn1 = neuron.IFNode(v_threshold=self.th)
        self.sn2 = neuron.IFNode(v_threshold=self.th)

        # self.sn1 = spiking_neuron(**deepcopy(kwargs))
        # self.sn2 = spiking_neuron(**deepcopy(kwargs))  # 同一个神经元不能用两次
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)

        m = round(64*self.n_basis) #for n_basis   16
        n = round(64*self.basis_fract) #for basis_size   8
        cfg = [(m,n),(m,n)]
        self.layer1 = self._make_layer(block, 64, layers[0], cfg, stride=1, spiking_neuron=spiking_neuron, **kwargs)


        m = round(128*self.n_basis)
        n = round(64*self.basis_fract)
        n2 = round(128*self.basis_fract)
        cfg = [(m,n),(m,n2)]
        self.layer2 = self._make_layer(block, 128, layers[1], cfg, stride=2,
                                       dilate=replace_stride_with_dilation[0], spiking_neuron=spiking_neuron, **kwargs)

        m = round(256*self.n_basis)
        n = round(128*self.basis_fract)
        n2 = round(256*self.basis_fract)
        cfg = [(m,n),(m,n2)]
        self.layer3 = self._make_layer(block, 256, layers[2], cfg, stride=2,
                                       dilate=replace_stride_with_dilation[1], spiking_neuron=spiking_neuron, **kwargs)

        m = round(512*self.n_basis)
        n = round(256*self.basis_fract)
        n2 = round(512*self.basis_fract)
        cfg = [(m,n),(m,n2)]
        self.layer4 = self._make_layer(block, 512, layers[3], cfg, stride=2,
                                       dilate=replace_stride_with_dilation[2], spiking_neuron=spiking_neuron, **kwargs)


        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(round(512 * block.expansion * self.net_fract), num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # 设置为多步模式
        functional.set_step_mode(self, step_mode='m')



    def _make_layer(self, block, planes, blocks, cfg, stride=1, dilate=False, spiking_neuron: callable = None, **kwargs):
        planes = round(planes*self.net_fract) #if planes!=64 else planes
        downsample = stride != 1 or self.inplanes != planes * block.expansion

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
        # layers.append(block(self.inplanes, *cfg[0], planes, block.expansion, stride, downsample, groups=self.groups, self.base_width, previous_dilation, norm_layer, spiking_neuron=spiking_neuron, **kwargs))
        layers.append(block(self.inplanes, planes, *cfg[0], block.expansion, stride, downsample, conv3x3=self.conv3x3, conv1x1=self.conv1x1, spiking_neuron=spiking_neuron, **kwargs))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, *cfg[i], groups=self.groups,
                                dilation=self.dilation, base_width=self.base_width,
                                norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # x.shape = [N, C, H, W]
        # print('传入：', x.shape)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

        # print('加入时间步 输出：', x.shape)
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        # print('1 conv1 SpikingResNet _forward_impl 输出：', x.shape)
        # x = self.maxpool(x)  # 因为池化所以32变成16
        # print('maxpool 2 SpikingResNet _forward_impl conv1输出：', x.shape)

        x = self.layer1(x)  # 这里出问题
        # print('2 layer1 SpikingResNet _forward_impl 输出：', x.shape)

        x = self.layer2(x)
        # print('3 layer2 SpikingResNet _forward_impl 输出：', x.shape)

        x = self.layer3(x)
        # print('4 layer3 SpikingResNet _forward_impl 输出：', x.shape)

        x = self.layer4(x)
        # print('5 layer4 SpikingResNet _forward_impl 输出：', x.shape)



        x = self.avgpool(x)

        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        # x=x
        x = self.fc(x)
        x = self.sn2(x)
        # x = x  # x.shape（10，32，10）

        return x

    def forward(self, x):
        # print("SpikingResNet forward", x.shape)
        return self._forward_impl(x)

class spiking_ResNet18_FLANC(SpikingResNet):
    def __init__(self, args, conv3x3=conv3x3, conv1x1=conv1x1, spiking_neuron: callable = None,
                 **kwargs):

        num_classes = 100 if "cifar100" in args.data_train else 10

        super(spiking_ResNet18_FLANC, self).__init__(T = args.T, block = BasicBlock, layers = [2, 2, 2, 2], n_basis=args.n_basis, basis_fract=args.basis_fraction, net_fract=args.net_fraction, num_classes=num_classes, conv3x3=conv3x3, conv1x1=conv1x1, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
        spiking_neuron=spiking_neuron)

        pretrained = args.pretrained == 'True'
        if conv3x3 == common.default_conv:
            if pretrained:
                self.load(args, strict=True)

    def load(self, args, strict):
        self.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=strict)

def loss_type(loss_para_type):
    if loss_para_type == 'L1':
        loss_fun = nn.L1Loss()
        # loss_fun = F.l1_loss()     # SNN L1Loss
    elif loss_para_type == 'L2':
        loss_fun = nn.MSELoss()
        # loss_fun = F.mse_loss()  # SNN MSEloss
    else:
        raise NotImplementedError
    return loss_fun

# orth_loss 用于计算权重矩阵的正交性损失
def orth_loss(model, args, para_loss_type='L2'):
    #from IPython import embed; embed(); exit()

    #current_state = list(model.named_parameters())
    loss_fun = loss_type(para_loss_type)

    loss = 0
    for l_id in range(1,5):
        layer = getattr(model,"layer"+str(l_id))
        for b_id in range(2):
            block = getattr(layer,str(b_id))
            for f_id in range(1,3):
                filter_bank = getattr(block,"filter_bank_"+str(f_id))
                #filter_bank_2 = getattr(block,"filter_bank_2")
                all_bank = filter_bank
                num_all_bank = filter_bank.shape[0]
                B = all_bank.view(num_all_bank, -1)
                D = torch.mm(B,torch.t(B))
                D = loss_fun(D, torch.eye(num_all_bank, num_all_bank).cuda())
                loss = loss + D
    return loss



def _spiking_resnet(arch, block, layers, pretrained, progress, spiking_neuron, **kwargs):
    model = SpikingResNet(block, layers, spiking_neuron=spiking_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def spiking_resnet18(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module

    A spiking version of ResNet-18 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """

    return _spiking_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, spiking_neuron, **kwargs)

def spiking_resnet34(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-34
    :rtype: torch.nn.Module

    A spiking version of ResNet-34 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, spiking_neuron, **kwargs)

def spiking_resnet50(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-50
    :rtype: torch.nn.Module

    A spiking version of ResNet-50 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, spiking_neuron, **kwargs)


def spiking_resnet101(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-101
    :rtype: torch.nn.Module

    A spiking version of ResNet-101 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, spiking_neuron, **kwargs)


def spiking_resnet152(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-152
    :rtype: torch.nn.Module

    A spiking version of ResNet-152 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, spiking_neuron, **kwargs)

def spiking_resnext50_32x4d(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-50 32x4d
    :rtype: torch.nn.Module

    A spiking version of ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _spiking_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, spiking_neuron, **kwargs)


def spiking_resnext101_32x8d(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-101 32x8d
    :rtype: torch.nn.Module

    A spiking version of ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _spiking_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, spiking_neuron, **kwargs)

def spiking_wide_resnet50_2(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-50-2
    :rtype: torch.nn.Module

    A spiking version of Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _spiking_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, spiking_neuron, **kwargs)


def spiking_wide_resnet101_2(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-101-2
    :rtype: torch.nn.Module

    A spiking version of Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _spiking_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, spiking_neuron, **kwargs)

