import math
import numpy as np
import torch
import torch.nn as nn
import torch.autograd.variable as Variable
import torch.nn.functional as F

###初始化一个默认的卷积，使图像进入默认卷积后只改变维度
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

####这个有待商榷不知道干什么
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

####上采样层，将特征图按照倍数扩大，目前只能扩大2的倍数倍以及3倍
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

####下采样，目前看来只能缩小2的倍数
class DownBlock(nn.Module):
    def __init__(self, opt, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = opt.negval

        if nFeat is None:
            nFeat = opt.n_feats

        if in_channels is None:
            in_channels = opt.n_colors

        if out_channels is None:
            out_channels = opt.n_colors

        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x

class DownBlockplus(nn.Module):
    def __init__(self, opt, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlockplus, self).__init__()
        negval = opt.negval

        if nFeat is None:
            nFeat = opt.n_feats

        if in_channels is None:
            in_channels = opt.n_colors

        if out_channels is None:
            out_channels = opt.n_colors

        self.dual_block1 = nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True),
                nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

            )
        self.dual_block2=nn.Sequential(
                nn.Conv2d(out_channels, nFeat, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True),
                nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True),
                nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

            )





    def forward(self, x):
        result=[]
        x = self.dual_block1(x)
        result.append(x)
        x= self.dual_block2(x)
        result.append(x)
        return result


## Channel Attention (CA) Layer
####实现下面attention机制的一部分
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
####attention检查相似度机制
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
###胶囊网络
class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        # print("原本",x.shape)
        u = [capsule(x) for capsule in self.capsules]  # 8 *（32，6，6）
        # print("capsule",u[0].shape)
        u = torch.stack(u, dim=1)  # （8，32，6，6）
        # print("u",u.shape)
        # print(u.view(x.size(0), self.num_routes,-1).shape)
        u = u.view(x.size(0), self.num_routes, -1)  # （2048，8）
        # print("u2", u[0].shape)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        # [1, 2048, 10, 16, 8]
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        # print("xxxxxxxxxxxxx1",x.shape)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)  # (2048,10,8,1)
        # print("xxxxxxxxx2", x.shape)
        # print(self.W.shape)
        W = torch.cat([self.W] * batch_size, dim=0)
        # print("w",W.shape,"x",x.shape)
        u_hat = torch.matmul(W, x)

        b_ij = torch.zeros(1, self.num_routes, self.num_capsules, 1)  # (2048,10,1)
        b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)  # (2048,10,1,1)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class Decoder(nn.Module):
    def __init__(self, input_width=28, input_height=28, input_channel=1):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * input_channel, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_height * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        classes = torch.sqrt((x ** 2).sum(2))  # （10，1）计算每个向量的长度
        classes = F.softmax(classes, dim=0)  # （10，1）  通过长度计算每个类别的概率

        _, max_length_indices = classes.max(dim=1)  # 找到预测概率最大的类别
        # one hot编码
        masked = torch.sparse.torch.eye(self.input_channel).to(device="cuda")
            # .to(device="cuda")
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        # 找到长度最长的向量进行重构
        # print("masked",masked.shape)
        # print("part1",x * masked[:, :, None, None].shape)
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        # print("t",t.shape)
        reconstructions = self.reconstraction_layers(t)  # （784）
        # print("reconstructions",reconstructions.shape)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions