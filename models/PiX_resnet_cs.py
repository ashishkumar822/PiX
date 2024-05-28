import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pix_layer_cuda

from timm.models.registry import register_model

# gradients in the backward are received in the order of tensor as they were output in forward function
class PixOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zeta: int, tau: float,  input: torch.Tensor, p: torch.Tensor):
        outputs = pix_layer_cuda.forward(zeta, tau, input, p)
        ctx.save_for_backward(input, p)
        ctx.zeta = zeta
        ctx.tau = tau
        return outputs[0]

    @staticmethod
    def backward(ctx, out_grad):
        input, p = ctx.saved_tensors
        zeta = ctx.zeta
        tau = ctx.tau
        input_grad, fusion_prob_grad = pix_layer_cuda.backward(zeta, tau, input, p, out_grad)
        return None, None, input_grad, fusion_prob_grad

class PiX(torch.nn.Module):
    def __init__(self, zeta, tau = 0.5):
        super(PiX, self).__init__()
        self.zeta = int(zeta)
        self.tau = tau

    def forward(self, input, p):
        return PiXOperator.apply(self.zeta, self.tau, input, p)


class PixBottleneck(nn.Module):
    def __init__(self, n_ip, n_op, stride, reduction_ratio, use_projection, tau = 0.5):
        super(PixBottleneck, self).__init__()

        self.bn_momentum = 0.05

        self.use_projection = use_projection
        sqz_n_op = int(n_ip / reduction_ratio)

        self.pix = PiX(reduction_ratio, tau)

        self.conv_sqz = nn.Conv2d(n_ip, sqz_n_op, 1, 1, 0, bias=False)
        self.sigmoid_sqz = nn.Sigmoid()


        self.conv_ce = nn.Conv2d(sqz_n_op, sqz_n_op, 3, stride, 1, bias=False)
        self.bn_ce = nn.BatchNorm2d(sqz_n_op, momentum=self.bn_momentum)
        self.relu_ce = nn.ReLU(True)

        self.conv_exp = nn.Conv2d(sqz_n_op, n_op, 1, 1, 0, bias=False)
        self.bn_exp = nn.BatchNorm2d(n_op, momentum=self.bn_momentum)
        self.final_relu = nn.ReLU(True)

        if(self.use_projection):
            self.conv_proj = nn.Conv2d(n_ip, n_op, 1, stride, 0, bias=False)
            self.bn_proj = nn.BatchNorm2d(n_op, momentum=self.bn_momentum)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, ip):

        global_pool = self.global_pool(ip)
        sampling_prob = self.sigmoid_sqz(self.conv_sqz(global_pool))

        x = self.pix.forward(ip, sampling_prob)

        x = self.relu_ce(self.bn_ce(self.conv_ce(x)))
        x = self.bn_exp(self.conv_exp(x))

        if (self.use_projection):
            y = self.bn_proj(self.conv_proj(ip))
            z = y + x
            return self.final_relu(z)
        else:
            y = ip
            z = y + x
            return self.final_relu(z)


class PiXResNet_stage(nn.Module):
    def __init__(self, n_ip, n_op, n_blocks, stride, reduction_factor_first, tau, bn_momentum=0.05):
        super(PiXResNet_stage, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            use_projection = True if (i==0) else False
            strd = stride if (i==0) else 1
            reduction_ratio = reduction_factor_first if (i==0) else 4
            n_ip = n_ip if (i==0) else n_op
            self.blocks.append(PixBottleneck(n_ip, n_op, strd, reduction_ratio, use_projection, tau=tau, bn_momentum=bn_momentum))

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x


class PiXResNet(nn.Module):
    def __init__(self, blocks=[3,4,6,3], strides=[1, 2, 2, 2], nb_classes=1000, drop = 0.1):
        super(PiXResNet, self).__init__()

        tau = 0.5

        self.conv_stem = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn_stem = nn.BatchNorm2d(64, momentum= 0.05)
        self.relu_stem = nn.ReLU(True)
        self.pool = nn.MaxPool2d(3,2,1)

        reduction_factor_first = [1, 2, 2, 2]
        n_ip = [64, 256, 512, 1024]
        n_op = [256, 512, 1024, 2048]
        self.layer1 = PiXResNet_stage(n_ip[0], n_op[0], blocks[0], strides[0], reduction_factor_first[0], tau = tau)
        self.layer2 = PiXResNet_stage(n_ip[1], n_op[1], blocks[1], strides[1], reduction_factor_first[1], tau = tau)
        self.layer3 = PiXResNet_stage(n_ip[2], n_op[2], blocks[2], strides[2], reduction_factor_first[2], tau = tau)
        self.layer4 = PiXResNet_stage(n_ip[3], n_op[3], blocks[3], strides[3], reduction_factor_first[3], tau = tau)

        self.classifier = nn.Conv2d(n_op[-1], nb_classes, 1)
        self.gp = nn.AdaptiveAvgPool2d([-1, -1])
        self.dropout = nn.Dropout2d(drop)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, ip):
        x = self.relu_stem(self.bn_stem(self.conv_stem(ip)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gp(x)
        x = self.dropout(x)
        
        x = self.classifier(x)

        return x

@register_model
def PiXResNet50_cs(pretrained= False,  **kwargs):
    model = PiXResNet(blocks=[3, 4, 6, 3], strides=[1,2,2,2], **kwargs)
    return model

@register_model
def PiXResNet101_cs(pretrained= False, **kwargs):
    model = PiXResNet(blocks=[3, 4, 23, 3], strides=[1,2,2,2], **kwargs)
    return model

@register_model
def PiXResNet152_cs(pretrained= False, **kwargs):
    model = PiXResNet(blocks=[3, 8, 36, 3], strides=[1,2,2,2], **kwargs)
    return model
