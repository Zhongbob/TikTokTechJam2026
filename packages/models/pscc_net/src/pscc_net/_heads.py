# ------------------------------------------------------------------------------
# Vendored from proteus1991/PSCC-Net (MIT):
#   * models/NLCDetection.py  -> NonLocalMask + NLCDetection  (localization head)
#   * models/detection_head.py -> DetectionHead              (binary detection head)
#
# Both heads take an ``args`` mapping with an ``args['crop_size']`` = [H, W]:
# the HRNet features are bilinearly resized to that grid before the head runs
# (a no-op when the input is already H x W, since PSCC-Net's HRNet stem is
# stride-1). Unmodified apart from this header.
# ------------------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscc_net._hrnet import get_hrnet_cfg

BN_MOMENTUM = 0.01


# --- localization head (models/NLCDetection.py) --------------------------------

class NonLocalMask(nn.Module):
    def __init__(self, in_channels, reduce_scale):
        super(NonLocalMask, self).__init__()

        self.r = reduce_scale

        # input channel number
        self.ic = in_channels * self.r * self.r

        # middle channel number
        self.mc = self.ic

        self.g = nn.Conv2d(in_channels=self.ic, out_channels=self.ic,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                             kernel_size=1, stride=1, padding=0)

        self.W_s = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.W_c = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.gamma_s = nn.Parameter(torch.ones(1))

        self.gamma_c = nn.Parameter(torch.ones(1))

        self.getmask = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            value :
                f: B X (HxW) X (HxW)
                ic: intermediate channels
                z: feature maps( B X C X H X W)
            output:
                mask: feature maps( B X 1 X H X W)
        """

        b, c, h, w = x.shape

        x1 = x.reshape(b, self.ic, h // self.r, w // self.r)

        # g x
        g_x = self.g(x1).view(b, self.ic, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta
        theta_x = self.theta(x1).view(b, self.mc, -1)

        theta_x_s = theta_x.permute(0, 2, 1)
        theta_x_c = theta_x

        # phi x
        phi_x = self.phi(x1).view(b, self.mc, -1)

        phi_x_s = phi_x
        phi_x_c = phi_x.permute(0, 2, 1)

        # non-local attention
        f_s = torch.matmul(theta_x_s, phi_x_s)
        f_s_div = F.softmax(f_s, dim=-1)

        f_c = torch.matmul(theta_x_c, phi_x_c)
        f_c_div = F.softmax(f_c, dim=-1)

        # get y_s
        y_s = torch.matmul(f_s_div, g_x)
        y_s = y_s.permute(0, 2, 1).contiguous()
        y_s = y_s.view(b, c, h, w)

        # get y_c
        y_c = torch.matmul(g_x, f_c_div)
        y_c = y_c.view(b, c, h, w)

        # get z
        z = x + self.gamma_s * self.W_s(y_s) + self.gamma_c * self.W_c(y_c)

        # get mask
        mask = torch.sigmoid(self.getmask(z.clone()))

        return mask, z


class NLCDetection(nn.Module):
    def __init__(self, args):
        super(NLCDetection, self).__init__()

        self.crop_size = args['crop_size']

        FENet_cfg = get_hrnet_cfg()

        num_channels = FENet_cfg['STAGE4']['NUM_CHANNELS']

        feat1_num, feat2_num, feat3_num, feat4_num = num_channels

        self.getmask4 = NonLocalMask(feat4_num, 1)
        self.getmask3 = NonLocalMask(feat3_num, 2)
        self.getmask2 = NonLocalMask(feat2_num, 2)
        self.getmask1 = NonLocalMask(feat1_num, 4)

    def forward(self, feat):
        """
            inputs :
                feat : a list contains features from s1, s2, s3, s4
            output:
                mask1: output mask ( B X 1 X H X W)  (plus coarser mask2/3/4)
        """
        s1, s2, s3, s4 = feat

        if s1.shape[2:] == self.crop_size:
            pass
        else:
            s1 = F.interpolate(s1, size=self.crop_size, mode='bilinear', align_corners=True)
            s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode='bilinear', align_corners=True)
            s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode='bilinear', align_corners=True)
            s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode='bilinear', align_corners=True)

        mask4, z4 = self.getmask4(s4)
        mask4U = F.interpolate(mask4, size=s3.size()[2:], mode='bilinear', align_corners=True)

        s3 = s3 * mask4U
        mask3, z3 = self.getmask3(s3)
        mask3U = F.interpolate(mask3, size=s2.size()[2:], mode='bilinear', align_corners=True)

        s2 = s2 * mask3U
        mask2, z2 = self.getmask2(s2)
        mask2U = F.interpolate(mask2, size=s1.size()[2:], mode='bilinear', align_corners=True)

        s1 = s1 * mask2U
        mask1, z1 = self.getmask1(s1)

        return mask1, mask2, mask3, mask4


# --- binary detection head (models/detection_head.py) -------------------------

class _DetBottleneck(nn.Module):
    """PSCC-Net's detection-head bottleneck (expansion=2, distinct from the
    HRNet bottleneck)."""

    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_DetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DetectionHead(nn.Module):
    def __init__(self, args):
        super(DetectionHead, self).__init__()
        self.crop_size = args['crop_size']

        FENet_cfg = get_hrnet_cfg()
        pre_stage_channels = FENet_cfg['STAGE4']['NUM_CHANNELS']

        # classification head
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels)

        self.classifier = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2)
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_head(self, pre_stage_channels):
        head_block = _DetBottleneck
        head_channels = pre_stage_channels

        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, feat):
        s1, s2, s3, s4 = feat

        if s1.shape[2:] == self.crop_size:
            pass
        else:
            s1 = F.interpolate(s1, size=self.crop_size, mode='bilinear', align_corners=True)
            s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode='bilinear', align_corners=True)
            s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode='bilinear', align_corners=True)
            s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode='bilinear', align_corners=True)

        y_list = [s1, s2, s3, s4]

        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + \
                self.downsamp_modules[i](y)

        y = self.final_layer(y)

        # average and flatten
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)

        logit = self.classifier(y)

        return logit
