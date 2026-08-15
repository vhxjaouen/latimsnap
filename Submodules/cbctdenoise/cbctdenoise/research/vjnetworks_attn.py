"""2D ResNet (Triplet-Attention) generator used by the SynthRAD2025 NECSR runs.

This reproduces the ``MyUNet_plus``/``Residual_block``/``Attention`` topology
that the metric-scored ``2025-NECSR_ResNetSynthRAD_VJ016_*`` checkpoints were
trained with (weights keyed ``generator_A_to_B.resblockN.*``). It is only used
to (re)load those checkpoints and export them to ONNX for the I2I server.

The attention machinery is CBAM-flavored: a channel-attention module (CAMMax)
plus a StripPool-style spatial module (SAM) plus a "Triplet Attention" block
(cw/hc/hw channel, height-cross, width-cross views). ``Attention(mode=3)`` is
used, i.e. only the triplet path is active at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Triplet attention (paper: "Rotate to Attend: Convolutional Triplet Attention")
# ---------------------------------------------------------------------------

class ZPool(nn.Module):
    """Stack per-channel max and mean into 2 channels."""
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 relu=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                              padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.relu is not None:
            x = self.relu(x)
        return x


class AttentionGate(nn.Module):
    """Spatial attention gate for one rotation view."""
    def __init__(self):
        super().__init__()
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, 7, padding=3)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(self.compress(x)))


class TripletAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.hw = AttentionGate()

    def forward(self, x):
        # cross-width view (rotate H<->C)
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out11 = self.cw(x_perm1).permute(0, 2, 1, 3).contiguous()
        # cross-height view (rotate C<->W)
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out21 = self.hc(x_perm2).permute(0, 3, 2, 1).contiguous()
        return self.hw(x) + x_out11 + x_out21


# ---------------------------------------------------------------------------
# Channel / spatial attention
# ---------------------------------------------------------------------------

class CAM(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x


class SAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        ic = in_channels // 4
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, ic, 1, bias=False),
                                   nn.BatchNorm2d(ic), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(ic, ic, (1, 3), 1, (0, 1), bias=False),
                                   nn.BatchNorm2d(ic))
        self.conv3 = nn.Sequential(nn.Conv2d(ic, ic, (3, 1), 1, (1, 0), bias=False),
                                   nn.BatchNorm2d(ic))
        self.conv4 = nn.Sequential(nn.Conv2d(ic, ic, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(ic), nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(ic, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w),
                           mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w),
                           mode='bilinear', align_corners=True)
        x4 = self.conv4(F.relu_(x2 + x3))
        return self.sigmoid(self.conv5(x4)) * x


class Attention(nn.Module):
    def __init__(self, in_channels, mode=3):
        super().__init__()
        self.cam = CAM(in_channels)
        self.sam = SAM(in_channels)
        self.triplet = TripletAttention()
        self.mode = mode

    def forward(self, x):
        if self.mode == 1:
            return self.cam(x) + x
        if self.mode == 2:
            return self.sam(x) + x
        if self.mode == 3:
            return self.triplet(x)
        return self.cam(x) + self.sam(x) + x


# ---------------------------------------------------------------------------
# Residual block + U-Net with triangular skip fusions
# ---------------------------------------------------------------------------

class Residual_block(nn.Module):
    def __init__(self, in_ch, out_ch, attention=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_ch)
        self.conv_branch = nn.Conv2d(in_ch, out_ch, 1, padding=0)
        self.in_branch = nn.InstanceNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.in_ch, self.out_ch, self.attention = in_ch, out_ch, attention
        self.attention_module = Attention(out_ch, mode=3)

    def forward(self, x):
        x_identity = x
        x = self.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        if self.in_ch != self.out_ch:
            x_identity = self.in_branch(self.conv_branch(x_identity))
        out = self.relu(x_identity + x)
        if self.attention:
            out = self.attention_module(out)
        return out


class MyUNet1(nn.Module):
    """2D ResNet generator (``generator_A_to_B``) of the SynthRAD ResNet runs.

    This is the 1-channel variant of the ``MyUNet_plus`` topology used by the
    metric-scored ``2025-NECSR_ResNetSynthRAD_VJ016_*`` checkpoints: a U-Net
    made of Triplet-Attention residual blocks with triangle skip fusions and a
    final tanh activation (``act=True``).
    """

    def __init__(self, in_ch=32, act=True):
        super().__init__()
        self.resblock1 = Residual_block(1, 2 * in_ch)
        self.resblock2 = Residual_block(2 * in_ch, 4 * in_ch)
        self.resblock2_2 = Residual_block(4 * in_ch, 4 * in_ch)
        self.resblock3 = Residual_block(4 * in_ch, 8 * in_ch)
        self.resblock3_2 = Residual_block(8 * in_ch, 8 * in_ch)
        self.resblock4 = Residual_block(8 * in_ch, 16 * in_ch)
        self.resblock5 = Residual_block(16 * in_ch, 8 * in_ch)
        self.resblock5_2 = Residual_block(8 * in_ch, 8 * in_ch)
        self.resblock6 = Residual_block(8 * in_ch, 4 * in_ch)
        self.resblock6_2 = Residual_block(4 * in_ch, 4 * in_ch)
        self.resblock7 = Residual_block(4 * in_ch, 2 * in_ch)
        self.lastconv = nn.Conv2d(2 * in_ch, 1, 1, padding=0)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                 nn.Conv2d(16 * in_ch, 8 * in_ch, 1, padding=0))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                 nn.Conv2d(8 * in_ch, 4 * in_ch, 1, padding=0))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                 nn.Conv2d(4 * in_ch, 2 * in_ch, 1, padding=0))
        self.act = act
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.resblock1(x)
        p1 = self.pool1(x1)
        x2 = self.resblock2(p1)
        x2 = self.resblock2_2(x2)
        p2 = self.pool2(x2)
        x3 = self.resblock3(p2)
        x3 = self.resblock3_2(x3)
        p3 = self.pool3(x3)
        x4 = self.resblock4(p3)
        u1 = self.up1(x4)
        merge1 = torch.cat([u1, x3], dim=1)
        x5 = self.resblock5(merge1)
        x5 = self.resblock5_2(x5)
        u2 = self.up2(x5)
        merge2 = torch.cat([u2, x2], dim=1)
        x6 = self.resblock6(merge2)
        x6 = self.resblock6_2(x6)
        u3 = self.up3(x6)
        merge3 = torch.cat([u3, x1], dim=1)
        x7 = self.resblock7(merge3)
        out = self.lastconv(x7)
        if self.act:
            out = self.tanh(out)
        return out