import torch
import torch.nn as nn 
import torch.nn.functional as F
import spconv.pytorch as spconv
import math


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(
            image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([
            image_features, atrous_block1, atrous_block6,
            atrous_block12, atrous_block18], dim=1))
        return net


class DenseBasicBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DenseBasicBlock3D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.in_conv = nn.Conv3d(self.in_channel, self.out_channel, kernel_size=1, 
            stride=1, padding=0, bias=False)
        self.in_bn = nn.BatchNorm3d(self.out_channel)

        self.conv1 = nn.Conv3d(self.in_channel, self.out_channel, kernel_size=3, 
            stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(self.out_channel, self.out_channel, kernel_size=3, 
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.out_channel)

    def forward(self, x):
        identity = self.in_conv(x)
        identity = self.in_bn(identity)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out


class SparseBasicBlock3D(spconv.SparseModule):
    def __init__(self, in_channel, out_channel, indice_key):
        super(SparseBasicBlock3D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.in_conv = spconv.SubMConv3d(self.in_channel, self.out_channel, kernel_size=1, 
            stride=1, padding=0, indice_key=indice_key+"inconv", bias=False)
        self.in_bn = nn.BatchNorm1d(self.out_channel)

        self.conv1 = spconv.SubMConv3d(self.in_channel, self.out_channel, kernel_size=3, 
            stride=1, padding=1, indice_key=indice_key+"conv1", bias=False)
        self.bn1 = nn.BatchNorm1d(self.out_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = spconv.SubMConv3d(self.out_channel, self.out_channel, kernel_size=3, 
            stride=1, padding=1, indice_key=indice_key+"conv2", bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_channel)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.in_conv(x)
        identity = identity.replace_feature(self.in_bn(identity.features))
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu1(out.features))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu2(out.features + identity.features))
        return out


class DenseVFE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DenseVFE, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.block1 = DenseBasicBlock3D(self.in_channel, self.out_channel)
        self.block2 = DenseBasicBlock3D(self.out_channel, self.out_channel)
        self.block3 = DenseBasicBlock3D(self.out_channel * 2, self.out_channel)
        self.block4 = DenseBasicBlock3D(self.out_channel * 2, self.out_channel)
        self.downsample = nn.Sequential(
            nn.Conv3d(self.out_channel, self.out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(self.out_channel, self.out_channel, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(in_channel=self.out_channel, depth=self.out_channel)

    def forward(self, x):
        x1 = self.block1(x)
        x_down = self.downsample(x1)
        x2 = self.block2(x_down)
        B, C, H, W, D = x2.shape
        x2_bev = x2.permute(0, 4, 1, 2, 3).reshape(B*D, -1, H, W)
        x2_bev = self.aspp(x2_bev)
        x2_vox = x2_bev.reshape(B, D, -1, H, W).permute(0, 2, 3, 4, 1)
        x3 = torch.cat((x2_vox, x2), dim=1)
        x3 = self.block3(x3)
        x_up = self.upsample(x3)
        x4 = torch.cat((x_up, x1), dim=1)
        out = self.block4(x4)
        return out


class SparseVFE(nn.Module):
    def __init__(self, in_channel, out_channel, indice_key):
        super(SparseVFE, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.block1 = SparseBasicBlock3D(self.in_channel, self.out_channel, indice_key+"block1")
        self.block2 = SparseBasicBlock3D(self.out_channel, self.out_channel, indice_key+"block2")
        self.block3 = SparseBasicBlock3D(self.out_channel, self.out_channel, indice_key+"block3")
        self.block4 = SparseBasicBlock3D(self.out_channel, self.out_channel, indice_key+"block4")

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x2_add = x2.replace_feature(x1.features + x2.features)
        x3 = self.block3(x2_add)
        x3_add = x3.replace_feature(x2.features + x3.features)
        x4 = self.block4(x3_add)
        return x4