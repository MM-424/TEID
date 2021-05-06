#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2021/03/27
    Description:
"""
import torch.nn as nn
from thirdparty.deform_conv import DeformConv


def offset_conv(inC, kH, kW, num_deformable_groups):
    conv = nn.Conv2d(
        inC,
        num_deformable_groups * 2 * kH * kW,
        kernel_size=(kH, kW),
        stride=(1, 1),
        padding=(1, 1),
        bias=False)
    return conv


def build_DeformConv(inC, outC, kH, kW, num_deformable_groups):
    conv = DeformConv(
        inC,
        outC, (kH, kW),
        stride=1,
        padding=1,
        num_deformable_groups=num_deformable_groups)
    return conv
