"""
PiDiNet inference model used by HF-GS edge supervision.

This is a minimal, self-contained adaptation of the official PiDiNet
implementation by Zhuo Su and Wenzhe Liu:
https://github.com/hellozhuo/pidinet

The bundled ``table5_pidinet.pth`` weights use the full PiDiNet model with the
CARv4 pixel-difference convolutions, compact dilation, and spatial attention.
See LICENSE in this directory for the upstream research-use terms.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


PDCFunction = Callable[..., torch.Tensor]


def _create_pdc(op_type: str) -> PDCFunction:
    """Create one of PiDiNet's pixel-difference convolution functions."""
    if op_type == "cv":
        return F.conv2d
    if op_type == "cd":
        def central_difference(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None = None,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
        ) -> torch.Tensor:
            center_weight = weight.sum(dim=(2, 3), keepdim=True)
            center = F.conv2d(x, center_weight, stride=stride, padding=0, groups=groups)
            regular = F.conv2d(
                x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
            )
            return regular - center

        return central_difference
    if op_type == "ad":
        def angular_difference(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None = None,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
        ) -> torch.Tensor:
            shape = weight.shape
            flattened = weight.reshape(shape[0], shape[1], -1)
            rotated = flattened[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
            difference_weight = (flattened - rotated).reshape(shape)
            return F.conv2d(
                x,
                difference_weight,
                bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

        return angular_difference
    if op_type == "rd":
        def radial_difference(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None = None,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
        ) -> torch.Tensor:
            del padding
            shape = weight.shape
            flattened = weight.reshape(shape[0], shape[1], -1)
            expanded = weight.new_zeros((shape[0], shape[1], 25))
            expanded[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = flattened[:, :, 1:]
            expanded[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -flattened[:, :, 1:]
            kernel = expanded.reshape(shape[0], shape[1], 5, 5)
            return F.conv2d(
                x,
                kernel,
                bias,
                stride=stride,
                padding=2 * dilation,
                dilation=dilation,
                groups=groups,
            )

        return radial_difference
    raise ValueError("Unsupported PiDiNet PDC operation: {}".format(op_type))


class PDCConv2d(nn.Module):
    """Convolution layer whose kernel is interpreted by a PDC operation."""

    def __init__(
        self,
        pdc: PDCFunction,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.pdc = pdc
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            nn.init.uniform_(self.bias, -1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pdc(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CSAM(nn.Module):
    """Compact spatial-attention module."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.sigmoid(self.conv2(self.conv1(self.relu1(x))))
        return x * attention


class CDCM(nn.Module):
    """Compact dilation-convolution module."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, 3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, 3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, 3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, 3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.relu1(x))
        return self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + self.conv2_4(x)


class MapReduce(nn.Module):
    """Reduce an intermediate feature tensor to one edge channel."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PDCBlock(nn.Module):
    """Residual PiDiNet block."""

    def __init__(
        self,
        pdc: PDCFunction,
        inplane: int,
        outplane: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, outplane, kernel_size=1)
        self.conv1 = PDCConv2d(
            pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False
        )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, outplane, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv2(self.relu2(self.conv1(x)))
        if self.stride > 1:
            x = self.shortcut(x)
        return y + x


class PiDiNet(nn.Module):
    """Full PiDiNet with CARv4, dilation=24, and spatial attention."""

    def __init__(self, inplane: int, pdcs: list[PDCFunction], dilation_channels: int = 24) -> None:
        super().__init__()
        self.init_block = PDCConv2d(pdcs[0], 3, inplane, kernel_size=3, padding=1)

        self.block1_1 = PDCBlock(pdcs[1], inplane, inplane)
        self.block1_2 = PDCBlock(pdcs[2], inplane, inplane)
        self.block1_3 = PDCBlock(pdcs[3], inplane, inplane)

        stage2_channels = inplane * 2
        self.block2_1 = PDCBlock(pdcs[4], inplane, stage2_channels, stride=2)
        self.block2_2 = PDCBlock(pdcs[5], stage2_channels, stage2_channels)
        self.block2_3 = PDCBlock(pdcs[6], stage2_channels, stage2_channels)
        self.block2_4 = PDCBlock(pdcs[7], stage2_channels, stage2_channels)

        stage3_channels = stage2_channels * 2
        self.block3_1 = PDCBlock(pdcs[8], stage2_channels, stage3_channels, stride=2)
        self.block3_2 = PDCBlock(pdcs[9], stage3_channels, stage3_channels)
        self.block3_3 = PDCBlock(pdcs[10], stage3_channels, stage3_channels)
        self.block3_4 = PDCBlock(pdcs[11], stage3_channels, stage3_channels)

        self.block4_1 = PDCBlock(pdcs[12], stage3_channels, stage3_channels, stride=2)
        self.block4_2 = PDCBlock(pdcs[13], stage3_channels, stage3_channels)
        self.block4_3 = PDCBlock(pdcs[14], stage3_channels, stage3_channels)
        self.block4_4 = PDCBlock(pdcs[15], stage3_channels, stage3_channels)

        fuse_channels = [inplane, stage2_channels, stage3_channels, stage3_channels]
        self.dilations = nn.ModuleList(
            [CDCM(channels, dilation_channels) for channels in fuse_channels]
        )
        self.attentions = nn.ModuleList([CSAM(dilation_channels) for _ in fuse_channels])
        self.conv_reduces = nn.ModuleList([MapReduce(dilation_channels) for _ in fuse_channels])
        self.classifier = nn.Conv2d(4, 1, kernel_size=1)
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        height, width = x.shape[2:]
        x = self.init_block(x)

        x1 = self.block1_3(self.block1_2(self.block1_1(x)))
        x2 = self.block2_4(self.block2_3(self.block2_2(self.block2_1(x1))))
        x3 = self.block3_4(self.block3_3(self.block3_2(self.block3_1(x2))))
        x4 = self.block4_4(self.block4_3(self.block4_2(self.block4_1(x3))))

        fused = [
            attention(dilation(stage))
            for stage, dilation, attention in zip(
                (x1, x2, x3, x4), self.dilations, self.attentions
            )
        ]
        outputs = [
            F.interpolate(reducer(feature), (height, width), mode="bilinear", align_corners=False)
            for reducer, feature in zip(self.conv_reduces, fused)
        ]
        outputs.append(self.classifier(torch.cat(outputs, dim=1)))
        return [torch.sigmoid(output) for output in outputs]


def build_pidinet() -> PiDiNet:
    """Build the exact architecture expected by the official table-5 checkpoint."""
    carv4 = ["cd", "ad", "rd", "cv"] * 4
    return PiDiNet(60, [_create_pdc(op_type) for op_type in carv4], dilation_channels=24)
