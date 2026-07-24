"""Compact 2D U-Nets for super-resolution and segmentation.

One small, readable U-Net implementation is shared by both tasks:
  * SR U-Net: 1-channel in, 1-channel out, dropout kept for Monte Carlo
    uncertainty (enable_mc_dropout() turns dropout on at inference).
  * Segmentation U-Net: 1-channel in, 1-channel logits out (binary tumor).

Kept deliberately small (base=32) so training is a few-hour job on a free GPU
and inference is cheap enough to benchmark on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """A small 4-level U-Net.

    Args:
        in_ch, out_ch: channel counts.
        base: width of the first level (channels double each level down).
        dropout: dropout prob in each block (used for MC-dropout uncertainty).
        residual: if True the network predicts a residual added to the input
            (good for super-resolution, where output ~= input + detail).
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32,
                 dropout: float = 0.0, residual: bool = False):
        super().__init__()
        self.residual = residual
        b = base
        self.d1 = DoubleConv(in_ch, b, dropout)
        self.d2 = DoubleConv(b, b * 2, dropout)
        self.d3 = DoubleConv(b * 2, b * 4, dropout)
        self.bott = DoubleConv(b * 4, b * 8, dropout)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.u3 = DoubleConv(b * 8, b * 4, dropout)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.u2 = DoubleConv(b * 4, b * 2, dropout)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.u1 = DoubleConv(b * 2, b, dropout)
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        cb = self.bott(self.pool(c3))
        u3 = self.u3(torch.cat([self.up3(cb), c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        y = self.out(u1)
        if self.residual:
            y = y + x
        return y


def sr_unet(base: int = 32, dropout: float = 0.2) -> UNet:
    """Super-resolution network: residual, dropout on (for MC uncertainty)."""
    return UNet(1, 1, base=base, dropout=dropout, residual=True)


def seg_unet(base: int = 32) -> UNet:
    """Segmentation network: 1-channel logits, no residual, no dropout."""
    return UNet(1, 1, base=base, dropout=0.0, residual=False)


def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode but re-enable Dropout layers.

    This is the standard Monte Carlo dropout setup: BatchNorm uses running
    stats (eval), while Dropout stays stochastic so repeated forward passes
    give a distribution we can turn into an uncertainty map.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
