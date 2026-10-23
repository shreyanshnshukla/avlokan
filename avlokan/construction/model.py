"""U-Net for built-up segmentation, sized for a 4 GB laptop GPU (RTX 3050).

~1.9M parameters with base=24. Input: 6 standardised Sentinel-2 bands.
Output: one logit per pixel (built-up or not).
"""

from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 6, base: int = 24, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.down.append(_block(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottom = nn.Sequential(_block(chs[-2], chs[-1]), nn.Dropout2d(dropout))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for ch_hi, ch_lo in zip(chs[::-1][:-1], chs[::-1][1:]):
            self.up.append(nn.ConvTranspose2d(ch_hi, ch_lo, 2, stride=2))
            self.dec.append(_block(ch_lo * 2, ch_lo))
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for d in self.down:
            x = d(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x).squeeze(1)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """BCE + soft Dice, ignoring no-data pixels. Dice helps because built-up
    is a minority class at the city fringe."""
    v = valid.float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * v).sum() / v.sum().clamp(min=1)
    p = torch.sigmoid(logits) * v
    t = target * v
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return bce + dice


@torch.no_grad()
def iou_counts(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, thr: float = 0.5):
    pred = (torch.sigmoid(logits) > thr) & valid
    t = (target > 0.5) & valid
    return int((pred & t).sum()), int((pred | t).sum())
