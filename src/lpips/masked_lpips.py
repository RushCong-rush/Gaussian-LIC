"""Spatial LPIPS v0.1 on valid pixels; inputs are NCHW RGB in [0, 1]."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import alexnet


class SpatialLPIPS(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        state = torch.jit.load(str(checkpoint), map_location="cpu").state_dict()
        self.layers = alexnet(weights=None).features
        self.layers.load_state_dict({key.removeprefix("net.layers."): value
                                     for key, value in state.items() if key.startswith("net.layers.")})
        self.register_buffer("mean", state["net.mean"])
        self.register_buffer("std", state["net.std"])
        self.linear = nn.ModuleList([nn.Conv2d(channels, 1, 1, bias=False)
                                     for channels in (64, 192, 384, 256, 256)])
        for index, layer in enumerate(self.linear):
            layer.weight.data.copy_(state[f"lin.{index}.1.weight"])
        self.requires_grad_(False)

    def features(self, image):
        value = (image - self.mean) / self.std
        result = []
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index in (1, 4, 7, 9, 11):
                result.append(value / (value.square().sum(1, keepdim=True).sqrt() + 1e-10))
            if index == 11:
                break
        return result

    def forward(self, first, second, valid):
        # LPIPS v0.1 accepts RGB in [-1, 1]; invalid context is identical in both images.
        first = (first * valid) * 2 - 1
        second = (second * valid) * 2 - 1
        value = first.new_zeros(())
        for a, b, layer in zip(self.features(first), self.features(second), self.linear):
            distance = F.interpolate(layer((a - b).square()), size=valid.shape[-2:],
                                     mode="bilinear", align_corners=False)
            value = value + (distance * valid).sum() / valid.sum()
        return value

