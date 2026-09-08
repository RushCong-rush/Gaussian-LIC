"""Export masked spatial LPIPS from the bundled weights, without downloading models."""

from pathlib import Path

import torch

from masked_lpips import SpatialLPIPS


if __name__ == "__main__":
    torch.set_num_threads(2)
    directory = Path(__file__).resolve().parent
    model = SpatialLPIPS(directory / "lpips_alex.pt").eval()
    images = torch.zeros(1, 3, 64, 96)
    mask = torch.ones(1, 1, 64, 96)
    with torch.inference_mode():
        exported = torch.jit.trace(model, (images, images, mask))
        exported.save(str(directory / "lpips_alex_valid.pt"))
