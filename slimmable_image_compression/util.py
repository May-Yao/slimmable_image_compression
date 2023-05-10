import cv2
import numpy as np
from torchvision.utils import make_grid
import math
import torch
from PIL import Image
from pytorch_msssim import ms_ssim
import torch.nn as nn

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net, imgs):
    size = imgs.size()
    num_pixels = size[0] * size[2] * size[3]
    bpp = sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()
    
    return bpp


class Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, alpha=20, beta=0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target, f_true):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["ms_ssim"] = compute_msssim(output["x_hat"], target)
        out["mse_f"] = self.mse(output["f"], f_true)
        out["distortion"] = out["mse_loss"] + self.alpha * out["mse_f"] + self.beta * out["ms_ssim"]
        out["loss"] = self.lmbda * 255 ** 2 * out["distortion"] + out["bpp_loss"]

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

