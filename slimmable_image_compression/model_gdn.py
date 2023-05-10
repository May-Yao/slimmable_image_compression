import math
import sys

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models.google import CompressionModel
from compressai.models.utils import conv, deconv
from slimmable_ops import (SlimmableConv2d, SlimmableConvTranspose2d,
                           SlimmableLinear, SwitchableBatchNorm2d,
                           SwitchableGDN)

width_mult_list = [0.67, 1]

class Slimable_Modelgdn(CompressionModel):
    def __init__(self, N = 192, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.channels = [
            int(N * width_mult) for width_mult in width_mult_list]

        self.g_a = nn.Sequential(
            SlimmableConv2d(
                    [3 for _ in range(len(self.channels))], self.channels, 5, 2, 2,bias=False),
            SwitchableGDN(self.channels),
            SlimmableConv2d(self.channels, self.channels, 5, 2, 2),
            SwitchableGDN(self.channels),
            SlimmableConv2d(self.channels, self.channels, 5, 2, 2),
            SwitchableGDN(self.channels),
            SlimmableConv2d(self.channels, self.channels, 5, 2, 2),
        )

        self.g_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )
        self.latent_transform = nn.Sequential(
            deconv(self.channels[0], N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(N, N, stride=1, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256, stride=1, kernel_size=3)
        )

        self.entropy_bottleneck_fea = EntropyBottleneck(self.channels[0])

    def forward(self, x):
        y = self.g_a(x)
        ch = y.shape[1]
        if ch == self.channels[0]:
            min_y_hat, min_y_likelihoods = self.entropy_bottleneck_fea(y)
            fea = self.latent_transform(min_y_hat)
            return {
            "y_hat": min_y_hat,
            "fea": fea,
            "likelihoods": {"y": min_y_likelihoods,},
            }
            
        else:
            y_hat, y_likelihoods = self.entropy_bottleneck(y)
            x_hat = self.g_s(y_hat)
            return {
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods,},
            }


class Slimable_HyperModelgdn(CompressionModel):
    def __init__(self, N = 192, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.channels = [
            int(N * width_mult) for width_mult in width_mult_list]

        self.g_a = nn.Sequential(
            SlimmableConv2d(
                    [3 for _ in range(len(self.channels))], self.channels, 5, 2, 2,bias=False),
            SwitchableGDN(self.channels),
            SlimmableConv2d(self.channels, self.channels, 5, 2, 2),
            SwitchableGDN(self.channels),
            SlimmableConv2d(self.channels, self.channels, 5, 2, 2),
            SwitchableGDN(self.channels),
            SlimmableConv2d(self.channels, self.channels, 5, 2, 2),
        )

        self.g_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )
        self.h_a = nn.Sequential(
                SlimmableConv2d(self.channels, self.channels, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                SlimmableConv2d(self.channels, self.channels, 5, 2, 2),
                nn.LeakyReLU(inplace=True),
                SlimmableConv2d(self.channels, self.channels, 5, 2, 2),
            )

        self.h_s = nn.Sequential(
                SlimmableConvTranspose2d(self.channels, self.channels, 5, stride=2, padding=2, output_padding = 1),
                nn.LeakyReLU(inplace=True),
                SlimmableConvTranspose2d(self.channels, self.channels, 5, stride=2, \
                    padding=2, output_padding = 1, bias=False),
                nn.LeakyReLU(inplace=True),
                SlimmableConvTranspose2d(self.channels, self.channels, \
                    3, stride=1, padding=1, output_padding = 0),
            )

        self.latent_transform = nn.Sequential(
            deconv(self.channels[0], N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(N, N, stride=1, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(N, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            deconv(N, 256, stride=1, kernel_size=3)
        )
        self.entropy_bottleneck_fea = EntropyBottleneck(self.channels[0])
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        ch = y.shape[1]        
        if ch == self.channels[0]:
            min_z_hat, min_z_likelihoods = self.entropy_bottleneck_fea(z)
            min_scales_hat = self.h_s(min_z_hat)
            min_y_hat, min_y_likelihoods = self.gaussian_conditional(y, min_scales_hat)
            fea = self.latent_transform(min_y_hat)
            return {
            "y_hat": min_y_hat,
            "fea": fea,
            "likelihoods": {"y": min_y_likelihoods, "z":min_z_likelihoods},
            }
            
        else:
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            scales_hat = self.h_s(z_hat)
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
            x_hat = self.g_s(y_hat)
            
            return {
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z":z_likelihoods},
            }




            


