import torch
from torch import nn

from denoiser.utils import capture_init
from denoiser.models.modules import ResnetBlock, PixelShuffle1D, WNConv1d

import logging
logger = logging.getLogger(__name__)


class SpCombModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.sub_pixel_layer = PixelShuffle1D(upscale_factor=2)

    def forward(self, low_pass_signal, high_pass_signal):
        """

        :param low_pass_signal: low sample rate signal [B,C,T]
        :param high_pass_signal: low sample rate signal [B,C,T]
        :return: high sample rate signal [B,C,2*T]
        """

        full_band_signal = torch.cat((low_pass_signal, high_pass_signal), dim=1) # [B,2*C,T]
        full_band_signal = self.sub_pixel_layer(full_band_signal) # [B,C,2*T]
        return full_band_signal


class ConditionalBias(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, signal):
        """

        :param signal: [B,C,T]
        :return:
        """
        bias = signal.permute(0,2,1)
        bias = self.linear(bias)
        bias = bias.permute(0,2,1)
        return bias


class LowPassModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conditional_bias = ConditionalBias(in_channels, out_channels)

        low_pass_wrapper_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(out_channels, out_channels, kernel_size=7, padding=0)
        ]
        self.low_pass_module = nn.Sequential(*low_pass_wrapper_layer)

    def forward(self, signal):
        bias = self.conditional_bias(signal)
        return self.low_pass_module(signal + bias)


class HighPassModule(nn.Module):

    def __init__(self, depth, residual_layers, hidden_channels, out_channels):
        super().__init__()
        high_pass_module_list = []

        high_pass_wrapper_layer_1 = [
            nn.ReflectionPad1d(3),
            WNConv1d(1, hidden_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        high_pass_module_list += high_pass_wrapper_layer_1

        for i in range(depth):
            for j in range(residual_layers):
                high_pass_module_list += [ResnetBlock(hidden_channels, dilation=3 ** j)]

        high_pass_wrapper_layer_2 = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(hidden_channels, out_channels, kernel_size=7, padding=0)
        ]
        high_pass_module_list += high_pass_wrapper_layer_2

        self.high_pass_module = nn.Sequential(*high_pass_module_list)

    def forward(self, signal):
        return self.high_pass_module(signal)


class Generator(nn.Module):

    @capture_init
    def __init__(self, depth=4, residual_layers=3, hidden_channels=64, out_channels=4, scale_factor=2):
        super().__init__()
        self.depth = depth
        self.scale_factor = scale_factor

        self.low_pass_module = LowPassModule(1, out_channels)

        self.high_pass_module = HighPassModule(depth, residual_layers, hidden_channels, out_channels)

        self.comb_module = SpCombModule()

        fine_tune_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(out_channels, 1, kernel_size=7, padding=0)
        ]

        self.fine_tune_module = nn.Sequential(*fine_tune_layer)

    def estimate_output_length(self, input_length):
        return input_length*self.scale_factor

    def forward(self, signal):
        # logger.info(f'signal shape: {signal.shape}')
        low_pass_signal = self.low_pass_module(signal)
        high_pass_signal = self.high_pass_module(signal)

        full_band_signal = self.comb_module(low_pass_signal, high_pass_signal)
        # logger.info(f'full_band_signal shape: {full_band_signal.shape}')

        full_band_signal = self.fine_tune_module(full_band_signal)
        # logger.info(f'full_band_signal shape: {full_band_signal.shape}')

        return full_band_signal
