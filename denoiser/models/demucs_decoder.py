import math

from torch import nn

from denoiser.models.config_dataclasses import DemucsDecoderConfig
from denoiser.models.demucs import rescale_module
from denoiser.resample import downsample2
from denoiser.utils import capture_init

class DemucsDecoder(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.

    """

    @capture_init
    def __init__(self, demucs_config: DemucsDecoderConfig):

        super().__init__()
        if demucs_config.resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chout = demucs_config.chout
        self.hidden = demucs_config.hidden
        self.depth = demucs_config.depth
        self.kernel_size = demucs_config.kernel_size
        self.stride = demucs_config.stride
        self.resample = demucs_config.resample
        self.scale_factor = demucs_config.scale_factor

        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if demucs_config.glu else nn.ReLU()
        ch_scale = 2 if demucs_config.glu else 1
        hidden, chout = demucs_config.hidden, demucs_config.chout

        for index in range(demucs_config.depth):
            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, demucs_config.kernel_size, demucs_config.stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            hidden = min(int(demucs_config.growth * hidden), demucs_config.max_hidden)

        if demucs_config.rescale:
            rescale_module(self, reference=demucs_config.rescale)

    def estimate_output_length(self, input_length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = input_length
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    def forward(self, signal, skips=None):
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)

        x = signal

        for decode in self.decoder:
            if skips is not None:
                skip = skips.pop(-1)
                x = x + skip
            x = decode(x)

        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)
        else:
            pass

        return x