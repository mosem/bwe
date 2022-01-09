import torch.nn as nn
from torchaudio.transforms import Resample
from denoiser.utils import capture_init

class Sinc(nn.Module):

    @capture_init
    def __init__(self, sample_rate=16_000, scale_factor=2):
        super().__init__()
        self.sample_rate = sample_rate
        self.scale_factor = scale_factor

        self.resample_transform = Resample(sample_rate/scale_factor, sample_rate)

    def estimate_output_length(self, length):
        return length*self.scale_factor

    def forward(self, x):
        return self.resample_transform(x)
