import math
import torch
from torch import nn
from torch.nn import Conv1d, functional as F, Conv2d, AvgPool1d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import GRU
from torch.nn.modules.normalization import LayerNorm
from denoiser.utils import get_padding, init_weights, capture_init
from denoiser.resample import downsample2, upsample2

import logging
logger = logging.getLogger(__name__)
LRELU_SLOPE = 0.1

# Demucs related

class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def estimate_output_length(self, length):
        return length

    def forward(self, x, hidden=None):
        x = x.permute(2, 0, 1)
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


class OneDimDualTransformer(nn.Module):
    def __init__(self, dim,
                 frame_size=8,
                 n_head=4,
                 n_layers=6):
        super().__init__()
        self.dim = dim
        self.frame_size = frame_size
        self.frame_shift = frame_size // 2
        self.n_head = n_head
        self.n_layers = n_layers

        self.signalPreProcessor = TorchSignalToFrames(frame_size=self.frame_size,
                                                      frame_shift=self.frame_shift)
        self.attention = DualTransformer(dim, dim, nhead=self.n_head,
                                         num_layers=self.n_layers)
        self.ola = TorchOLA(self.frame_shift)

    def estimate_output_length(self, length):
        n_frames = math.ceil((length - self.frame_size) / self.frame_shift + 1)
        length = (n_frames - 1) * self.frame_shift + self.frame_size
        return length

    def forward(self, x):
        x = self.signalPreProcessor(x)
        x = self.attention(x)
        x = self.ola(x)
        return x


# Caunet related


class TorchSignalToFrames(object):
    r"""Chunks a tensor into frames
         required input shape is [1, 1, -1]
         input params:    (frame_size: window_size,  frame_shift: overlap(samples))
         output:   [1, 1, num_frames, frame_size]
    """

    def __init__(self, frame_size=512, frame_shift=256):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        nframes = math.ceil((sig_len - self.frame_size) / self.frame_shift + 1)
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a



class TorchOLA(nn.Module):
    r"""Performs overlap-and-add on gpu using torch tensor
        required input is tensor
        perform frames into signal
        used in the output of network
    """

    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device,
                          requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        self.gru = GRU(d_model, d_model, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)
        if bidirectional:
            # self.linear2 = Linear(d_model*2*2, d_model)
            self.linear2 = GRU(d_model * 2, d_model, bidirectional=False)
        else:
            self.linear2 = Linear(d_model * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        self.linear2.flatten_parameters()
        src2, h_n1 = self.linear2(self.dropout(self.activation(out)))
        del h_n1
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DualTransformer(nn.Module):
    """
    Deep duaL-path RNN.
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, input_size, output_size, nhead=4, dropout=0, num_layers=1):
        super(DualTransformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(
                TransformerEncoderLayer(d_model=input_size // 2, nhead=nhead, dropout=dropout, bidirectional=True))
            self.col_trans.append(
                TransformerEncoderLayer(d_model=input_size // 2, nhead=nhead, dropout=dropout, bidirectional=True))
            self.row_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size // 2, output_size, 1)
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape
        # output = input
        output = self.input(input)
        for i in range(len(self.row_trans)):
            row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)  # [dim1, b*dim2, c]
            row_output = self.row_trans[i](row_input)  # [dim1, b*dim2, c]
            row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [b, c, dim2, dim1]
            row_output = self.row_norm[i](row_output)  # [b, c, dim2, dim1]
            output = output + row_output  # [b, c, dim2, dim1]

            col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)  # [dim2, b*dim1, c]
            col_output = self.col_trans[i](col_input)  # [dim2, b*dim1, c]
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()  # [b, c, dim2, dim1]
            col_output = self.col_norm[i](col_output)  # [b, c, dim2, dim1]
            output = output + col_output  # [b, c, dim2, dim1]

        del row_input, row_output, col_input, col_output
        output = self.output(output)  # [b, c, dim2, dim1]

        return output


# HiFi related

class HifiResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(HifiResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class HifiResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(HifiResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class HifiDiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(HifiDiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 64, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 128, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(128, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class HifiMultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(HifiMultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            HifiDiscriminatorP(2),
            HifiDiscriminatorP(3),
            HifiDiscriminatorP(5),
            HifiDiscriminatorP(7),
            HifiDiscriminatorP(11),
        ])
        self._init_args_kwargs = (None, None)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class HifiDiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(HifiDiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 32, 15, 1, padding=7)),
            norm_f(Conv1d(32, 32, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(32, 64, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(64, 128, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(128, 128, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(128, 128, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(128, 128, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(128, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class HifiMultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(HifiMultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            HifiDiscriminatorS(use_spectral_norm=True),
            HifiDiscriminatorS(),
            HifiDiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])
        self._init_args_kwargs = (None, None)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# Seanet related


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):
    @capture_init
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        self.num_D = num_D
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


class LaplacianDiscriminator(Discriminator):

    def forward(self, x):
        results = []
        for i, (key, disc) in enumerate(self.model.items()):
            if i == 0:  # insert total frequency range
                results.append(disc(x))
            elif i == self.num_D - 1:  # insert 0 - total frequency range / 2^num_D
                downsampled_x = downsample2(x)
                results.append(disc(downsampled_x))
            else:  # insert total_frequency_range / 2^i - total_frequency_range/2^(i-1)
                downsampled_x = downsample2(x)
                laplacian = x - upsample2(downsampled_x)
                results.append(disc(laplacian))
                x = downsampled_x
        return results


# Comb related

# from: https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class PixelUnshuffle1D(torch.nn.Module):
    """
    Inverse of 1D pixel shuffler
    Upscales channel length, downscales sample length
    "long" is input, "short" is output
    """

    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor

        x = x.contiguous().view([batch_size, long_channel_len, short_width, self.downscale_factor])
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x