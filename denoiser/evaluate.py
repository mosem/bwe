# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss
import os
from concurrent.futures import ProcessPoolExecutor
import logging

from torch.nn import functional as F
from torchaudio.transforms import Spectrogram
import wandb

from pesq import pesq
from pystoi import stoi
import torch
from .stft_loss import stft

from .enhance import get_estimate
from . import distrib
from .resample import upsample2
from .utils import bold, LogProgress, convert_spectrogram_to_heatmap

logger = logging.getLogger(__name__)


def evaluate(args, model, data_loader, epoch):
    total_pesq = 0
    total_stoi = 0
    total_lsd = 0
    total_cnt = 0
    updates = 5
    model.eval()
    include_ft = args.experiment.features_model.include_ft if hasattr(args.experiment, "features_model") else False
    files_to_log = []

    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                (noisy, noisy_path), (clean, clean_path) = data
                filename = os.path.basename(clean_path[0]).rstrip('_clean.wav')
                noisy = noisy.to(args.device)
                clean = clean.to(args.device)
                if args.wandb.n_files_to_log == -1 or len(files_to_log) < args.wandb.n_files_to_log:
                    files_to_log.append(filename)
                # If device is CPU, we do parallel evaluation in each CPU worker.
                if args.device == 'cpu':
                    pendings.append(
                        pool.submit(estimate_and_run_metrics, clean, model, noisy, args, filename, include_ft))
                else:
                    estimate = get_estimate(model, noisy)[0] if include_ft else get_estimate(model, noisy)
                    noisy = upsample_noisy(args, noisy)
                    estimate = estimate.cpu()
                    clean = clean.cpu()
                    pendings.append(
                        pool.submit(run_metrics, clean, estimate, noisy.shape[-1], args, filename))
                total_cnt += clean.shape[0]

        for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
            pesq_i, stoi_i, snr_i, lsd_i, estimate_i, filename_i = pending.result()
            if filename_i in files_to_log:
                log_to_wandb(estimate_i, pesq_i, stoi_i, snr_i, lsd_i, filename_i, epoch, args.experiment.sample_rate)
            total_pesq += pesq_i
            total_stoi += stoi_i
            total_lsd += lsd_i

    metrics = [total_pesq, total_stoi, total_lsd]
    avg_pesq, avg_stoi, avg_lsd = distrib.average([m/total_cnt for m in metrics], total_cnt)
    logger.info(bold(f'Test set performance:PESQ={avg_pesq}, STOI={avg_stoi}, LSD={avg_lsd}.'))
    return avg_pesq, avg_stoi, avg_lsd


def log_to_wandb(signal, pesq, stoi, snr, lsd, filename, epoch, sr):
    spectrogram_transform = Spectrogram()
    enhanced_spectrogram = spectrogram_transform(signal).log2()[0, :, :].numpy()
    enhanced_spectrogram_wandb_image = wandb.Image(convert_spectrogram_to_heatmap(enhanced_spectrogram), caption=filename)
    enhanced_wandb_audio = wandb.Audio(signal.squeeze().numpy(), sample_rate=sr, caption=filename)
    wandb.log({f'test samples/{filename}/pesq': pesq,
               f'test samples/{filename}/stoi': stoi,
               f'test samples/{filename}/snr': snr,
               f'test samples/{filename}/lsd': lsd,
               f'test samples/{filename}/spectrogram': enhanced_spectrogram_wandb_image,
               f'test samples/{filename}/audio': enhanced_wandb_audio},
              step=epoch)


def estimate_and_run_metrics(clean, model, noisy, args, filename, include_ft=False):
    estimate = get_estimate(model, noisy)
    noisy = upsample_noisy(args, noisy)
    return run_metrics(clean, estimate[0] if include_ft else estimate, noisy.shape[-1], args, filename)


def run_metrics(clean, estimate, noisy_len, args, filename):
    clean, estimate = pad_signals_to_noisy_length(clean, noisy_len, estimate)
    pesq, stoi, snr, lsd = get_metrics(clean, estimate, args.experiment.sample_rate)
    return pesq, stoi, snr, lsd, estimate, filename


def get_metrics(clean, estimate, sr):
    clean = clean.squeeze(dim=1)
    estimate = estimate.squeeze(dim=1)
    estimate_numpy = estimate.numpy()
    clean_numpy = clean.numpy()
    pesq = get_pesq(clean_numpy, estimate_numpy, sr=sr)
    stoi = get_stoi(clean_numpy, estimate_numpy, sr=sr)
    snr = get_snr(estimate, clean).item()
    lsd = get_lsd(estimate, clean).item()
    return pesq, stoi, snr, lsd


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        tmp = pesq(sr, ref_sig[i], out_sig[i], 'wb')  # from pesq
        pesq_val += tmp
    return pesq_val


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val


# def get_snr(signal, clean):
#     noise = clean - signal
#     return (signal**2).mean()/(noise**2).mean()


# get_snr and get_lsd are taken from: https://github.com/nanahou/metric/blob/master/measure_SNR_LSD.py

def get_snr(x, y):
    """
       Compute SNR (signal to noise ratio)
       Arguments:
           x: vector (torch.Tensor), enhanced signal [B,T]
           y: vector (torch.Tensor), reference signal(ground truth) [B,T]
    """
    ref = torch.pow(y, 2)
    diff = torch.pow(x - y, 2)

    ratio = torch.sum(ref,dim=-1) / torch.sum(diff, dim=-1)
    value = 10 * torch.log10(ratio)

    return value


def get_lsd(x, y):
    """
       Compute LSD (log spectral distance)
       Arguments:
           x: vector (torch.Tensor), enhanced signal [B,T]
           y: vector (torch.Tensor), reference signal(ground truth) [B,T]
    """

    fft_size = 1024
    shift_size = 120
    win_length = 600
    window = torch.hann_window(win_length)

    X = stft(x, fft_size, shift_size, win_length, window)
    Y = stft(y, fft_size, shift_size, win_length, window)


    diff = torch.pow(X - Y, 2)

    sum_freq = torch.sqrt(torch.sum(diff, dim=-1) / diff.size(-1))
    value = torch.sum(sum_freq, dim=-1) / sum_freq.size(-1)

    return value


def upsample_noisy(args, noisy):
    if args.experiment.scale_factor == 2:
        noisy = upsample2(noisy)
    elif args.experiment.scale_factor == 4:
        noisy = upsample2(noisy)
        noisy = upsample2(noisy)
    return noisy


def pad_signals_to_noisy_length(clean, noisy_len, enhanced):
    if clean.shape[-1] < noisy_len:
        clean = F.pad(clean, (0, noisy_len - clean.shape[-1]))
    if enhanced.shape[-1] < noisy_len:
        enhanced = F.pad(enhanced, (0, noisy_len - enhanced.shape[-1]))

    return clean, enhanced