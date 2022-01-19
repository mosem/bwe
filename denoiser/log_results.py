import math
import os
import pandas as pd
import torchaudio
import wandb
from torchaudio.transforms import Spectrogram
import numpy as np

from .evaluate import get_metrics, get_snr, upsample_noisy, pad_signals_to_noisy_length, logger
from .audio import find_audio_files

import logging

from .utils import convert_spectrogram_to_heatmap

logger = logging.getLogger(__name__)

SPECTOGRAM_EPSILON = 1e-13

RESULTS_DF_FILENAME = 'filename'
RESULTS_DF_NOISY_SNR = 'noisy snr'
RESULTS_DF_ENHANCED_SNR = 'enhanced snr'
RESULTS_DF_PESQ = 'pesq'
RESULTS_DF_STOI = 'stoi'
RESULTS_DF_LSD = 'lsd'
RESULTS_DF_SISNR= 'sisnr'
RESULTS_DF_VISQOL= 'visqol'

HISTOGRAM_DF_RANGE = 'range'
HISTOGRAM_DF_N_SAMPLES = 'n_samples'
HISTOGRAM_DF_AVG_PESQ = 'avg pesq'
HISTOGRAM_DF_AVG_STOI = 'avg stoi'
HISTOGRAM_DF_AVG_LSD = 'avg lsd'
HISTOGRAM_DF_AVG_SISNR = 'avg sisnr'
HISTOGRAM_DF_AVG_VISQOL = 'avg visqol'

DEFAULT_HISTOGRAM_MAX_SNR = 40.0

WANDB_CUSTOM_CHART_NAME = "huji-dl-audio-lab/non-sorted-bar-chart"


def create_results_df(args):
    wandb_table = init_wandb_table()
    df = pd.DataFrame(columns=[RESULTS_DF_FILENAME, RESULTS_DF_NOISY_SNR, RESULTS_DF_ENHANCED_SNR, RESULTS_DF_PESQ,
                               RESULTS_DF_STOI, RESULTS_DF_LSD, RESULTS_DF_SISNR, RESULTS_DF_VISQOL])
    files = find_audio_files(args.samples_dir, progress=False)
    clean_paths = [str(data[0]) for data in files if '_hr' in str(data[0])]
    noisy_paths = [str(data[0]) for data in files if '_lr' in str(data[0])]
    enhanced_paths = [str(data[0]) for data in files if '_pr' in str(data[0])]
    for i, (clean_path, noisy_path, enhanced_path) in enumerate(zip(clean_paths, noisy_paths, enhanced_paths)):
        clean, clean_sr = torchaudio.load(clean_path)
        noisy, noisy_sr = torchaudio.load(noisy_path)
        enhanced, enhanced_sr = torchaudio.load(enhanced_path)

        noisy = upsample_noisy(args, noisy)

        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
        enhanced = enhanced.unsqueeze(0)

        clean, enhanced = pad_signals_to_noisy_length(clean, noisy.shape[-1], enhanced)

        noisy_snr = get_snr(noisy, clean).item()

        filename = os.path.basename(clean_path).rstrip('_hr.wav')
        pesq, stoi, enhanced_snr, lsd, sisnr, visqol = get_metrics(clean, enhanced, args.experiment.sample_rate, filename)


        df.loc[i] = [filename, noisy_snr, enhanced_snr, pesq, stoi, lsd, sisnr, visqol]
        add_data_to_wandb_table((clean, noisy, enhanced), (pesq, stoi, lsd, sisnr, visqol), filename, args, wandb_table)

    wandb.log({"Results": wandb_table})
    return df


def get_histogram_intervals(max_snr_value, n_bins):
    step_size = DEFAULT_HISTOGRAM_MAX_SNR / n_bins
    max_snr_value = max(max_snr_value, DEFAULT_HISTOGRAM_MAX_SNR)
    intervals = np.arange(max_snr_value + 1e-3, step=step_size)
    while intervals[-1] < max_snr_value:
        intervals = np.append(intervals, intervals[-1]+step_size)
    return intervals


def create_results_histogram_df(results_df, n_bins):
    results_histogram_df = pd.DataFrame(columns=[HISTOGRAM_DF_RANGE, HISTOGRAM_DF_N_SAMPLES, HISTOGRAM_DF_AVG_PESQ,
                                                 HISTOGRAM_DF_AVG_STOI, HISTOGRAM_DF_AVG_LSD, HISTOGRAM_DF_AVG_SISNR,
                                                 HISTOGRAM_DF_AVG_VISQOL])
    noisy_snr_values = results_df[RESULTS_DF_NOISY_SNR]
    bin_indices, bins = pd.cut(noisy_snr_values, get_histogram_intervals(noisy_snr_values.max(), n_bins),
                               labels=False, retbins=True, right=False)
    wandb_ranges = []
    n_samples_per_bin = []
    wandb_pesq = []
    wandb_stoi = []
    wandb_lsd = []
    wandb_sisnr = []
    wandb_visqol = []
    total_n_samples = 0
    for i in range(len(bins)-1):
        bin_range = (float("{:.2f}".format(bins[i])), float("{:.2f}".format(bins[i + 1])))
        wandb_ranges.append(', '.join(str(x) for x in bin_range))
        n_samples_per_bin_i = len(bin_indices[bin_indices == i])
        n_samples_per_bin.append(n_samples_per_bin_i)
        total_n_samples += n_samples_per_bin_i
        bin_avg_pesq = results_df.pesq[bin_indices == i].mean()
        bin_avg_stoi = results_df.stoi[bin_indices == i].mean()
        bin_avg_lsd = results_df.lsd[bin_indices == i].mean()
        bin_avg_sisnr = results_df.sisnr[bin_indices == i].mean()
        bin_avg_visqol = results_df.visqol[bin_indices == i].mean()
        bin_avg_pesq = 0 if math.isnan(bin_avg_pesq) else bin_avg_pesq
        bin_avg_stoi = 0 if math.isnan(bin_avg_stoi) else bin_avg_stoi
        bin_avg_lsd = 0 if math.isnan(bin_avg_lsd) else bin_avg_lsd
        bin_avg_sisnr = 0 if math.isnan(bin_avg_sisnr) else bin_avg_sisnr
        bin_avg_visqol = 0 if math.isnan(bin_avg_visqol) else bin_avg_visqol
        wandb_pesq.append(bin_avg_pesq)
        wandb_stoi.append(bin_avg_stoi)
        wandb_lsd.append(bin_avg_lsd)
        wandb_sisnr.append(bin_avg_sisnr)
        wandb_visqol.append(bin_avg_visqol)
        results_histogram_df.loc[i] = [bin_range, n_samples_per_bin_i, bin_avg_pesq, bin_avg_stoi, bin_avg_lsd,
                                       bin_avg_sisnr, bin_avg_visqol]
    log_wandb_bar_chart([[wandb_range, pesq, n_samples_per_bin_i] for (wandb_range, pesq, n_samples_per_bin_i)
                                                                in zip(wandb_ranges, wandb_pesq, n_samples_per_bin)],
                        ['ranges', 'pesq', 'n_samples_per_bin_i'], 'pesq', 'Average PESQ per SNR range')
    log_wandb_bar_chart([[wandb_range, stoi, n_samples_per_bin_i] for (wandb_range, stoi, n_samples_per_bin_i)
                                                                in zip(wandb_ranges, wandb_stoi, n_samples_per_bin)],
                        ['ranges', 'stoi', 'n_samples_per_bin_i'], 'stoi', 'Average STOI per SNR range')
    log_wandb_bar_chart([[wandb_range, lsd, n_samples_per_bin_i] for (wandb_range, lsd, n_samples_per_bin_i)
                         in zip(wandb_ranges, wandb_lsd, n_samples_per_bin)],
                        ['ranges', 'lsd', 'n_samples_per_bin_i'], 'lsd', 'Average LSD per SNR range')
    log_wandb_bar_chart([[wandb_range, sisnr, n_samples_per_bin_i] for (wandb_range, sisnr, n_samples_per_bin_i)
                         in zip(wandb_ranges, wandb_sisnr, n_samples_per_bin)],
                        ['ranges', 'sisnr', 'n_samples_per_bin_i'], 'sisnr', 'Average SISNR per SNR range')
    log_wandb_bar_chart([[wandb_range, visqol, n_samples_per_bin_i] for (wandb_range, visqol, n_samples_per_bin_i)
                         in zip(wandb_ranges, wandb_visqol, n_samples_per_bin)],
                        ['ranges', 'visqol', 'n_samples_per_bin_i'], 'visqol', 'Average VISQOL per SNR range')
    return results_histogram_df


def log_wandb_bar_chart(data, column_names, table_name, title):
    table = wandb.Table(data=data, columns=column_names)
    fields = {"label": column_names[0], "value": column_names[1]}
    custom_chart = wandb.plot_table(WANDB_CUSTOM_CHART_NAME, table, fields, {"title": title})
    wandb.log({table_name: custom_chart})

def log_results(args):
    logger.info('logging results...')
    results_out_path = 'results.csv'
    if os.path.isfile(results_out_path):
        logger.info('results.csv file already exists.')
        results_df = pd.read_csv(results_out_path, index_col=False)
    else:
        results_df = create_results_df(args)
        results_df.to_csv(results_out_path)

    n_bins = args.n_bins
    histogram_out_path = 'results_histogram_' + str(n_bins) + '.csv'
    if not os.path.isfile(histogram_out_path):
        results_histogram_df = create_results_histogram_df(results_df, n_bins)
        results_histogram_df.to_csv(histogram_out_path)
    else:
        logger.info('histogram file already exists.')


def init_wandb_table():
    columns = ['filename', 'clean audio', 'clean spectogram', 'noisy audio', 'noisy spectogram', 'enhanced audio',
               'enhanced spectogram', 'noisy snr', 'enhanced snr', 'pesq', 'stoi', 'lsd', 'sisnr', 'visqol']
    table = wandb.Table(columns=columns)
    return table


def add_data_to_wandb_table(signals, metrics, filename, args, wandb_table):
    clean, noisy, enhanced = signals

    spectrogram_transform = Spectrogram()

    clean_spectrogram = spectrogram_transform(clean).log2()[0, :, :].numpy()
    noisy_spectrogram = (SPECTOGRAM_EPSILON + spectrogram_transform(noisy)).log2()[0, :, :].numpy()
    enhanced_spectrogram = spectrogram_transform(enhanced).log2()[0, :, :].numpy()
    clean_wandb_spec = wandb.Image(convert_spectrogram_to_heatmap(clean_spectrogram))
    noisy_wandb_spec = wandb.Image(convert_spectrogram_to_heatmap(noisy_spectrogram))
    enhaced_wandb_spec = wandb.Image(convert_spectrogram_to_heatmap(enhanced_spectrogram))
    pesq, stoi, lsd, sisnr, visqol = metrics
    noisy_snr = get_snr(noisy, clean).item()
    enhanced_snr = get_snr(enhanced, clean).item()

    clean_sr = args.experiment.sample_rate

    clean_wandb_audio = wandb.Audio(clean.squeeze().numpy(), sample_rate=clean_sr, caption=filename + '_hr')
    noisy_wandb_audio = wandb.Audio(noisy.squeeze().numpy(), sample_rate=clean_sr, caption=filename + '_lr')
    enhanced_wandb_audio = wandb.Audio(enhanced.squeeze().numpy(), sample_rate=clean_sr, caption=filename + '_pr')

    wandb_table.add_data(filename, clean_wandb_audio, clean_wandb_spec, noisy_wandb_audio, noisy_wandb_spec,
                         enhanced_wandb_audio, enhaced_wandb_spec,
                         noisy_snr, enhanced_snr, pesq, stoi, lsd, sisnr, visqol)
