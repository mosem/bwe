import json
import math
import os
from torch.utils.data import DataLoader
import argparse
from torchaudio.transforms import Resample
import numpy as np
from denoiser.evaluate import get_pesq, get_stoi
from concurrent.futures import ThreadPoolExecutor
import torchaudio
from torch.nn import functional as F
from denoiser.resample import downsample2

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_json_dir")
parser.add_argument("--src_sr", default=8000, required=False)
parser.add_argument("--target_sr", default=16000, required=False)
args = parser.parse_args()

def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    noisy.sort()
    clean.sort()


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None, mel_config=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.stride = stride or length
        self.length = length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.use_mel = mel_config.use_melspec if mel_config is not None else False
        if self.use_mel:
            self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=mel_config.sample_rate,
                                                                 n_fft=mel_config.n_fft,
                                                                 n_mels=mel_config.n_mels,
                                                                 hop_length=mel_config.hop_length)

        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride))
            else:
                examples = (file_length - self.length) // self.stride
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length

            if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                out, sr = torchaudio.load(str(file),
                                          frame_offset=offset,
                                          num_frames=num_frames or -1)
            else:
                out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            if self.sample_rate is not None:
                if sr != self.sample_rate:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{self.sample_rate}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.use_mel:
                if len(out.shape) == 3:
                    out = out.squeeze(1)
                out = self.mel_spec(out)[..., :-1]
            if self.with_path:
                return out, file
            else:
                return out


class NoisyCleanSet:
    def __init__(self, json_dir, calc_valid_length_func, matching="sort", clean_length=None, stride=None,
                 pad=True, sample_rate=None, scale_factor=1, with_path=False, is_training=False, mel_config=None):
        """__init__.
        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param clean_length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        self.scale_factor = scale_factor
        self.with_path = with_path
        self.clean_length = clean_length
        self.calc_valid_length_func = calc_valid_length_func
        self.is_training = is_training

        if self.is_training:
            input_training_length = math.ceil(self.clean_length / self.scale_factor)
            self.valid_length = self.calc_valid_length_func(input_training_length)
        else:
            self.valid_length = None

        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)

        match_files(noisy, clean, matching)
        kw = {'length': self.valid_length, 'stride': stride, 'pad': pad, 'with_path': with_path}
        self.clean_set = Audioset(clean, sample_rate=sample_rate, **kw, mel_config=mel_config)
        self.noisy_set = Audioset(noisy, sample_rate=sample_rate, **kw, mel_config=mel_config)

        assert len(self.clean_set) == len(self.noisy_set)

    def _process_data(self, noisy, clean):

        if self.scale_factor == 2:
            noisy = downsample2(noisy)
        elif self.scale_factor == 4:
            noisy = downsample2(noisy)
            noisy = downsample2(noisy)
        elif self.scale_factor != 1:
            raise RuntimeError(f"Scale factor should be 1, 2, or 4")

        return noisy, clean

    def _get_item_with_path(self, index):
        (noisy, noisy_path), (clean, clean_path) = self.noisy_set[index], self.clean_set[index]
        noisy, clean = self._process_data(noisy, clean)
        return (noisy, noisy_path), (clean, clean_path)

    def _get_item_without_path(self, index):
        noisy, clean = self.noisy_set[index], self.clean_set[index]
        noisy, clean = self._process_data(noisy, clean)
        return noisy, clean

    def __getitem__(self, index):
        if self.with_path:
            return self._get_item_with_path(index)
        else:
            return self._get_item_without_path(index)

    def __len__(self):
        return len(self.noisy_set)


class SincIntrpolationBaseline:

    def __init__(self, metrics, src_sample_rate: int=8000, target_sample_rate: int=16000):
        self.metrics = metrics
        self.upsample = Resample(src_sample_rate, target_sample_rate)
        self.sr = target_sample_rate

    def evaluate_single_batch(self, batch):
        nb, wb = batch

        metrics_out = dict()

        # upsample
        upsampled_nb = self.upsample(nb)

        # trim to fix lengths
        if upsampled_nb.shape[-1] < wb.shape[-1]:
            wb = wb[..., :upsampled_nb.shape[-1]]
        elif upsampled_nb.shape[-1] > wb.shape[-1]:
            upsampled_nb = wb[..., :wb.shape[-1]]

        # collect all metrics for batch
        for metric, f in self.metrics.items():
            metrics_out[metric] = f(wb, upsampled_nb, self.sr)

        return metrics_out

    def evaluate(self, dataloader: DataLoader):
        """
        :param metrics: dictionary{metric_name: metric_function} where metric_function has args: (ref_sig, out_sig, sr)

        This method assumes each batch consists of x_nb, x_wb then it upsamples x_nb by sinc interpolation and evaluates
        all of the given metrics over the diff between x_nb_upsampled and x_wb
        """
        # initialize metrics
        metrics_out = {key: list() for key in self.metrics.keys()}

        # loop over data-loader and collect metrics
        with ThreadPoolExecutor() as executor:
            for tmp_metrics in executor.map(self.evaluate_single_batch, dataloader):
                for key, val in tmp_metrics.items():
                    metrics_out[key].append(val)

        # take mean of each metric
        return {key: np.mean(value.numpy()) for key, value in metrics_out.items()}


def run_pesq_stoi_eval_from_json(path_to_json_dir, src_sr=8000, target_sr=16000):
    metrics = {'pesq': get_pesq, 'stoi': get_stoi}
    baseline = SincIntrpolationBaseline(src_sr, target_sr)
    data_loader = DataLoader(NoisyCleanSet(path_to_json_dir, None, scale_factor=target_sr//src_sr),
                                              batch_size=1)
    return baseline.evaluate(data_loader, metrics)


if __name__ == "__main__":
    metrics = run_pesq_stoi_eval_from_json(args.path_to_json_dir, args.src_sr, args.target_sr)
    print("Evaluation Results for Sinc-Interpolation Baseline")
    for k, v in metrics.items():
        print(f"{k}:\t{v}")