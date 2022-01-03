# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez and adiyoss

import json
import logging
import math
import os
import re

from .audio import Audioset
from .models.dataclasses import MelSpecConfig
from .resample import downsample2
from torch.nn import functional as F
from torchaudio.transforms import Resample

logger = logging.getLogger(__name__)


def match_dns(noisy, clean):
    """match_dns.
    Match noisy and clean DNS dataset filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    """
    logger.debug("Matching noisy and clean for dns dataset")
    noisydict = {}
    extra_noisy = []
    for path, size in noisy:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            # maybe we are mixing some other dataset in
            extra_noisy.append((path, size))
        else:
            noisydict[match.group(1)] = (path, size)
    noisy[:] = []
    extra_clean = []
    copied = list(clean)
    clean[:] = []
    for path, size in copied:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            extra_clean.append((path, size))
        else:
            noisy.append(noisydict[match.group(1)])
            clean.append((path, size))
    extra_noisy.sort()
    extra_clean.sort()
    clean += extra_clean
    noisy += extra_noisy


def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")


def pad_signal_to_valid_length(signal, calc_valid_length_func, scale_factor):
    valid_length = calc_valid_length_func(math.ceil(signal.shape[-1] / scale_factor))

    if valid_length > signal.shape[-1]:
        signal = F.pad(signal, (0, valid_length - signal.shape[-1]))

    return signal


class NoisyCleanSet:
    def __init__(self, json_dir, calc_valid_length_func, matching="sort", clean_length=None, stride=None,
                 pad=True, sample_rate=None, scale_factor=1, with_path=False, is_training=False, mel_config:MelSpecConfig=None):
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

        self.resample = Resample(sample_rate, sample_rate / scale_factor)

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
        if not self.is_training:
            noisy = pad_signal_to_valid_length(noisy, self.calc_valid_length_func, self.scale_factor)
            clean = pad_signal_to_valid_length(clean, self.calc_valid_length_func, self.scale_factor)

        # if self.scale_factor == 2:
        #     noisy = downsample2(noisy)
        # elif self.scale_factor == 4:
        #     noisy = downsample2(noisy)
        #     noisy = downsample2(noisy)
        # elif self.scale_factor != 1:
        #     raise RuntimeError(f"Scale factor should be 1, 2, or 4")

        noisy = self.resample(noisy)

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