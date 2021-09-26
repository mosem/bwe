# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

from collections import namedtuple
import json
from pathlib import Path
import math
import os
import sys

import torchaudio
from torch.nn import functional as F
from scipy.io.wavfile import write

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None, source_sample_rate=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        # if source_sample_rate is not None and sample_rate is not None and source_sample_rate != sample_rate:
        #     length = int(length * source_sample_rate/sample_rate)
        self.stride = stride or length
        # # force even sample length over target rate samples
        # self.length = length if length is None or length % 2 == 0 or \
        #                         (source_sample_rate is not None and source_sample_rate != sample_rate) \
        #     else length - 1
        self.length = length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.ssr = source_sample_rate

        # (or) added resampling support
        # if source_sample_rate is not None and source_sample_rate != sample_rate:
        #     self.resample = torchaudio.transforms.Resample(sample_rate, source_sample_rate)
        # else:
        #     self.resample = None

        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride))
                # examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride
                # examples = (file_length - self.length) // self.stride + 1
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

            if num_frames or (self.length is not None and out.shape[-1] < self.length):
                out = F.pad(out, (0, self.length - out.shape[-1]))
            if self.with_path:
                return out, file
            else:
                return out


if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_audio_files(path)
    json.dump(meta, sys.stdout, indent=4)