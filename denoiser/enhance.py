# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import math
from concurrent.futures import ProcessPoolExecutor
import logging
import os
import json

import torch
import torchaudio
from . import distrib

from .utils import LogProgress

logger = logging.getLogger(__name__)


def get_estimate(model, noisy_sigs):
    torch.set_num_threads(1)
    with torch.no_grad():
        estimate = model(noisy_sigs)
    return estimate


def trim(noisy, clean, enhanced, raw_lengths_pair):
     noisy_raw_length, clean_raw_length = raw_lengths_pair
     noisy = noisy[..., :noisy_raw_length]
     clean = clean[..., :clean_raw_length]
     enhanced = enhanced[..., :clean_raw_length]
     return noisy, clean, enhanced


def save_wavs(noisy_sigs, clean_sigs, enhanced_sigs, raw_lengths_pairs, filenames, source_sr=16_000, target_sr=16_000):
    # Write result
    for noisy, clean, enhanced, raw_lengths_pair, filename in zip(noisy_sigs, clean_sigs, enhanced_sigs,
                                                                  raw_lengths_pairs, filenames):
        noisy, clean, enhanced = trim(noisy, clean, enhanced, raw_lengths_pair)
        write(noisy, filename + "_lr.wav", sr=source_sr)
        write(clean, filename + "_hr.wav", sr=target_sr)
        write(enhanced, filename + "_pr.wav", sr=target_sr)


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr, bits_per_sample=16)


def estimate_and_save(model, noisy_sigs, clean_sigs, raw_lengths_pairs, filenames, source_sr=16_000, target_sr=16_000):
    estimate_sigs = get_estimate(model, noisy_sigs)
    save_wavs(noisy_sigs, clean_sigs, estimate_sigs, raw_lengths_pairs, filenames, source_sr=source_sr, target_sr=target_sr)


def get_raw_lengths_dicts(args):
    noisy_json_path = os.path.join(args.dset.test, 'noisy.json')
    clean_json_path = os.path.join(args.dset.test, 'clean.json')
    with open(noisy_json_path, 'r') as f:
        noisy_json = json.load(f)
    with open(clean_json_path, 'r') as f:
        clean_json = json.load(f)
    noisy_json_dict = {path: int(math.ceil(length / args.experiment.scale_factor)) for (path, length) in noisy_json}
    clean_json_dict = {path: length for (path, length) in clean_json}
    return noisy_json_dict, clean_json_dict


def get_raw_lengths_pairs(noisy_json_dict, clean_json_dict, noisy_file_paths, clean_file_paths):
    return [(noisy_json_dict[noisy_file_path], clean_json_dict[clean_file_path])
            for noisy_file_path, clean_file_path in zip(noisy_file_paths, clean_file_paths)]


def enhance(args, model, out_dir, data_loader):

    model.eval()

    if distrib.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    distrib.barrier()
    noisy_json_dict, clean_json_dict = get_raw_lengths_dicts(args)

    with ProcessPoolExecutor(args.num_workers) as pool:
        iterator = LogProgress(logger, data_loader, name="Generate enhanced files")
        pendings = []
        for data in iterator:
            # Get batch data, expected to be batch size of 1 - but can work with larger batches as well.
            (noisy_sigs, noisy_paths), (clean_sigs, clean_paths) = data
            noisy_sigs = noisy_sigs.to(args.device)
            clean_sigs = clean_sigs.to(args.device)

            basenames = [os.path.join(out_dir, os.path.basename(path).rsplit(".", 1)[0]) for path in noisy_paths]
            raw_lengths_pairs = get_raw_lengths_pairs(noisy_json_dict, clean_json_dict, noisy_paths, clean_paths)

            noisy_sr = math.ceil(args.experiment.sample_rate / args.experiment.scale_factor)
            if args.device == 'cpu' and args.num_workers > 1:
                pendings.append(
                    pool.submit(estimate_and_save, model,
                                noisy_sigs, clean_sigs,
                                raw_lengths_pairs,
                                basenames, noisy_sr,
                                args.experiment.sample_rate))
            else:
                # Forward
                estimate = get_estimate(model, noisy_sigs)

                save_wavs(noisy_sigs, clean_sigs, estimate, raw_lengths_pairs, basenames,
                          source_sr=noisy_sr, target_sr=args.experiment.sample_rate)

        if pendings:
            print('Waiting for pending jobs...')
            for pending in LogProgress(logger, pendings, updates=5, name="Generate enhanced files"):
                pending.result()