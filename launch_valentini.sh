#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dset=valentini_dummy \
  experiment.experiment_name=testing_sisnr \
  experiment=initial \
  experiment.scale_factor=2 \
  epochs=1 \


