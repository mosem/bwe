#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dset=valentini \
  experiment.experiment_name=relative_low_pass_module_pow \
  experiment=initial \
  experiment.scale_factor=2 \

