#!/bin/bash

python test.py \
  dset=valentini \
  experiment=seanet \
  continue_from=../seanet/checkpoint.th \
  hydra.run.dir=./outputs/seanet_evaluate \
