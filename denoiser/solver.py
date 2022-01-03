# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import json
import logging
from pathlib import Path
import os
import time

import torch
import wandb

from . import distrib
from .augment import Augment
from .enhance import enhance
from .evaluate import evaluate
from .stft_loss import MultiResolutionSTFTLoss
from .utils import bold, pull_metric, LogProgress
from .log_results import log_results

logger = logging.getLogger(__name__)

SERIALIZE_KEY_MODELS = 'models'
SERIALIZE_KEY_OPTIMIZERS = 'optimizers'
SERIALIZE_KEY_HISTORY = 'history'
SERIALIZE_KEY_STATE = 'state'
SERIALIZE_KEY_BEST_STATES = 'best_states'
SERIALIZE_KEY_ARGS = 'args'

METRICS_KEY_EVALUATION_LOSS = 'evaluation_loss'
METRICS_KEY_BEST_LOSS = 'best_loss'
METRICS_KEY_PESQ = 'total pesq'
METRICS_KEY_STOI = 'total stoi'
METRICS_KEY_LSD = 'total lsd'
METRICS_KEY_SISNR = 'total sisnr'


class Solver(object):
    def __init__(self, data, batch_solver, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.batch_solver = batch_solver

        self.augment = Augment(args)

        # Training config
        self.device = args.device
        self.epochs = args.epochs

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_states = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.samples_dir = args.samples_dir  # Where to save samples
        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.args = args
        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)
        self._reset()

    def _serialize(self):
        package = {}
        package[SERIALIZE_KEY_MODELS], package[SERIALIZE_KEY_OPTIMIZERS] = self.batch_solver.serialize()
        package[SERIALIZE_KEY_HISTORY] = self.history
        package[SERIALIZE_KEY_BEST_STATES] = self.best_states
        package[SERIALIZE_KEY_ARGS] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        models = package[SERIALIZE_KEY_MODELS]
        for model_name, best_state in package[SERIALIZE_KEY_BEST_STATES].items():
            models[model_name][SERIALIZE_KEY_STATE] = best_state
            model_filename = model_name + '_' + self.best_file.name
            tmp_path = os.path.join(self.best_file.parent, model_filename) + ".tmp"
            torch.save(models[model_name], tmp_path)
            model_path = Path(self.best_file.parent / model_filename)
            os.rename(tmp_path, model_path)

    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True
        # Reset
        if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = self.args.continue_best
            keep_history = self.args.keep_history

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            self.batch_solver.load(package, load_best)
            if keep_history:
                self.history = package[SERIALIZE_KEY_HISTORY]
            self.best_states = package[SERIALIZE_KEY_BEST_STATES]

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

        logger.info('-' * 70)
        logger.info("Trainable Params:")
        for name, model in self.batch_solver.get_models().items():
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mb = n_params * 4 / 2 ** 20
            logger.info(f"{name}: parameters: {n_params}, size: {mb} MB")

        if (self.epochs > len(self.history)):
            logger.info("Training...")

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.batch_solver.train()
            start = time.time()
            # added logging support for printing out model params

            losses = self._run_one_epoch(epoch)
            logger_msg = f'Train Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | ' \
                         + ' | '.join([f'{k} Loss {v:.5f}' for k,v in losses.items()])
            logger.info(bold(logger_msg))
            losses = {k + '_loss': v for k, v in losses.items()}
            logger.info('-' * 70)
            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.batch_solver.eval()
                with torch.no_grad():
                    valid_losses = self._run_one_epoch(epoch, cross_valid=True)
                evaluation_loss = self.batch_solver.get_evaluation_loss(valid_losses)
                logger_msg = f'Validation Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | ' \
                             + ' | '.join([f'{k} Valid Loss {v:.5f}' for k, v in valid_losses.items()])
                logger.info(bold(logger_msg))
                valid_losses = {'valid_'  + k + '_loss': v for k,v in valid_losses.items()}
            else:
                valid_losses = {}
                evaluation_loss = 0

            best_loss = min(pull_metric(self.history, METRICS_KEY_EVALUATION_LOSS) + [evaluation_loss])
            metrics = {**losses, **valid_losses, METRICS_KEY_EVALUATION_LOSS: evaluation_loss, METRICS_KEY_BEST_LOSS: best_loss}
            # Save the best model
            if evaluation_loss == best_loss:
                logger.info(bold('New best evaluation loss %.4f'), evaluation_loss)
                self.best_states = self.batch_solver.copy_models_states()

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')

                generator = self.batch_solver.get_generator_for_evaluation(self.best_states)
                with torch.no_grad():
                    pesq, stoi, lsd, sisnr = evaluate(self.args, generator, self.tt_loader, epoch)

                    metrics.update({METRICS_KEY_PESQ: pesq, METRICS_KEY_STOI: stoi, METRICS_KEY_LSD: lsd,
                                    METRICS_KEY_SISNR: sisnr})

                # enhance some samples
                logger.info('Enhance and save samples...')
                enhance(self.args, generator, self.samples_dir, self.tt_loader)

            wandb.log(metrics, step=epoch)
            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize()
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
        if self.args.log_results:
            log_results(self.args)

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_losses = {k:0 for k in self.batch_solver.get_losses_names()}
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)

        for i, (noisy, clean) in enumerate(logprog):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            if not cross_valid:
                noisy, clean = self.augment.augment_data(noisy, clean)

            losses = self.batch_solver.run((noisy, clean), cross_valid, epoch)
            for k in self.batch_solver.get_losses_names():
                total_losses[k] += losses[k]
            losses_info = {k: format(v/(i+1), ".5f") for k,v in total_losses.items()}
            logprog.update(**losses_info)
            del losses

        for k,v in total_losses.items():
            total_losses[k] = v/(i+1)

        return total_losses

