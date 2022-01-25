from pathlib import Path
import hydra
import logging
import torch
import wandb
import os

from denoiser.utils import bold
from denoiser.batch_solvers.batch_solver_factory import BatchSolverFactory
from denoiser import distrib
from denoiser.enhance import enhance
from denoiser.models.sinc import Sinc
from denoiser.log_results import log_results
from denoiser.evaluate import evaluate
from denoiser.data import NoisyCleanSet

from .train import _get_wandb_config

logger = logging.getLogger(__name__)

WANDB_PROJECT_NAME = 'Bandwidth Extension'
WANDB_ENTITY = 'huji-dl-audio-lab'

def get_generator(args):
    if args.experiment.model == "sinc":
        generator = Sinc(**args.experiment.sinc)
    else:
        batch_solver = BatchSolverFactory.get_bs(args)
        logger.info(f'current path: {Path().resolve()}')
        checkpoint_file = Path(args.continue_from)
        if checkpoint_file:
            logger.info(f'Loading checkpoint model: {checkpoint_file}')
            package = torch.load(checkpoint_file, 'cpu')
            batch_solver.load(package, args.continue_best)
            best_states = package['best_states']
            generator = batch_solver.get_generator_for_evaluation(best_states)
        else:
            raise ValueError('No checkpoint file found.')
    return generator


def _main(args):
    global __file__
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    wandb_mode = os.environ['WANDB_MODE'] if 'WANDB_MODE' in os.environ.keys() else args.wandb.mode
    wandb.init(mode=wandb_mode, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=_get_wandb_config(args),
               group=args.experiment.experiment_name, resume=(args.continue_from != ""))

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)

    logger.info(f'evaluating model {args.experiment.model}')
    generator = get_generator(args)

    if torch.cuda.is_available() and args.device == 'cuda':
        generator.cuda()
    tt_dataset = NoisyCleanSet(args.dset.test, generator.estimate_output_length,
                               scale_factor=args.experiment.scale_factor, with_path=True,
                               matching=args.dset.matching, sample_rate=args.experiment.sample_rate)
    tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
    epoch = 0
    with torch.no_grad():
        pesq, stoi, lsd, sisnr, visqol = evaluate(args, generator, tt_loader, epoch)
    enhance(args, generator, args.samples_dir, tt_loader)

    metrics = {'Average pesq': pesq, 'Average stoi': stoi, 'Average lsd': lsd,
               'Average sisnr': sisnr, 'Average visqol': visqol}
    wandb.log(metrics, step=epoch)
    info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
    logger.info('-' * 70)
    logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

    if args.log_results:
        log_results(args)
    return


@hydra.main(config_path="conf", config_name="main_config")  # for latest version of hydra=1.0
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()