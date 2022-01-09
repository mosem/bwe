from denoiser.batch_solvers.generator_bs import GeneratorBS
from denoiser.batch_solvers.adversarial_bs import AdversarialBS
from denoiser.models.generator import Generator
from denoiser.models.modules import Discriminator
from denoiser.models.seanet import Seanet
from denoiser.models.sinc import Sinc


class BatchSolverFactory:

    @staticmethod
    def get_bs(args):
        if args.experiment.model == "dummy":
            generator = Generator(**args.experiment.generator)
        elif args.experiment.model == "seanet":
            generator = Seanet(**args.experiment.seanet)
        else:
            raise ValueError("Given model name is not supported")

        if 'adversarial' in args.experiment and args.experiment.adversarial:
            discriminator = Discriminator(**args.experiment.discriminator)
            return AdversarialBS(args, generator, discriminator)
        else:
            return GeneratorBS(args, generator)

