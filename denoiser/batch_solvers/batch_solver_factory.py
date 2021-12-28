from denoiser.batch_solvers.generator_bs import GeneratorBS
from denoiser.models.generator import Generator


class BatchSolverFactory:

    @staticmethod
    def get_bs(args):
        if args.experiment.model == "dummy":
            generator = Generator()
            return GeneratorBS(args, generator)
        else:
            raise ValueError("Given model name is not supported")

