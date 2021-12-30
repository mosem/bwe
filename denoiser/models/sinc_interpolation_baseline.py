import torch
import argparse
from torchaudio.transforms import Resample
from denoiser.data import NoisyCleanSet
from denoiser.evaluate import get_pesq, get_stoi
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_json_dir")
parser.add_argument("--src_sr", default=8000, required=False)
parser.add_argument("--target_sr", default=16000, required=False)
args = parser.parse_args()


class SincIntrpolationBaseline:

    def __init__(self, metrics, src_sample_rate: int=8000, target_sample_rate: int=16000):
        self.metrics = metrics
        self.upsample = Resample(src_sample_rate, target_sample_rate)
        self.sr = target_sample_rate

    def evaluate_single_batch(self, batch):
        nb, wb = batch

        metrics_out = dict()

        # upsample
        upsampled_nb = self.upsample(nb)

        # trim to fix lengths
        if upsampled_nb.shape[-1] < wb.shape[-1]:
            wb = wb[..., :upsampled_nb.shape[-1]]
        elif upsampled_nb.shape[-1] > wb.shape[-1]:
            upsampled_nb = wb[..., :wb.shape[-1]]

        # collect all metrics for batch
        for metric, f in self.metrics.items():
            metrics_out[metric] = f(wb, upsampled_nb, self.sr)

        return metrics_out

    def evaluate(self, dataloader: torch.utils.data.DataLoader):
        """
        :param metrics: dictionary{metric_name: metric_function} where metric_function has args: (ref_sig, out_sig, sr)

        This method assumes each batch consists of x_nb, x_wb then it upsamples x_nb by sinc interpolation and evaluates
        all of the given metrics over the diff between x_nb_upsampled and x_wb
        """
        # initialize metrics
        metrics_out = {key: list() for key in self.metrics.keys()}

        # loop over data-loader and collect metrics
        with ThreadPoolExecutor() as executor:
            for tmp_metrics in executor.map(self.evaluate_single_batch, dataloader):
                for key, val in tmp_metrics.items():
                    metrics_out[key].append(val)

        # take mean of each metric
        return {key: torch.mean(value) for key, value in metrics_out.items()}


def run_pesq_stoi_eval_from_json(path_to_json_dir, src_sr=8000, target_sr=16000):
    metrics = {'pesq': get_pesq, 'stoi': get_stoi}
    baseline = SincIntrpolationBaseline(src_sr, target_sr)
    data_loader = torch.utils.data.DataLoader(NoisyCleanSet(path_to_json_dir, None, scale_factor=target_sr//src_sr),
                                              batch_size=1)
    return baseline.evaluate(data_loader, metrics)


if __name__ == "__main__":
    metrics = run_pesq_stoi_eval_from_json(args.path_to_json_dir, args.src_sr, args.target_sr)
    print("Evaluation Results for Sinc-Interpolation Baseline")
    for k, v in metrics.items():
        print(f"{k}:\t{v}")