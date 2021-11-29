from abc import ABC, abstractmethod

import torch
import torchaudio
import torch.nn.functional as F
from denoiser.utils import serialize_model, copy_state, load_lexical_model
from denoiser.models.dataclasses import FeaturesConfig


class BatchSolver(ABC):
    def __init__(self, args, features_config: FeaturesConfig=None):
        self.args = args
        self._models = {}
        self._optimizers = {}
        self._losses_names = []
        self.features_config = features_config
        self.include_ft = features_config.include_ft if features_config is not None else False
        if self.include_ft:
            self.ft_model = load_lexical_model(features_config.feature_model,
                                               features_config.state_dict_path,
                                               args.device)
            self.ft_factor = features_config.features_factor

    def train(self):
        for model in self.get_models().values():
            model.train()

    def eval(self):
        for model in self.get_models().values():
            model.eval()

    def serialize(self):
        serialized_models = {}
        serialized_optimizers = {}
        for name, model in self.get_models().items():
            serialized_models[name] = serialize_model(model)
        for name, optimizer in self.get_optimizers().items():
            serialized_optimizers[name] = optimizer.state_dict()
        return serialized_models, serialized_optimizers

    def load(self, package, load_best=False):
        if load_best:
            for name, model_package in package['best_states']['models'].items():
                self.get_models()[name].load_state_dict(model_package['state'])
        else:
            for name, model_package in package['models'].items():
                self.get_models()[name].load_state_dict(model_package['state'])
            for name, opt_package in package['optimizers'].items():
                self.get_optimizers()[name].load_state_dict(opt_package)

    def copy_models_states(self):
        states = {}
        for name, model in self.get_models().items():
            states[name] = copy_state(model.state_dict())
        return states

    def get_models(self) -> dict:
        return self._models

    def get_optimizers(self) -> dict:
        return self._optimizers

    def get_losses_names(self) -> list:
        return self._losses_names

    @abstractmethod
    def estimate_valid_length(self, input_length):
        """
        estimates the input length that will run smoothly through full pipeline.
        """
        pass

    @abstractmethod
    def run(self, data, cross_valid=False):
        """
        run on single batch
        """
        pass

    @abstractmethod
    def get_evaluation_loss(self, losses_dict):
        pass

    @abstractmethod
    def get_generator_for_evaluation(self, best_states):
        """
        loads the best state dict seen so far and returns a generator model ready for evaluation
        """
        pass

    def get_features_loss(self, estimated_embedded_dim, signal_to_extract_features_from):
        if not self.include_ft:
            return 0
        with torch.no_grad():
            estimated_embedded_dim.permute(0, 2, 1)
            y_ft = self.ft_model.extract_feats(signal_to_extract_features_from)
            estimated_embedded_dim = (torchaudio.transforms.Resample(estimated_embedded_dim.shape[-1], y_ft.shape[-1]).to(self.args.device)(estimated_embedded_dim)).permute(0, 2, 1)
            if estimated_embedded_dim.shape[-2] != y_ft.shape[-2]:
                estimated_embedded_dim = torchaudio.transforms.Resample(estimated_embedded_dim.shape[-2], y_ft.shape[-2]).to(self.args.device)(estimated_embedded_dim.permute(0, 2, 1)).permute(0, 2, 1)
            return F.l1_loss(y_ft, estimated_embedded_dim) * self.ft_factor