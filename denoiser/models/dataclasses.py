# from dataclasses import dataclass
#
# @dataclass
# class FeaturesConfig:
#     include_ft: bool = False
#     feature_model: str = 'hubert'
#     state_dict_path: str = '/cs/labs/adiyoss/shared/pretrained_weights/hubert/hubert_base_ls960.pt'
#     features_factor: float = 0.01
#     get_ft_after_lstm: bool = True
#
# @dataclass
# class DemucsConfig:
#     chin: int = 1
#     chout: int = 1
#     hidden: int = 48
#     max_hidden: int = 10000
#     causal: bool = True
#     floor: float = 1e-3
#     glu: bool = True
#     depth: int = 5
#     kernel_size: int = 8
#     stride: int = 2
#     normalize: bool = True
#     resample: int = 1
#     growth: int = 2
#     rescale: float = 0.1
#     scale_factor: int = 1
#     skips: bool = False
#
# @dataclass
# class DemucsEncoderConfig:
#     chin: int = 1
#     hidden: int = 48
#     max_hidden: int = 10000
#     causal: bool = True
#     glu: bool = True
#     depth: int = 5
#     kernel_size: int = 8
#     stride: int = 2
#     resample: int = 1
#     growth: int = 2
#     rescale: float = 0.1
#     scale_factor: int = 1
#     skips: bool = False
#
#
# @dataclass
# class DemucsDecoderConfig:
#     chout: int = 1
#     hidden: int = 48
#     max_hidden: int = 10000
#     glu: bool = True
#     depth: int = 5
#     kernel_size: int = 8
#     stride: int = 2
#     resample: int = 1
#     growth: int = 2
#     rescale: float = 0.1
#     scale_factor: int = 1
#
# @dataclass
# class MelSpecConfig:
#     use_melspec: bool = False
#     sample_rate: int = 16000
#     n_fft: int = 512
#     n_mels: int = 128
#     hop_length: int = 256
