import os

# limit the threads
os.environ["OMP_NUM_THREADS"] = str(15)

import json
from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import soundfile as sf
import torch
from data_loaders import SS_SemiOnlineDataModule, SS_SemiOnlineDataset
from pandas.core.frame import DataFrame
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import (LightningArgumentParser, LightningCLI)
from torch import Tensor
from models.utils.metrics import cal_metrics_functional
from torchmetrics.functional.audio import permutation_invariant_training as pit, pit_permutate, scale_invariant_signal_distortion_ratio as si_sdr

COMMON_AUDIO_METRICS = ['SDR', 'SI_SDR', 'SI_SNR', 'NB_PESQ', 'WB_PESQ']

#### code borrowed from https://github.com/Enny1991/beamformers ####
import numpy as np
from scipy.linalg import solve, eigh, LinAlgError
from scipy.signal import stft as _stft, istft as _istft
import os
import soundfile as sf

eps = 1e-15


def stft(x, frame_len=2048, frame_step=512):
    return _stft(x, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]


def istft(x, frame_len=2048, frame_step=512, input_len=None):
    _reconstructed = _istft(x, noverlap=(frame_len - frame_step))[1].astype('float32' if x.dtype == 'complex64' else 'float64')
    if input_len is None:
        return _reconstructed
    else:
        rec_len = len(_reconstructed)
        if input_len <= rec_len:
            return _reconstructed[:input_len]
        else:
            return np.append(_reconstructed, np.zeros((input_len - rec_len, ), dtype=x.dtype))


def MVDR(mixture, noise, target=None, frame_len=2048, frame_step=512, ref_mic=0):
    """
    ftp://ftp.esat.kuleuven.ac.be/stadius/spriet/reports/08-211.pdf
    Frequency domain Minimum Variance Distortionless Response (MVDR) beamformer
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param noise: nd_array (n_mics, time) of the noise recordings
    :param target: nd_array (n_mics, time) of the target recordings
    :param frame_len: int (self explanatory)
    :param frame_step: int (self explanatory)
    :return: the enhanced signal
    """
    # calculate stft
    mixture_stft = stft(mixture, frame_len=frame_len, frame_step=frame_step)

    # estimate steering vector for desired speaker (depending if target is available)
    if target is not None:
        target_stft = stft(target, frame_len=frame_len, frame_step=frame_step)
        h = estimate_steering_vector(target_stft=target_stft)
    else:
        noise_spec = stft(noise, frame_len=frame_len, frame_step=frame_step)
        h = estimate_steering_vector(mixture_stft=mixture_stft, noise_stft=noise_spec)

    # calculate weights
    w = mvdr_weights(mixture_stft, h)

    # apply weights
    sep_spec = apply_beamforming_weights(mixture_stft, w)

    # reconstruct wav
    recon = istft(sep_spec, frame_len=frame_len, frame_step=frame_step, input_len=None)

    return recon


def estimate_steering_vector(target_stft=None, mixture_stft=None, noise_stft=None):
    """
    Estimation of steering vector based on microphone recordings. The eigenvector technique used is described in
    Sarradj, E. (2010). A fast signal subspace approach for the determination of absolute levels from phased microphone
    array measurements. Journal of Sound and Vibration, 329(9), 1553-1569.
    The steering vector is represented by the leading eigenvector of the covariance matrix calculated for each
    frequency separately.
    :param target_stft: nd_array (channels, time, freq_bins)
    :param mixture_stft: nd_array (channels, time, freq_bins)
    :param noise_stft: nd_array (channels, time, freq_bins)
    :return: h: nd_array (freq_bins, ): steering vector
    """

    if target_stft is None:
        if mixture_stft is None or noise_stft is None:
            raise ValueError("If no target recordings are provided you need to provide both mixture recordings "
                             "and noise recordings")
        C, F, T = mixture_stft.shape  # (channels, freq_bins, time)
    else:
        C, F, T = target_stft.shape  # (channels, freq_bins, time)

    eigen_vec, eigen_val, h = [], [], []

    for f in range(F):  # Each frequency separately

        # covariance matrix
        if target_stft is None:
            # covariance matrix estimated by subtracting mixture and noise covariances
            _R0 = mixture_stft[:, f].dot(np.conj(mixture_stft[:, f].T))
            _R1 = noise_stft[:, f].dot(np.conj(noise_stft[:, f].T))
            _Rxx = _R0 - _R1
        else:
            # covariance matrix estimated directly from single speaker
            _Rxx = target_stft[:, f].dot(np.conj(target_stft[:, f].T))

        # eigendecomposition
        [_d, _v] = np.linalg.eig(_Rxx)

        # index of leading eigenvector
        idx = np.argsort(_d)[::-1][0]

        # collect leading eigenvector and eigenvalue
        eigen_val.append(_d[idx])
        eigen_vec.append(_v[:, idx])

    # rescale eigenvectors by eigenvalues for each frequency
    for vec, val in zip(eigen_vec, eigen_val):
        if val != 0.0:
            # the part is modified from the MVDR implementation https://github.com/Enny1991/beamformers
            # vec = vec * val / np.abs(val)
            vec = vec / vec[0]  # normalized to the first channel
            h.append(vec)
        else:
            h.append(np.ones_like(vec))

    # return steering vector
    return np.vstack(h)


def apply_beamforming_weights(signals, weights):
    """
    Fastest way to apply beamforming weights in frequency domain.
    :param signals: nd_array (freq_bins (a), n_mics (b))
    :param weights: nd_array (n_mics (b), freq_bins (a), time_frames (c))
    :return: nd_array (freq_bins (a), time_frames (c)): filtered stft
    """
    return np.einsum('ab,bac->ac', np.conj(weights), signals)


def mvdr_weights(mixture_stft, h):
    C, F, T = mixture_stft.shape  # (channels, freq_bins, time)

    # covariance matrix

    R_y = np.einsum('a...c,b...c', mixture_stft, np.conj(mixture_stft)) / T
    R_y = condition_covariance(R_y, 1e-6)
    R_y /= np.trace(R_y, axis1=-2, axis2=-1)[..., None, None] + 1e-15
    # preallocate weights
    W = np.zeros((F, C), dtype='complex64')

    # compute weights for each frequency separately
    for i, r, _h in zip(range(F), R_y, h):
        # part = np.linalg.inv(r + np.eye(C, dtype='complex') * eps).dot(_h)
        part = solve(r, _h)
        _w = part / np.conj(_h).T.dot(part)

        W[i, :] = _w

    return W


def condition_covariance(x, gamma):
    """Code borrowed from https://github.com/fgnt/nn-gev/blob/master/fgnt/beamforming.py
    Please refer to the repo and to the paper (https://ieeexplore.ieee.org/document/7471664) for more information.
    see https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3)"""
    scale = gamma * np.trace(x, axis1=-2, axis2=-1)[..., None, None] / x.shape[-1]
    n = len(x.shape) - 2
    scaled_eye = np.eye(x.shape[-1], dtype=x.dtype)[(None, ) * n] * scale
    return (x + scaled_eye) / (1 + gamma)


#### https://github.com/Enny1991/beamformers end ####


# oracle beamformer
class OracleBeamformer(pl.LightningModule):

    def __init__(self, speaker_num: int = 2, ref_channel: int = 0, give_target: bool = True, exp_name: str = 'exp'):
        super().__init__()

        # save all the hyperparameters to the checkpoint
        self.save_hyperparameters()
        # self.ref_chn_idx = self.hparams.channels.index(self.hparams.ref_channel)

        print(f"using MVDR beamformer")

        self.give_target = give_target
        self.ref_channel = ref_channel

        import warnings
        warnings.filterwarnings("ignore")

    def forward(self, x: Tensor, ys: Tensor) -> Tuple[Tensor, DataFrame]:  # type: ignore
        # x: shape [batch, channel, time]
        # ys: shape [batch, speaker, channel, time]
        assert x.shape[0] == 1, "only accept one sample per batch"
        x = x[0, ...]
        ys = ys[0, ...]

        predictions = []
        for spk in range(ys.shape[0]):
            target = ys[spk, ...]
            noise = x - target
            if self.give_target:
                pred = MVDR(mixture=x.numpy(), noise=noise.numpy(), target=target.numpy(), ref_mic=self.ref_channel)
            else:
                pred = MVDR(mixture=x.numpy(), noise=noise.numpy())
            predictions.append(torch.tensor(pred))

        preds = torch.stack(predictions)[None, :, :x.shape[1]]
        return preds, predictions  # [1, speaker, channel, time]

    def collate_func_train(self, batches):
        return SS_SemiOnlineDataset.collate_fn(batches)

    def training_step(self, batch, batch_idx):
        raise RuntimeError('you could not train an oracle beamformer')

    def collate_func_val(self, batches):
        return SS_SemiOnlineDataset.collate_fn(batches)

    def validation_step(self, batch, batch_idx):
        raise RuntimeError('you could not validate an oracle beamformer')

    def on_test_epoch_start(self):
        os.makedirs(self.logger.log_dir, exist_ok=True)
        self.df = pd.DataFrame([], columns=['id', 'wavname'])
        self.expt_num = 0

    def test_epoch_end(self, results):
        import torch.distributed as dist

        if self.trainer.world_size > 1:
            results_list = [None for obj in results]
            dist.all_gather_object(results_list, results)  # gather results from all gpus
            # merge them
            exist = set()
            results = []
            for rs in results_list:
                if rs == None:
                    continue
                for r in rs:
                    if r['wavname'] not in exist:
                        results.append(r)
                        exist.add(r['wavname'])

        if self.trainer.is_global_zero:
            # Tensor to list or number
            for r in results:
                for key, val in r.items():
                    if isinstance(val, Tensor):
                        if val.numel() == 1:
                            r[key] = val.item()
                        else:
                            r[key] = val.detach().cpu().numpy().tolist()

            import datetime
            x = datetime.datetime.now()
            dtstr = x.strftime('%Y%m%d_%H%M%S.%f')
            path = os.path.join(self.logger.log_dir, 'results_{}.json'.format(dtstr))
            # write results to json
            f = open(path, 'w', encoding='utf-8')
            json.dump(results, f, indent=4, cls=NumpyEncoder)
            f.close()
            # write mean to json
            df = DataFrame(results)
            df['except_num'] = self.expt_num
            df.mean().to_json(os.path.join(self.logger.log_dir, 'results_mean.json'), indent=4)
        print('except num ' + str(self.expt_num))

    def collate_func_test(self, batches):
        return SS_SemiOnlineDataset.collate_fn(batches)

    def test_step(self, batch, batch_idx):
        x, ys, paras = batch

        ys_hat, predictions = self(x, ys)
        ys = ys[:, :, self.ref_channel, :]

        # 需要写入的统计信息
        wavname = os.path.basename(f"{paras[0]['index']}.wav")
        result_dict = {'id': batch_idx, 'wavname': wavname}

        try:
            # 计算metrics
            x_ref = x[0, self.ref_channel, :]
            _, ps = pit(target=ys, preds=ys_hat, metric_func=si_sdr, eval_func="max")
            ys_hat_perm = pit_permutate(ys_hat, ps)
            metrics, input_metrics, imp_metrics = cal_metrics_functional(COMMON_AUDIO_METRICS, ys_hat_perm[0], ys[0], x_ref.expand_as(ys[0]), 16000)

            for key, val in imp_metrics.items():
                self.log('test/' + key, val)
                result_dict[key] = val
            result_dict.update(metrics)
            result_dict.update(input_metrics)
        except:
            self.expt_num = self.expt_num + 1

        # 加入统计信息
        self.df = self.df.append(result_dict, ignore_index=True)

        # 写入预测的例子
        if paras[0]['index'] < 200 and self.local_rank == 0:
            abs_max = max(torch.max(torch.abs(x[0, ])), torch.max(torch.abs(ys_hat_perm[0, ])), torch.max(torch.abs(ys[0, ])))

            def write_wav(wav_path: str, wav: torch.Tensor):
                # make sure wav don't have illegal values (abs greater than 1)
                if abs_max > 1:
                    wav /= abs_max
                # write
                sf.write(wav_path, wav.detach().cpu().numpy(), 16000)

            example_dir = os.path.join(self.logger.log_dir, 'examples', str(paras[0]['index']))
            os.makedirs(example_dir, exist_ok=True)
            # copy files by creating hard links
            pattern = '.'.join(wavname.split('.')[:-1]) + '_{name}'
            for i in range(ys.shape[1]):
                # write ys
                wp = os.path.join(example_dir, pattern.format(name=f"spk{i+1}.wav"))
                write_wav(wp, ys[0, i])
                # write ys_hat
                wp = os.path.join(example_dir, pattern.format(name=f"spk{i+1}_p.wav"))
                write_wav(wp, ys_hat_perm[0, i])
            # write mix
            wp = os.path.join(example_dir, pattern.format(name=f"mix.wav"))
            write_wav(wp, x[0, :].T)  # to [time, channel]
            # write paras
            f = open(os.path.join(example_dir, pattern.format(name=f"_paras.json")), 'w', encoding='utf-8')
            paras[0]['metrics'] = result_dict
            json.dump(paras[0], f, indent=4, cls=NumpyEncoder)
            f.close()

        if 'metrics' in paras[0]:
            del paras[0]['metrics']  # remove circular reference
        result_dict['paras'] = paras[0]
        return result_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Tensor):
            return obj.detach().cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class BeamformerCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.set_defaults({
            "data.clean_speech_dataset": "wsj0-mix",
            "data.clean_speech_dir": "/dev/shm/quancs/",
            "data.rir_dir": "/dev/shm/quancs/rir_cfg_3",
            # "trainer.benchmark": True,
        })

        # link args
        parser.link_arguments("data.speaker_num", "model.speaker_num", apply_on="parse")  # when parse config file
        parser.link_arguments("model.collate_func_train", "data.collate_func_train", apply_on="instantiate")  # after instantiate model
        parser.link_arguments("model.collate_func_val", "data.collate_func_val", apply_on="instantiate")  # after instantiate model
        parser.link_arguments("model.collate_func_test", "data.collate_func_test", apply_on="instantiate")  # after instantiate model

        return super().add_arguments_to_parser(parser)

    def before_fit(self):
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)

        model_name = str(self.model_class).split('\'')[1].split('.')[-1]
        self.trainer.logger = TensorBoardLogger('logs/', name=model_name, default_hp_metric=False)

    def before_test(self):
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)

        model_name = str(self.model_class).split('\'')[1].split('.')[-1]
        self.trainer.logger = TensorBoardLogger('logs/', name=model_name, default_hp_metric=False)

# command: python -m models.oracle_beamformer test --data.seeds="{'train': 1,'val': 2,'test': 3}" --data.clean_speech_dir=~/datasets/ --data.rir_dir=~/datasets/rir_cfg_4 --data.audio_time_len="['headtail 4','headtail 4','headtail 4']"

cli = BeamformerCLI(OracleBeamformer, SS_SemiOnlineDataModule, seed_everything_default=None)
