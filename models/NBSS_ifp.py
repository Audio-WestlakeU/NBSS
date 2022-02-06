from typing import Any, Dict, List
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch import Tensor
import os
import soundfile as sf
from utils import perm_by_correlation, permutation_analysis, perm_by_ps, decode
import numpy as np
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from data_loaders import SS_SemiOnlineDataset
from torchmetrics.functional.audio import permutation_invariant_training as pit, pit_permutate, scale_invariant_signal_distortion_ratio as si_sdr
from models.metrics import cal_metrics_functional  #, COMMON_AUDIO_METRICS
import json
from torch.utils.tensorboard import SummaryWriter

COMMON_AUDIO_METRICS = ['SDR', 'SI_SDR', 'SI_SNR', 'NB_PESQ', 'WB_PESQ']


def neg_si_sdr_batch(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = target.shape[0]
    si_snr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_snr_val.view(batch_size, -1), dim=1)


def mse_batch(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = target.shape[0]
    res = (target - preds)**2
    return torch.mean(res.view(batch_size, -1), dim=1)


# NBSS implicit frequency permutation
class NBSS_ifp(pl.LightningModule):
    K, C, LIM = 10.0, 0.1, 9.99  # parameters for CIRM

    def __init__(self,
                 activation: str = "",
                 target_type: str = "WavForm",
                 band_num: int = 1,
                 speaker_num: int = 2,
                 ref_channel: int = 0,
                 channels: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
                 ft_len: int = 512,
                 ft_overlap: int = 256,
                 learning_rate: float = 0.001,
                 hidden_size: List[int] = [256, 128],
                 layer_norm: bool = False,
                 optimizer_kwargs: Dict[str, Any] = dict(),
                 lr_scheduler: str = 'ReduceLROnPlateau',
                 lr_scheduler_kwargs: Dict[str, Any] = {
                     'mode': 'min',
                     'factor': 0.5,
                     'patience': 10,
                     'min_lr': 1e-4
                 },
                 exp_name: str = "exp"):
        super().__init__()

        assert band_num == 1, f'band_num > 1 is not fully tested: {band_num}!'
        assert target_type in ['MRM', 'CIRM', 'TFMapping', 'WavForm'], f'Unsupported target type: {target_type}!'

        if target_type in ['CIRM', 'TFMapping', 'WavForm'] and (activation is not None and len(activation) > 0):
            print(f'changed activation function from {activation} to None for {target_type}')
            activation = ""
        # save all the hyperparameters to the checkpoint
        self.save_hyperparameters()

        if target_type in ['MRM', 'CIRM', 'TFMapping']:
            self.loss_func = mse_batch
        else:
            self.loss_func = neg_si_sdr_batch

        if self.hparams.target_type in ['MRM']:  # type:ignore
            self.dims_per_speaker = 1
        elif self.hparams.target_type in ['TFMapping', 'CIRM', 'WavForm']:  # type:ignore
            self.dims_per_speaker = 2
        else:
            raise Exception('Unknown output type: ' + self.hparams.target_type + '. All the legal types are MRM, TFMapping, CIRM')  # type:ignore

        self.ref_chn_idx = self.hparams.channels.index(self.hparams.ref_channel)  # type:ignore

        self.input_size = 2 * len(self.hparams.channels) * self.hparams.band_num  # type:ignore
        self.output_size = self.dims_per_speaker * self.hparams.band_num * self.hparams.speaker_num  # type:ignore

        self.blstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hparams.hidden_size[0], batch_first=True, bidirectional=True)  # type:ignore
        self.blstm2 = nn.LSTM(input_size=self.hparams.hidden_size[0] * 2, hidden_size=self.hparams.hidden_size[1], batch_first=True, bidirectional=True)  # type:ignore
        self.linear = nn.Linear(self.hparams.hidden_size[1] * 2, self.output_size)  # type:ignore
        if self.hparams.activation is not None and len(self.hparams.activation) > 0:  # type:ignore
            self.activation_func = getattr(nn, self.hparams.activation)()  # type:ignore
        else:
            self.activation_func = None

        self.window_cpu = torch.hann_window(self.hparams.ft_len)  # type:ignore
        self.register_buffer('window', self.window_cpu, False)  # self.window, will be moved to self.device at training time

    def on_train_start(self):
        if self.current_epoch == 0:
            self.logger.log_hyperparams(self.hparams, {"val/neg_si_sdr": 0, "val/fploss": 1})

            if self.trainer.is_global_zero:
                # add git tags for better change tracking
                import subprocess
                gitout = open(self.logger.log_dir + '/git.out', 'a')
                del_tag = f'git tag -d ifp_v{self.logger.version}'
                add_tag = f'git tag -a ifp_v{self.logger.version} -m "{self.hparams.exp_name}"'
                print_branch = "git branch -vv"
                print_status = 'git status'
                cmds = [del_tag, add_tag, print_branch, print_status]
                for cmd in cmds:
                    o = subprocess.getoutput(cmd)
                    gitout.write(f'========={cmd}=========\n{o}\n\n\n')
                gitout.close()

    def forward(self, x):
        x, _ = self.blstm1(x)
        if self.hparams.layer_norm:
            x = nn.functional.layer_norm(x, x.shape[1:])
        x, _ = self.blstm2(x)
        if self.hparams.layer_norm:
            x = nn.functional.layer_norm(x, x.shape[1:])
        if self.activation_func is not None:
            y = self.activation_func(self.linear(x))
        else:
            y = self.linear(x)

        return y

    def run_one_step(self, X_bands, Ys_bands, Xr_bands, XrMM_bands, ys):
        # X_bands (batch, freq, time, band_num, chn_num, 2)
        # Ys_bands (batch, spk, freq, time, band_num) complex
        # Xr_bands (batch, freq, time, band_num) complex
        # XrMM_bands (batch, freq, band_num)
        # ys (batch, spk, time)
        batch_size, spk_num, freq_num, time, band_num = Ys_bands.shape
        chn_num = X_bands.shape[4]

        X_bands = X_bands.reshape(batch_size * freq_num, time, band_num * chn_num * 2)
        # outputs (batch * freq, time, spk * band_num * dims_per_spk)
        outputs = self(X_bands)

        # outputs to (batch, freq, time, spk, band_num, dims_per_spk)
        outputs = outputs.reshape(batch_size, freq_num, time, spk_num, band_num, self.dims_per_speaker)
        # (batch, freq, spk, time, band)
        if self.hparams.target_type == 'MRM':
            mrms = torch.empty(size=(batch_size, spk_num, freq_num, time, band_num), dtype=torch.float32, device=outputs.device)
            Xr_bands_abs = torch.abs(Xr_bands) + 1e-8
            Ys_bands_abs = torch.abs(Ys_bands)
            for spk in range(spk_num):
                mrms[:, spk, :, :, :] = Ys_bands_abs[:, spk, :, :, :] / Xr_bands_abs
            mrms = (mrms >= 1) + ((mrms < 1) * mrms)

            targets = mrms
            estimates = outputs.permute(0, 3, 1, 2, 4, 5).contiguous().squeeze(dim=5)
        elif self.hparams.target_type == 'CIRM':
            cirm = (Ys_bands + 1e-8 + 1j * 1e-8) / (Xr_bands.unsqueeze(dim=1) + 1e-8 + 1j * 1e-8)
            # print('cirm2', (cirm2 * Xr_bands.unsqueeze(dim=1) - Ys_bands).abs().max().item(), (cirm2 * Xr_bands.unsqueeze(dim=1) - Ys_bands).abs().mean().item())
            # cirm3 = (Ys_bands) / (Xr_bands.unsqueeze(dim=1) + 1e-8 + 1j * 1e-8)
            # print('cirm3', (cirm3 * Xr_bands.unsqueeze(dim=1) - Ys_bands).abs().max().item(), (cirm3 * Xr_bands.unsqueeze(dim=1) - Ys_bands).abs().mean().item())
            cirm_lim = -100.0 * (torch.view_as_real(cirm) <= -100) + torch.view_as_real(cirm) * (torch.view_as_real(cirm) > -100)  # to make sure e^(-C * cirm2_lim)!=inf
            cirm_cmp = self.K * (1.0 - torch.exp(-self.C * cirm_lim)) / (1.0 + torch.exp(-self.C * cirm_lim))

            # cirm_cmp = self.LIM * (cirm_cmp >= self.LIM) - self.LIM * (cirm_cmp <= -self.LIM) + cirm_cmp * (cirm_cmp.abs() < self.LIM)  # to (-10, 10) not [-10, 10]
            # cirm_uncmp = -1 / self.C * torch.log((self.K - cirm_cmp) / (self.K + cirm_cmp))
            # cirm = torch.view_as_complex(cirm_uncmp)
            # print('cirm', (cirm * Xr_bands.unsqueeze(dim=1) - Ys_bands).abs().max().item(), (cirm * Xr_bands.unsqueeze(dim=1) - Ys_bands).abs().mean().item())

            targets = torch.view_as_complex(cirm_cmp)
            estimates = outputs.permute(0, 3, 1, 2, 4, 5).contiguous().squeeze(dim=5)
            estimates = torch.view_as_complex(estimates)
        elif self.hparams.target_type == 'TFMapping':
            Ys_bands_hat = torch.empty(size=(batch_size, spk_num, freq_num, time, band_num), dtype=torch.complex64, device=outputs.device)
            outputs = torch.view_as_complex(outputs)
            XrMM_bands = torch.unsqueeze(XrMM_bands, dim=2).expand(-1, -1, time, -1)
            for spk in range(spk_num):
                Ys_bands_hat[:, spk, :, :, :] = outputs[:, :, :, spk, :] * XrMM_bands[:, :, :, :]
            targets = Ys_bands
            estimates = Ys_bands_hat
        elif self.hparams.target_type == 'WavForm':
            assert self.hparams.band_num == 1, "not implemented for band_num>1 case"

            Ys_bands_hat = torch.empty(size=(batch_size, spk_num, freq_num, time, band_num), dtype=torch.complex64, device=outputs.device)
            outputs = torch.view_as_complex(outputs)
            XrMM_bands = torch.unsqueeze(XrMM_bands, dim=2).expand(-1, -1, time, -1)
            for spk in range(spk_num):
                Ys_bands_hat[:, spk, :, :, :] = outputs[:, :, :, spk, :] * XrMM_bands[:, :, :, :]

            ys_hat = torch.istft(Ys_bands_hat.reshape(batch_size * spk_num, freq_num, time),
                                 n_fft=self.hparams.ft_len,
                                 hop_length=self.hparams.ft_overlap,
                                 window=self.window,
                                 win_length=self.hparams.ft_len,
                                 length=ys.shape[-1])
            ys_hat = ys_hat.reshape(batch_size, spk_num, ys_hat.shape[1])
            estimates = ys_hat
            targets = ys
        else:
            raise NotImplementedError('not implemented for target_type=={}'.format(self.hparams.target_type))

        # cal loss
        if estimates.dtype.is_complex:
            fplosses, ps_t = pit(preds=torch.view_as_real(estimates), target=torch.view_as_real(targets), metric_func=self.loss_func, eval_func='min')
        else:
            fplosses, ps_t = pit(preds=estimates, target=targets, metric_func=self.loss_func, eval_func='min')

        fploss = torch.mean(fplosses)
        bps_t = ps_t.view(batch_size, spk_num)

        return fploss, targets, estimates, outputs, bps_t

    def encode(self, x, ys):
        batch_size, chn_num, time = x.shape
        _, spk_num, _ = ys.shape
        # stft x
        x = x.reshape((batch_size * chn_num, time))
        X = torch.stft(x, n_fft=self.hparams.ft_len, hop_length=self.hparams.ft_overlap, window=self.window_cpu, win_length=self.hparams.ft_len, return_complex=True)
        X = X.reshape((batch_size, chn_num, X.shape[-2], X.shape[-1]))  # (batch, channel, freq, time)
        X = X.permute(0, 2, 3, 1)  # (batch, freq, time, channel)

        # stft ys
        ys = ys.reshape((batch_size * spk_num, -1))
        Ys = torch.stft(ys, n_fft=self.hparams.ft_len, hop_length=self.hparams.ft_overlap, window=self.window_cpu, win_length=self.hparams.ft_len, return_complex=True)
        Ys = Ys.reshape((batch_size, spk_num, Ys.shape[-2], Ys.shape[-1]))  # (batch, spk, freq, time)
        # Ys = Ys.permute(0, 2, 1, 3)  # (batch, freq, spk, time) for the reason of narrow band pit (each freq can be regarded as a part of batch)

        batch_size, freq_num, time, chn_num = X.shape
        # normalization by using ref_channel
        Xr = X[..., self.ref_chn_idx].clone()  # copy
        XrMM = torch.abs(Xr).mean(dim=2)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of Xm
        X[:, :, :, :] /= (XrMM.reshape(batch_size, freq_num, 1, 1) + 1e-8)

        # concatenate neighbour freq bands
        band_num = self.hparams.band_num
        X_bands = torch.zeros((batch_size, freq_num - band_num + 1, time, band_num, chn_num, 2), device=x.device)
        Ys_bands = torch.zeros((batch_size, spk_num, freq_num - band_num + 1, time, band_num), device=x.device, dtype=torch.complex64)
        Xr_bands = torch.zeros((batch_size, freq_num - band_num + 1, time, band_num), device=x.device, dtype=torch.complex64)
        XrMM_bands = torch.zeros((batch_size, freq_num - band_num + 1, band_num), device=x.device)
        for f in range(freq_num - band_num + 1):
            for band in range(band_num):
                X_bands[:, f, :, band, :, 0] = torch.real(X[:, f + band, :, :])
                X_bands[:, f, :, band, :, 1] = torch.imag(X[:, f + band, :, :])
                Ys_bands[:, :, f, :, band] = Ys[:, :, f + band, :]
                Xr_bands[:, f, :, band] = Xr[:, f + band, :]
                XrMM_bands[:, f, band] = XrMM[:, f + band]

        return X_bands, Ys_bands, Xr_bands, XrMM_bands

    def collate_func_train(self, batches):
        x, ys, _ = SS_SemiOnlineDataset.collate_fn(batches)
        x = x[:, self.hparams.channels, :]
        ys = ys[:, :, self.ref_chn_idx, :]
        with torch.no_grad():
            X_bands, Ys_bands, Xr_bands, XrMM_bands = self.encode(x, ys)
        return X_bands, Ys_bands, Xr_bands, XrMM_bands, ys

    def training_step(self, batch, batch_idx):
        # get input
        X_bands, Ys_bands, Xr_bands, XrMM_bands, ys = batch

        # forward
        fploss, targets, estimates, outputs, bps_t = self.run_one_step(X_bands, Ys_bands, Xr_bands, XrMM_bands, ys)
        if self.hparams.target_type == 'WavForm':
            self.log('train/neg_si_sdr', fploss)
        else:
            self.log('train/fploss', fploss)
        return fploss

    def collate_func_val(self, batches):
        x, ys, _ = SS_SemiOnlineDataset.collate_fn(batches)
        x = x[:, self.hparams.channels, :]
        ys = ys[:, :, self.ref_chn_idx, :]
        with torch.no_grad():
            X_bands, Ys_bands, Xr_bands, XrMM_bands = self.encode(x, ys)
        return ys, X_bands, Ys_bands, Xr_bands, XrMM_bands

    def decode(self, Ys_bands_hat, original_len):
        """Input the version solved freq permutation, output time domain signal

        Args:
            Ys_bands_hat (torch.Tensor): permuted version of Ys_bands_hat

        Returns:
            torch.Tensor: time domain signal of shape [batch_size, spk_num, time]
        """
        batch_size, spk_num, freq_num_band, time, band_num = Ys_bands_hat.shape
        freq_num = freq_num_band + band_num - 1
        # align overlaping bands
        Ys_bands_hat_aligned = torch.empty((batch_size, spk_num, freq_num, time, band_num), device=Ys_bands_hat.device, dtype=torch.complex64)
        for f in range(freq_num):
            for b in range(band_num):
                if f - b >= 0 and f - b < freq_num_band:
                    Ys_bands_hat_aligned[:, :, f, :, b] = Ys_bands_hat[:, :, f - b, :, b]
                else:
                    valid_bands = []
                    if not (f - b >= 0):
                        for bb in range(f + 1):
                            valid_bands.append(Ys_bands_hat[:, :, f - bb, :, bb])
                    else:  # not (f - b < freq_num_band)
                        for bb in range(f - freq_num_band + 1, band_num):
                            valid_bands.append(Ys_bands_hat[:, :, f - bb, :, bb])
                    vbcat = torch.cat(valid_bands, 0).view(len(valid_bands), *valid_bands[0].shape)
                    Ys_bands_hat_aligned[:, :, f, :, b] = torch.mean(vbcat, dim=0)[:, :, :]
        # average
        Ys_hat = torch.mean(Ys_bands_hat_aligned, dim=4)  # (batch, spk_num, freq, time)
        # istft
        ys_hat = torch.istft(Ys_hat.reshape(batch_size * spk_num, freq_num, time),
                             n_fft=self.hparams.ft_len,
                             hop_length=self.hparams.ft_overlap,
                             window=self.window,
                             win_length=self.hparams.ft_len,
                             length=original_len)
        ys_hat = ys_hat.reshape(batch_size, spk_num, ys_hat.shape[1])
        return ys_hat

    def validation_step(self, batch, batch_idx):
        ys, X_bands, Ys_bands, Xr_bands, XrMM_bands = batch
        batch_size, spk_num, freq_num_band, time, band_num = Ys_bands.shape

        fploss, targets, estimates, outputs, bps_t = self.run_one_step(X_bands, Ys_bands, Xr_bands, XrMM_bands, ys)
        if self.hparams.target_type in ['MRM', 'CIRM']:
            # Xr_bands [batch, freq, time, band] complex
            # estimates [batch, spk, freq, time, band]
            if self.hparams.target_type == 'CIRM':
                cirm_cmpd = torch.view_as_real(estimates)
                cirm_cmpd = self.LIM * (cirm_cmpd >= self.LIM) - self.LIM * (cirm_cmpd <= -self.LIM) + cirm_cmpd * (cirm_cmpd.abs() < self.LIM)  # to (-10, 10) not [-10, 10]
                cirm_uncmpd = -1 / self.C * torch.log((self.K - cirm_cmpd) / (self.K + cirm_cmpd))
                mask = torch.view_as_complex(cirm_uncmpd)
            else:
                mask = estimates

            Ys_bands_hat = torch.empty(size=Ys_bands.shape, dtype=torch.complex64, device=outputs.device)
            for spk in range(self.hparams.speaker_num):  # Xr_bands * mask
                Ys_bands_hat[:, spk, :, :, :] = Xr_bands[:, :, :, :] * mask[:, spk, :, :, :]
        elif self.hparams.target_type == 'TFMapping':
            Ys_bands_hat = estimates
        else:
            assert self.hparams.target_type == "WavForm"
            ys_hat = estimates
            self.log('val/neg_si_sdr', fploss, sync_dist=True)
            return fploss

        self.log('val/fploss', fploss, sync_dist=True)

        # 如果band_num>1，则使用均值的方式组装各个频带并计算si_sdr
        ys_hat = self.decode(Ys_bands_hat, ys.shape[-1])  # ifPIT

        targets_f = targets.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, freq_num, spk_num, time, band_num] from [batch_size, spk_num, freq_num, time, band_num]
        estimates_f = estimates.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, freq_num, spk_num, time, band_num] from [batch_size, spk_num, freq_num, time, band_num]
        targets_f = targets_f.view(batch_size * freq_num_band, spk_num, time, band_num)
        estimates_f = estimates_f.view(batch_size * freq_num_band, spk_num, time, band_num)

        if estimates_f.dtype.is_complex:
            fploss_t, ps_t = pit(target=torch.view_as_real(targets_f), preds=torch.view_as_real(estimates_f), metric_func=self.loss_func, eval_func='min')
        else:
            fploss_t, ps_t = pit(target=targets_f, preds=estimates_f, metric_func=self.loss_func, eval_func='min')

        self.log('val/fploss_t', fploss_t.mean().item())
        fbps_t = ps_t.view(batch_size, freq_num_band, spk_num)
        fbps_e = bps_t.unsqueeze(1).expand(-1, freq_num_band, -1)

        # 计算permutation正确数量
        right_avg, wrong_avg, wfs = permutation_analysis(bps_t=fbps_t, bps_e=fbps_e)
        self.log('val/perm_right', right_avg)
        self.log('val/perm_wrong_local', wrong_avg)

        Ys_bands_hat_f = Ys_bands_hat.permute(0, 2, 1, 3, 4).contiguous()  # Ys_bands_hat (batch, spk, freq, time, band_num) complex
        Ys_bands_hat_per2 = perm_by_ps(Ys_bands_hat_f, fbps_t)
        ys_hat_t = decode(Ys_bands_hat_per2, self.hparams.ft_len, self.hparams.ft_overlap, ys.shape[-1])

        # 计算si-sdr
        si_sdr_val, ps = pit(preds=ys_hat, target=ys, metric_func=si_sdr, eval_func='max')
        self.log('val/neg_si_sdr', -si_sdr_val.mean())
        si_sdr_t_val, ps = pit(preds=ys_hat_t, target=ys, metric_func=si_sdr, eval_func='max')
        self.log('val/neg_si_sdr_t', -si_sdr_t_val.mean())

        return fploss

    def on_test_epoch_start(self):
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path, exist_ok=True)
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
            path = os.path.join(self.exp_save_path, 'results_{}.json'.format(dtstr))
            # write results to json
            f = open(path, 'w', encoding='utf-8')
            json.dump(results, f, indent=4, cls=NumpyEncoder)
            f.close()
            # write mean to json
            df = DataFrame(results)
            df['except_num'] = self.expt_num
            df.mean().to_json(os.path.join(self.exp_save_path, 'results_mean.json'), indent=4)
        print('except num ' + str(self.expt_num))

    def collate_func_test(self, batches):
        x, ys, paras = SS_SemiOnlineDataset.collate_fn(batches)
        x = x[:, self.hparams.channels, :]
        ys = ys[:, :, self.ref_chn_idx, :]
        with torch.no_grad():
            X_bands, Ys_bands, Xr_bands, XrMM_bands = self.encode(x, ys)
        return x, ys, X_bands, Ys_bands, Xr_bands, XrMM_bands, paras

    def test_step(self, batch, batch_idx):
        x, ys, X_bands, Ys_bands, Xr_bands, XrMM_bands, paras = batch  # the last one might be path batch[6]
        batch_size, spk_num, freq_num_band, time, band_num = Ys_bands.shape

        # predict
        fploss, targets, estimates, outputs, bps_t = self.run_one_step(X_bands, Ys_bands, Xr_bands, XrMM_bands, ys)
        if self.hparams.target_type in ['MRM', 'CIRM']:
            # Xr_bands [batch, freq, time, band] complex
            # estimates [batch, spk, freq, time, band]
            if self.hparams.target_type == 'CIRM':
                cirm_cmpd = torch.view_as_real(estimates)
                cirm_cmpd = self.LIM * (cirm_cmpd >= self.LIM) - self.LIM * (cirm_cmpd <= -self.LIM) + cirm_cmpd * (cirm_cmpd.abs() < self.LIM)  # to (-10, 10) not [-10, 10]
                cirm_uncmpd = -1 / self.C * torch.log((self.K - cirm_cmpd) / (self.K + cirm_cmpd))
                mask = torch.view_as_complex(cirm_uncmpd)
            else:
                mask = estimates

            Ys_bands_hat = torch.empty(size=Ys_bands.shape, dtype=torch.complex64, device=outputs.device)
            for spk in range(self.hparams.speaker_num):  # Xr_bands * mask
                Ys_bands_hat[:, spk, :, :, :] = Xr_bands[:, :, :, :] * mask[:, spk, :, :, :]
        elif self.hparams.target_type == 'TFMapping':
            Ys_bands_hat = estimates
        else:
            assert self.hparams.target_type == "WavForm"
            ys_hat = estimates

            Ys_bands_hat = torch.empty(size=(batch_size, spk_num, freq_num_band, time, band_num), dtype=torch.complex64, device=outputs.device)
            XrMM_bands = torch.unsqueeze(XrMM_bands, dim=2).expand(-1, -1, time, -1)
            for spk in range(spk_num):
                Ys_bands_hat[:, spk, :, :, :] = outputs[:, :, :, spk, :] * XrMM_bands[:, :, :, :]

        if self.hparams.target_type != "WavForm":
            self.log('test/fploss', fploss)
        else:
            self.log('test/neg_si_sdr', fploss)

        # write results & infos
        wavname = os.path.basename(f"{paras[0]['index']}.wav")
        result_dict = {'id': batch_idx, 'wavname': wavname, 'fploss': fploss.item()}

        def corr_perm(estimates):
            estimates_f = estimates.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, freq_num, spk_num, time, band_num] from [batch_size, spk_num, freq_num, time, band_num]
            # solve F permutation problem using correlation method
            Ys_bands_hat_per, fbps_corr, avg_loss = perm_by_correlation(estimates_f)
            ys_hat_corr = decode(Ys_bands_hat_per, self.hparams.ft_len, self.hparams.ft_overlap, ys.shape[-1])
            return ys_hat_corr

        # compute the frequency
        if self.hparams.target_type != "WavForm":
            ys_hat = self.decode(Ys_bands_hat, ys.shape[-1])  # to time domain signals
            ys_hat_corr = corr_perm(estimates=estimates)
        else:
            ys_hat_corr = corr_perm(estimates=Ys_bands_hat)

        try:
            # calculate metrics
            x_ref = x[0, self.ref_chn_idx, :]
            _, ps = pit(preds=ys_hat, target=ys, metric_func=si_sdr, eval_func='max')
            ys_hat_perm = pit_permutate(ys_hat, ps)  # reorder first
            metrics, input_metrics, imp_metrics = cal_metrics_functional(COMMON_AUDIO_METRICS, ys_hat_perm[0], ys[0], x_ref.expand_as(ys[0]), 16000)
            for key, val in imp_metrics.items():
                self.log('test/' + key, val)
                result_dict[key] = val
            result_dict.update(metrics)
            result_dict.update(input_metrics)

            # calculate metrics for corr
            _, ps = pit(preds=ys_hat_corr, target=ys, metric_func=si_sdr, eval_func='max')
            ys_hat_corr_perm = pit_permutate(ys_hat_corr, ps)
            metrics, input_metrics, imp_metrics = cal_metrics_functional(COMMON_AUDIO_METRICS, ys_hat_corr_perm[0], ys[0], input_metrics, 16000)
            for key, val in imp_metrics.items():
                self.log('test/' + key + '_corr', val)
                result_dict[key + '_corr'] = val
            for key, val in metrics.items():
                result_dict[key + '_corr'] = val
        except:
            self.expt_num = self.expt_num + 1

        # write examples
        if paras[0]['index'] < 200:
            sw: SummaryWriter = self.logger.experiment

            abs_max = max(torch.max(torch.abs(x[0,])), torch.max(torch.abs(ys_hat_perm[0,])), torch.max(torch.abs(ys[0,])))

            def write_wav(wav_path: str, tag: str, wav: torch.Tensor):
                # make sure wav don't have illegal values (abs greater than 1)
                if abs_max > 1:
                    wav /= abs_max
                sw.add_audio(tag=tag, snd_tensor=wav, sample_rate=16000)
                sf.write(wav_path, wav.detach().cpu().numpy(), 16000)

            pattern = '.'.join(wavname.split('.')[:-1]) + '{name}'
            example_dir = os.path.join(self.exp_save_path, 'examples', str(paras[0]['index']))
            os.makedirs(example_dir, exist_ok=True)
            for i in range(self.hparams.speaker_num):
                # write ys
                wp = os.path.join(example_dir, pattern.format(name=f"_spk{i+1}.wav"))
                write_wav(wav_path=wp, tag=pattern.format(name=f"/spk{i+1}.wav"), wav=ys[0, i])
                # write ys_hat
                wp = os.path.join(example_dir, pattern.format(name=f"_spk{i+1}_p.wav"))
                write_wav(wav_path=wp, tag=pattern.format(name=f"/spk{i+1}_p.wav"), wav=ys_hat_perm[0, i])
            # write mix
            wp = os.path.join(example_dir, pattern.format(name=f"_mix.wav"))
            write_wav(wav_path=wp, tag=pattern.format(name=f"/mix.wav"), wav=x[0, self.ref_chn_idx])
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, **self.hparams.optimizer_kwargs)

        if self.hparams.lr_scheduler != None and len(self.hparams.lr_scheduler) > 0:
            lr_scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler(optimizer, **self.hparams.lr_scheduler_kwargs),
                    'monitor': 'val/neg_si_sdr',
                }
            }
        else:
            return optimizer


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Tensor):
            return obj.detach().cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)
