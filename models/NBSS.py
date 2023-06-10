import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn as nn
from data_loaders import SS_SemiOnlineDataset
from pandas import DataFrame
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr

from models.io.narrow_band import NBIO
from models.utils import MyJsonEncoder, tag_and_log_git_status
from models.utils.metrics import cal_metrics_functional


class NBSS(pl.LightningModule):
    """Narrow-band deep speech separation with full-band PIT used, \
        which controls the training, testing, and inference of NBSS with given arch and io
    """

    def __init__(
        self,
        arch: nn.Module,
        io: NBIO,
        speaker_num: int = 2,
        ref_channel: int = 0,
        channels: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
        learning_rate: float = 0.001,
        optimizer: str = "Adam",
        optimizer_kwargs: Dict[str, Any] = dict(),
        lr_scheduler: str = 'ReduceLROnPlateau',
        lr_scheduler_kwargs: Dict[str, Any] = {
            'mode': 'min',
            'factor': 0.5,
            'patience': 10,
            'min_lr': 1e-4
        },
        exp_name: str = "exp",
        metrics: List[str] = ['SNR', 'SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ'],
    ):
        """
        Args:
            arch: the network module, whose input_size and output_size are given by the NBIO class
            io: narrow-band input, output, and loss for the given arch
            speaker_num: Defaults to 2.
            ref_channel: Defaults to 0.
            channels: Defaults to [0, 1, 2, 3, 4, 5, 6, 7].
            learning_rate: Defaults to 0.001.
            optimizer: Defaults to "Adam".
            optimizer_kwargs: Defaults to dict().
            lr_scheduler: give None to remove lr_scheduler. Defaults to 'ReduceLROnPlateau'.
            lr_scheduler_kwargs: Defaults to { 'mode': 'min', 'factor': 0.5, 'patience': 10, 'min_lr': 1e-4 }.
            exp_name: set exp_name to notag when debug things. Defaults to "exp".
            metrics: metrics used at test time. Defaults to ['SNR', 'SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ'].
        """

        super().__init__()

        # save all the hyperparameters to self.hparams
        self.save_hyperparameters(ignore=['arch', 'io'])
        
        self.ref_chn_idx = channels.index(ref_channel)

        self.io = io
        self.arch = arch

    def on_train_start(self):
        """Called by PytorchLightning automatically at the start of training"""
        if self.current_epoch == 0:
            # self.logger.log_hyperparams(self.hparams, {"val/neg_si_sdr": 0, "val/fploss": 1})

            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir') and 'notag' not in self.hparams.exp_name:
                # add git tags for better change tracking
                # note: if change self.logger.log_dir to self.trainer.log_dir, the training will stuck on multi-gpu training
                tag_and_log_git_status(self.logger.log_dir + '/git.out', self.logger.version, self.hparams.exp_name, model_name='NBSS')

            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir'):
                with open(self.logger.log_dir + '/model.txt', 'a') as f:
                    f.write(str(self))
                    f.write('\n\n\n')

    def forward(self, input) -> Tuple[Tensor, Tensor]:
        """returns the preds and raw output from arch"""
        if isinstance(input, tuple) or isinstance(input, list):
            output = self.arch(input[0])
        else:
            output = self.arch(input)
        preds = self.io.prepare_prediction(output, input)  # frequency binding is applied implicitly
        return preds, output

    def collate_func_train(self, batches):
        """called by dataloader threads on CPU for preparing the training batches"""
        x, ys, _ = SS_SemiOnlineDataset.collate_fn(batches)
        x = x[:, self.hparams.channels, :]
        ys = ys[:, :, self.hparams.channels, :]
        ys = ys[:, :, self.ref_chn_idx, :]
        with torch.no_grad():
            input = self.io.prepare_input(x)
            target = self.io.prepare_target(ys, input=input)
        return input, target

    def training_step(self, batch, batch_idx):
        """training step on self.device, called automaticly by PytorchLightning"""
        input, target = batch

        preds, output = self.forward(input)
        loss, perms = self.io.loss(preds, target)
        self.log('train/' + self.io.loss_name, loss, batch_size=target.shape[0])
        return loss

    def collate_func_val(self, batches):
        """called by dataloader threads on CPU for preparing the validation batches"""
        x, ys, _ = SS_SemiOnlineDataset.collate_fn(batches)
        x = x[:, self.hparams.channels, :]
        ys = ys[:, :, self.ref_chn_idx, :]
        with torch.no_grad():
            input = self.io.prepare_input(x)
            target = self.io.prepare_target(ys, input=input)
        return input, target, ys

    def validation_step(self, batch, batch_idx):
        """validation step on self.device, called automaticly by PytorchLightning"""
        input, target, ys = batch

        preds, output = self.forward(input)
        loss, perms = self.io.loss(preds, target)
        self.log('val/' + self.io.loss_name, loss, sync_dist=True, batch_size=ys.shape[0])

        if self.io.loss_name != 'neg_si_sdr':
            # always computes the neg_si_sdr for the comparison of different runs in Tensorboard
            ys_hat = self.io.prepare_time_domain(output, input, preds)
            si_sdr_val, perms = pit(ys_hat, ys, metric_func=si_sdr, eval_func='max')
            self.log('val/neg_si_sdr', -si_sdr_val.mean(), sync_dist=True, batch_size=ys.shape[0])
        return loss

    def on_test_epoch_start(self):
        """Called by PytorchLightning automatically at the start of test epoch"""
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path, exist_ok=True)
        self.expt_num = 0

    def on_test_epoch_end(self, results):
        """Called by PytorchLightning automatically at the end of test epoch"""
        import torch.distributed as dist

        # collect results from other gpus if world_size > 1
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

        # save collected data on 0-th gpu
        if self.trainer.is_global_zero:
            # Tensor to list or number
            for r in results:
                for key, val in r.items():
                    if isinstance(val, Tensor):
                        if val.numel() == 1:
                            r[key] = val.item()
                        else:
                            r[key] = val.detach().cpu().numpy().tolist()
            # save
            import datetime
            x = datetime.datetime.now()
            dtstr = x.strftime('%Y%m%d_%H%M%S.%f')
            path = os.path.join(self.exp_save_path, 'results_{}.json'.format(dtstr))
            # write results to json
            f = open(path, 'w', encoding='utf-8')
            json.dump(results, f, indent=4, cls=MyJsonEncoder)
            f.close()
            # write mean to json
            df = DataFrame(results)
            df['except_num'] = self.expt_num
            df.mean(numeric_only=True).to_json(os.path.join(self.exp_save_path, 'results_mean.json'), indent=4)
            self.print('results: ', os.path.join(self.exp_save_path, 'results_mean.json'), ' ', path)
        print('except num ' + str(self.expt_num))

    def collate_func_test(self, batches):
        """called by dataloader threads on CPU for preparing the test batches"""
        x, ys, paras = SS_SemiOnlineDataset.collate_fn(batches)
        x = x[:, self.hparams.channels, :]
        ys = ys[:, :, self.ref_chn_idx, :]
        with torch.no_grad():
            input = self.io.prepare_input(x)
            target = self.io.prepare_target(ys, input=input)
        return input, target, x, ys, paras

    def test_step(self, batch, batch_idx):
        """test step on self.device, called automaticly by PytorchLightning"""
        input, target, x, ys, paras = batch
        sample_rate = 16000 if 'sample_rate' not in paras[0] else paras[0]['sample_rate']

        preds, output = self.forward(input)
        loss, perms = self.io.loss(preds, target)
        self.log('test/' + self.io.loss_name, loss, logger=False, batch_size=ys.shape[0])

        ys_hat = self.io.prepare_time_domain(output, input, preds)

        # write results & infos
        wavname = os.path.basename(f"{paras[0]['index']}.wav")
        result_dict = {'id': batch_idx, 'wavname': wavname, self.io.loss_name: loss.item()}

        # calculate metrics
        exception = False
        try:
            x_ref = x[0, self.ref_chn_idx, :]
            # recover wav's original scale. solve min ||Y^T a - xref|| to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
            a = torch.linalg.lstsq(ys_hat[0,].T, x_ref.unsqueeze(-1)).solution
            ys_hat = ys_hat * a.unsqueeze(0)
            # reorder
            _, perms = pit(preds=ys_hat, target=ys, metric_func=si_sdr, eval_func='max')
            ys_hat_perm = pit_permutate(ys_hat, perms)  # reorder first by using si_sdr metric
            # calculate metrics, input_metrics, improve_metrics
            metrics, input_metrics, imp_metrics = cal_metrics_functional(self.hparams.metrics, ys_hat_perm[0], ys[0], x_ref.expand_as(ys[0]), sample_rate)
            for key, val in imp_metrics.items():
                self.log('test/' + key, val, logger=False, batch_size=ys.shape[0])
                result_dict[key] = val
            result_dict.update(metrics)
            result_dict.update(input_metrics)
        except Exception as e:
            # exception might happen
            exception = True
            import warnings
            warnings.warn(str(batch_idx) + ": " + str(e))
            self.expt_num = self.expt_num + 1

        # write examples
        if paras[0]['index'] < 200 or exception:
            abs_max = max(torch.max(torch.abs(x[0,])), torch.max(torch.abs(ys[0,])))

            def write_wav(wav_path: str, wav: torch.Tensor, norm_to: torch.Tensor = None):
                # make sure wav don't have illegal values (abs greater than 1)
                if norm_to:
                    wav = wav / torch.max(torch.abs(wav)) * norm_to
                if abs_max > 1:
                    wav /= abs_max
                sf.write(wav_path, wav.detach().cpu().numpy(), sample_rate)

            pattern = '.'.join(wavname.split('.')[:-1]) + '{name}'  # remove .wav in wavname
            example_dir = os.path.join(self.exp_save_path, 'examples', str(paras[0]['index']))
            os.makedirs(example_dir, exist_ok=True)
            for i in range(self.hparams.speaker_num):
                # write ys
                wav_path = os.path.join(example_dir, pattern.format(name=f"_spk{i+1}.wav"))
                write_wav(wav_path=wav_path, wav=ys[0, i])
                # write ys_hat
                wav_path = os.path.join(example_dir, pattern.format(name=f"_spk{i+1}_p.wav"))
                write_wav(wav_path=wav_path, wav=ys_hat_perm[0, i])  #, norm_to=ys[0, i].abs().max())
            # write mix
            wav_path = os.path.join(example_dir, pattern.format(name=f"_mix.wav"))
            write_wav(wav_path=wav_path, wav=x[0, self.ref_chn_idx])
            # write paras
            f = open(os.path.join(example_dir, pattern.format(name=f"_paras.json")), 'w', encoding='utf-8')
            paras[0]['metrics'] = result_dict
            json.dump(paras[0], f, indent=4, cls=MyJsonEncoder)
            f.close()

        # return metrics, which will be collected, saved in test_epoch_end
        if 'metrics' in paras[0]:
            del paras[0]['metrics']  # remove circular reference
        result_dict['paras'] = paras[0]
        return result_dict

    def on_predict_epoch_start(self) -> None:
        """Called by PytorchLightning automatically at the start of predict epoch"""
        import time
        self.predict_start_time = time.time()
        return super().on_predict_epoch_start()

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        """Called by PytorchLightning automatically at the end of predict epoch"""
        import time
        predict_end_time = time.time()
        t = predict_end_time - self.predict_start_time
        print(f'time total={t:.3f}s, per uttr={t/len(results[0]):.3f}s, uttr num={len(results[0])}')
        return super().on_predict_epoch_end(results)

    def predict_step(self, batch: Union[Tensor, Tuple[Tensor, Tensor, Dict]], batch_idx: Optional[int] = None, dataloader_idx: Optional[int] = None) -> Tensor:
        """predict step on self.device, could be called dirctly or by PytorchLightning automatically using predict dataset
        Args:
            batch: x or (x, ys, paras). shape of x [B, C, T]

        Returns:
            Tensor: ys_hat, shape [B, Spk, T]
        """
        if isinstance(batch, Tensor):
            x = batch
        else:
            x, ys, paras = batch
        x = x[:, self.hparams.channels, :]
        input = self.io.prepare_input(x)

        preds, output = self.forward(input)
        ys_hat = self.io.prepare_time_domain(output, input, preds)

        return ys_hat

    def configure_optimizers(self):
        """configure optimizer and lr_scheduler"""
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.learning_rate, **self.hparams.optimizer_kwargs)

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
