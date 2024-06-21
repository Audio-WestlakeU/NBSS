import json
import os
from typing import *
from pathlib import Path

import pytorch_lightning as pl
import soundfile as sf
import torch
from numpy import ndarray
from pandas import DataFrame
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch import Tensor

from models.utils import MyJsonEncoder, tag_and_log_git_status
from models.utils.ensemble import ensemble
from models.utils.flops import write_FLOPs
from models.utils.metrics import (cal_metrics_functional, cal_pesq, recover_scale)


def on_validation_epoch_end(self: pl.LightningModule, cpu_metric_input: List[Tuple[ndarray, ndarray, int]], N: int = 5) -> None:
    """calculate heavy metrics for every N epochs

    Args:
        self: LightningModule
        cpu_metric_input: the input list for cal_metrics_functional
        N: the number of epochs. Defaults to 5.
    """

    if self.current_epoch != 0 and self.current_epoch % N != (N - 1):
        cpu_metric_input.clear()
        return

    if len(cpu_metric_input) == 0:
        return

    torch.multiprocessing.set_sharing_strategy('file_system')
    num_thread = torch.multiprocessing.cpu_count() // (self.trainer.world_size * 2)
    p = torch.multiprocessing.Pool(min(num_thread, len(cpu_metric_input)))
    cpu_metrics = list(p.starmap(cal_metrics_functional, cpu_metric_input))
    p.close()
    p.join()

    for k in cpu_metric_input[0][0]:
        ms = list(filter(None, [m[0][k.lower()] for m in cpu_metrics]))
        if len(ms) > 0:
            self.log(f'val/{k}', sum(ms) / len(ms), sync_dist=True, batch_size=len(ms))

    cpu_metric_input.clear()


def on_test_epoch_end(self: pl.LightningModule, results: List[Dict[str, Any]], cpu_metric_input: List, exp_save_path: str):
    """ calculate cpu metrics on CPU, collect results, save results to file

    Args:
        self: LightningModule
        results: the result list
        cpu_metric_input: the input list for cal_metrics_functional
        exp_save_path: the path to save result file
    """

    # calculate metrics, input_metrics, improve_metrics on CPU using multiprocessing to speed up
    torch.multiprocessing.set_sharing_strategy('file_system')
    num_thread = torch.multiprocessing.cpu_count() // (self.trainer.world_size * 2)
    p = torch.multiprocessing.Pool(min(num_thread, len(cpu_metric_input)))
    cpu_metrics = list(p.starmap(cal_metrics_functional, cpu_metric_input))
    p.close()
    p.join()
    for i, m in enumerate(cpu_metrics):
        metrics, input_metrics, imp_metrics = m
        results[i].update(input_metrics)
        results[i].update(imp_metrics)
        results[i].update(metrics)

    # gather results from all GPUs
    import torch.distributed as dist

    # collect results from other gpus if world_size > 1
    if self.trainer.world_size > 1:
        dist.barrier()
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
        # save
        import datetime
        x = datetime.datetime.now()
        dtstr = x.strftime('%Y%m%d_%H%M%S.%f')
        path = os.path.join(exp_save_path, 'results_{}.json'.format(dtstr))
        # write results to json
        f = open(path, 'w', encoding='utf-8')
        json.dump(results, f, indent=4, cls=MyJsonEncoder)
        f.close()
        # write mean to json
        df = DataFrame(results)
        df.mean(numeric_only=True).to_json(os.path.join(exp_save_path, 'results_mean.json'), indent=4)
        self.print('results: ', os.path.join(exp_save_path, 'results_mean.json'), ' ', path)


def on_predict_batch_end(
    self: pl.LightningModule,
    outputs: Optional[Any],
    batch: Any,
) -> None:
    """save predicted results to `log_dir/examples`

    Args:
        self: LightningModule
        outputs: _description_
        batch: _description_
    """
    save_dir = self.trainer.logger.log_dir + '/' + 'examples'
    os.makedirs(save_dir, exist_ok=True)

    if not isinstance(batch, Tensor):
        input, target, paras = batch
        if 'saveto' in paras[0]:
            for b in range(len(paras)):
                saveto = paras[b]['saveto']
                if isinstance(saveto, str):
                    saveto = [saveto]

                assert isinstance(saveto, list), ('saveto should be a list of size num_speakers', type(saveto))
                for spk, spk_saveto in enumerate(saveto):
                    if isinstance(spk_saveto, dict):
                        input_saveto = spk_saveto['input'] if 'input' in spk_saveto else None
                        target_saveto = spk_saveto['target'] if 'target' in spk_saveto else None
                        pred_saveto = spk_saveto['prediction'] if 'prediction' in spk_saveto else None
                    else:
                        pred_saveto, input_saveto, target_saveto = spk_saveto, None, None

                    # save predictions
                    if pred_saveto:
                        y = outputs[b][spk]
                        assert len(y.shape) == 1, y.shape
                        save_path = Path(save_dir) / pred_saveto
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        sf.write(save_path, y.detach().cpu().numpy(), samplerate=paras[b]['sample_rate'])
                    # save input
                    if input_saveto:
                        y = input[b].T  # [T,CHN]
                        save_path = Path(save_dir) / input_saveto
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        sf.write(save_path, y.detach().cpu().numpy(), samplerate=paras[b]['sample_rate'])
                    # # save target
                    # if input_saveto and target is not None:
                    #     y = target[b, spk, :, :].T  # [T,CHN]
                    #     save_path = save_dir + '/' + target_saveto
                    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #     sf.write(save_path, y.detach().cpu().numpy(), samplerate=paras[b]['sample_rate'])


def on_load_checkpoint(
    self: pl.LightningModule,
    checkpoint: Dict[str, Any],
    ensemble_opts: Union[int, str, List[str], Literal[None]] = None,
    compile: bool = True,
    reset: List[str] = [],
) -> None:
    """load checkpoint

    Args:
        self: LightningModule
        checkpoint: the loaded weights
        ensemble_opts: opts for ensemble. Defaults to None.
        compile: whether the checkpoint is a compiled one. Defaults to True.
    """
    from pytorch_lightning.strategies import FSDPStrategy
    if isinstance(self.trainer.strategy, FSDPStrategy):
        rank_zero_warn('using fsdp, ensemble is disenabled')
        return super(pl.LightningModule, self).on_load_checkpoint(checkpoint)

    if ensemble_opts:
        ckpt = self.trainer.ckpt_path
        ckpts, state_dict = ensemble(opts=ensemble_opts, ckpt=ckpt)
        self.print(f'rank {self.trainer.local_rank}/{self.trainer.world_size}, ensemble {ensemble_opts}: {ckpts}')
        checkpoint['state_dict'] = state_dict

    # rename weights for removing _orig_mod in name
    name_mapping = {} # {name without _orig_mod: the actual name}
    parameters = self.state_dict()
    for k, v in parameters.items():
        name_mapping[k.replace('_orig_mod.', '')] = k

    state_dict = checkpoint['state_dict']
    state_dict_new = dict()
    for k, v, in state_dict.items():
        state_dict_new[name_mapping[k.replace('_orig_mod.', '')]] = v
    checkpoint['state_dict'] = state_dict_new

    # reset optimizer and lr_scheduler
    if reset is not None:
        for key in reset:
            assert key in ['optimizer', 'lr_scheduler'], f'unsupported reset key {key}'
            if key == 'optimizer':
                checkpoint['optimizer'] = dict()
                checkpoint['optimizer_states'] = []
                rank_zero_info('reset optimizer')
            elif key == 'lr_scheduler':
                checkpoint['lr_scheduler'] = dict()
                checkpoint['lr_schedulers'] = []
                rank_zero_info('reset lr_scheduler')

    return super(pl.LightningModule, self).on_load_checkpoint(checkpoint)


def on_train_start(self: pl.LightningModule, exp_name: str, model_name: str, num_chns: int, nfft: int, model_class_path: str = None):
    """ 1) add git tags/write requirements for better change tracking; 2) write model architecture to file; 3) measure the model FLOPs

    Args:
        self: LightningModule
        exp_name: `notag` or exp_name, add git tag e.g. 'model_name_v10' if exp_name!='notag'
        model_name: the model name
        num_chns: the number of channels for FLOPs test
        nfft: the number of fft points
        model_class_path: the path to import the self
    """
    if self.current_epoch == 0:
        if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir') and 'notag' not in exp_name:
            # add git tags for better change tracking
            # note: if change self.logger.log_dir to self.trainer.log_dir, the training will stuck on multi-gpu training
            tag_and_log_git_status(self.logger.log_dir + '/git.out', self.logger.version, exp_name, model_name=model_name)

        if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir'):
            # write model architecture to file
            with open(self.logger.log_dir + '/model.txt', 'a') as f:
                f.write(str(self))
                f.write('\n\n\n')
            # measure the model FLOPs, the num_chns here only means the original channels
            # write_FLOPs(model=self, save_dir=self.logger.log_dir, num_chns=num_chns, nfft=nfft, model_class_path=model_class_path)


def configure_optimizers(
    self: pl.LightningModule,
    optimizer: str,
    optimizer_kwargs: Dict[str, Any],
    monitor: str = 'val/loss',
    lr_scheduler: str = None,
    lr_scheduler_kwargs: Dict[str, Any] = None,
):
    """configure optimizer and lr_scheduler"""
    if optimizer == 'Adam' and self.trainer.precision == '16-mixed':
        if 'eps' not in optimizer_kwargs:
            optimizer_kwargs['eps'] = 1e-4  # according to https://discuss.pytorch.org/t/adam-half-precision-nans/1765
            rank_zero_info('setting the eps of Adam to 1e-4 for FP16 mixed precision training')
        else:
            allowed_minimum = torch.finfo(torch.float16).eps
            assert optimizer_kwargs['eps'] >= allowed_minimum, f"You should specify an eps greater than the allowed minimum of the FP16 precision: {optimizer_kwargs['eps']} {allowed_minimum}"
    optimizer = getattr(torch.optim, optimizer)(self.parameters(), **optimizer_kwargs)

    if lr_scheduler is not None and len(lr_scheduler) > 0:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler(optimizer, **lr_scheduler_kwargs),
                'monitor': monitor,
            }
        }
    else:
        return optimizer


def test_setp_write_example(self, xr: Tensor, yr: Tensor, yr_hat: Tensor, sample_rate: int, paras: Dict[str, Any], result_dict: Dict[str, Any], wavname: str, exp_save_path: str):
    """
    Args:
        xr: [B,T]
        yr: [B,Spk,T]
        yr_hat: [B,Spk,T]
    """

    # write examples
    abs_max = max(torch.max(torch.abs(xr[0, ...])), torch.max(torch.abs(yr[0, ...])))

    def write_wav(wav_path: str, wav: torch.Tensor, norm_to: torch.Tensor = None):
        # make sure wav don't have illegal values (abs greater than 1)
        if norm_to:
            wav = wav / torch.max(torch.abs(wav)) * norm_to
        if abs_max > 1:
            wav /= abs_max
        abs_max_wav = torch.max(torch.abs(wav))
        if abs_max_wav > 1:
            import warnings
            warnings.warn(f"abs_max_wav > 1, {abs_max_wav}")
            wav /= abs_max_wav
        sf.write(wav_path, wav.detach().cpu().numpy(), sample_rate)

    pattern = '.'.join(wavname.split('.')[:-1]) + '{name}'  # remove .wav in wavname
    example_dir = os.path.join(exp_save_path, 'examples', str(paras[0]['index']))
    os.makedirs(example_dir, exist_ok=True)
    # save preds and targets for each speaker
    for i in range(yr.shape[1]):
        # write ys
        wav_path = os.path.join(example_dir, pattern.format(name=f"_spk{i+1}.wav"))
        write_wav(wav_path=wav_path, wav=yr[0, i])
        # write ys_hat
        wav_path = os.path.join(example_dir, pattern.format(name=f"_spk{i+1}_p.wav"))
        write_wav(wav_path=wav_path, wav=yr_hat[0, i])  #, norm_to=ys[0, i].abs().max())
    # write mix
    wav_path = os.path.join(example_dir, pattern.format(name=f"_mix.wav"))
    write_wav(wav_path=wav_path, wav=xr[0, :])

    # write paras & results
    f = open(os.path.join(example_dir, pattern.format(name=f"_paras.json")), 'w', encoding='utf-8')
    paras[0]['metrics'] = result_dict
    json.dump(paras[0], f, indent=4, cls=MyJsonEncoder)
    f.close()
