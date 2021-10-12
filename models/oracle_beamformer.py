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
from beamformers import beamformers
from data_loaders import SS_SemiOnlineDataModule, SS_SemiOnlineDataset
from pandas.core.frame import DataFrame
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import (LightningArgumentParser, LightningCLI)
from torch import Tensor
from torchmetrics.audio.utils import cal_metrics_functional
from torchmetrics.functional.audio import pit, pit_permutate, si_sdr

COMMON_AUDIO_METRICS = ['SDR', 'SI_SDR', 'SI_SNR', 'NB_PESQ', 'WB_PESQ']


# oracle beamformer
class OracleBeamformer(pl.LightningModule):

    def __init__(self, speaker_num: int = 2, ref_channel: int = 0, beamformer: str = 'MVDR', give_target: bool = False, exp_name: str = 'exp'):
        super().__init__()

        # save all the hyperparameters to the checkpoint
        self.save_hyperparameters()
        # self.ref_chn_idx = self.hparams.channels.index(self.hparams.ref_channel)

        self.bf = getattr(beamformers, beamformer)
        print(f"using {beamformer} beamformer")

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
                pred = beamformers.MVDR(mixture=x.numpy(), noise=noise.numpy(), target=target.numpy(), ref_mic=self.ref_channel)
            else:
                pred = beamformers.MVDR(mixture=x.numpy(), noise=noise.numpy())
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
            abs_max = max(torch.max(torch.abs(x[0,])), torch.max(torch.abs(ys_hat_perm[0,])), torch.max(torch.abs(ys[0,])))

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


cli = BeamformerCLI(OracleBeamformer, SS_SemiOnlineDataModule, seed_everything_default=None)
