"""
Command Line Interface for NBSS, provides command line controls for training, test, and inference
"""

import os

os.environ["OMP_NUM_THREADS"] = str(8)  # limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine

import torch
from jsonargparse import lazy_instance
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI

from data_loaders import SS_SemiOnlineDataModule
from models.arch.blstm2_fc1 import BLSTM2_FC1
from models.io.narrow_band.td_signal_nb import TimeDomainSignalNB
from models.NBSS import NBSS
from models.utils import MyRichProgressBar as RichProgressBar
# from pytorch_lightning.loggers import TensorBoardLogger
from models.utils.my_logger import MyLogger as TensorBoardLogger


class NBSSCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.set_defaults({
            "data.clean_speech_dataset": "wsj0-mix",
            "data.clean_speech_dir": "/dev/shm/quancs/",
            "data.rir_dir": "/dev/shm/quancs/rir_cfg_4",
            # "trainer.benchmark": True,
            "model.arch": lazy_instance(BLSTM2_FC1, hidden_size=[256, 128], activation="", input_size=16, output_size=4),
            "model.io": lazy_instance(TimeDomainSignalNB, ft_len=512, ft_overlap=256),
        })

        # link args
        parser.link_arguments("model.channels", "model.arch.init_args.input_size", compute_fn=lambda channels: 2 * len(channels), apply_on="parse")
        import importlib
        parser.link_arguments(
            ("data.speaker_num", "model.io.class_path"),
            "model.arch.init_args.output_size",
            compute_fn=lambda spk_num, class_path: spk_num * getattr(importlib.import_module('.'.join(class_path.split('.')[:-1])),
                                                                     class_path.split('.')[-1]).size_per_spk,
            apply_on="parse",
        )  # when parse config file
        parser.link_arguments(
            ("model.channels", "model.ref_channel"),
            "model.io.init_args.ref_chn_idx",
            compute_fn=lambda channels, ref_channel: channels.index(ref_channel),
            apply_on="parse",
        )  # when parse config file
        parser.link_arguments("data.speaker_num", "model.io.init_args.spk_num", apply_on="parse")  # when parse config file
        parser.link_arguments("data.speaker_num", "model.speaker_num", apply_on="parse")  # when parse config file
        # link functions
        parser.link_arguments("model.collate_func_train", "data.collate_func_train", apply_on="instantiate")  # after instantiate model
        parser.link_arguments("model.collate_func_val", "data.collate_func_val", apply_on="instantiate")  # after instantiate model
        parser.link_arguments("model.collate_func_test", "data.collate_func_test", apply_on="instantiate")  # after instantiate model

        # RichProgressBar
        parser.add_lightning_class_args(RichProgressBar, nested_key='progress_bar')
        parser.set_defaults({
            "progress_bar.refresh_rate_per_second": 1,
        })

        # EarlyStopping
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        early_stopping_defaults = {
            "early_stopping.monitor": "val/neg_si_sdr",
            "early_stopping.patience": 30,
            "early_stopping.mode": "min",
            "early_stopping.min_delta": 0.01,
        }
        parser.set_defaults(early_stopping_defaults)

        # ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_neg_si_sdr{val/neg_si_sdr:.4f}",
            "model_checkpoint.monitor": "val/neg_si_sdr",
            "model_checkpoint.mode": "min",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 5,
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        # LearningRateMonitor
        parser.add_lightning_class_args(LearningRateMonitor, "learning_rate_monitor")
        learning_rate_monitor_defaults = {
            "learning_rate_monitor.logging_interval": "epoch",
        }
        parser.set_defaults(learning_rate_monitor_defaults)

        # ModelSummary
        parser.add_lightning_class_args(ModelSummary, 'model_summary')
        model_summary_defaults = {
            "model_summary.max_depth": -1,
        }
        parser.set_defaults(model_summary_defaults)

        return super().add_arguments_to_parser(parser)

    def before_fit(self):
        resume_from_checkpoint: str = self.config['fit']['trainer']["resume_from_checkpoint"] or self.config['fit']['ckpt_path']
        if resume_from_checkpoint is not None and resume_from_checkpoint.endswith('last.ckpt'):
            # log in same dir
            # resume_from_checkpoint example: /mnt/home/quancs/projects/NBSS_pmt/logs/NBSS_ifp/version_29/checkpoints/last.ckpt
            resume_from_checkpoint = os.path.normpath(resume_from_checkpoint)
            splits = resume_from_checkpoint.split(os.path.sep)
            version = int(splits[-3].replace('version_', ''))
            save_dir = os.path.sep.join(splits[:-3])
            self.trainer.logger = TensorBoardLogger(save_dir=save_dir, name="", version=version, default_hp_metric=False)
        else:
            model_name = str(self.model_class).split('\'')[1].split('.')[-1]
            self.trainer.logger = TensorBoardLogger('logs/', name=model_name, default_hp_metric=False)

    def after_fit(self):
        if self.trainer.limit_test_batches is not None and self.trainer.limit_test_batches <= 0:
            return
        # test
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)
        resume_from_checkpoint = self.trainer.checkpoint_callback.best_model_path
        if resume_from_checkpoint is None or resume_from_checkpoint == "":
            if self.trainer.is_global_zero:
                print("no checkpoint found, so test is ignored")
            return
        epoch = os.path.basename(resume_from_checkpoint).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(resume_from_checkpoint))
        exp_save_path = os.path.normpath(write_dir + '/' + epoch + '_' + self.config['fit']['data']["test_set"] + '_set')

        # comment the following code to disable the test after fit
        import torch.distributed as dist
        if self.trainer.is_global_zero:
            self.trainer.logger = TensorBoardLogger(exp_save_path, name="", default_hp_metric=False)
            versions = [self.trainer.logger.version]
        else:
            versions = [None]
        if self.trainer.world_size > 1:
            dist.broadcast_object_list(versions)
            self.trainer.logger = TensorBoardLogger(exp_save_path, name="", version=versions[0], default_hp_metric=False)
        self.trainer.test(ckpt_path=resume_from_checkpoint, datamodule=self.datamodule)
        self.after_test()

    def before_test(self):
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)
        if self.config['test']['ckpt_path'] != None:
            ckpt_path = self.config['test']['ckpt_path']
        else:
            raise Exception('You should give --ckpt_path if you want to test')
        epoch = os.path.basename(ckpt_path).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(ckpt_path))
        exp_save_path = os.path.normpath(write_dir + '/' + epoch + '_' + self.config['test']['data']["test_set"] + '_set')

        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)

    def after_test(self):
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for test is removed: ' + self.trainer.log_dir + '/' + f)


if __name__ == '__main__':
    cli = NBSSCLI(NBSS, SS_SemiOnlineDataModule, seed_everything_default=None, save_config_overwrite=True)
