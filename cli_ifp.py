import os
# limit the threads
os.environ["OMP_NUM_THREADS"] = str(15)

from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data_loaders import SS_SemiOnlineDataModule
from models.NBSS_ifp import NBSS_ifp
from jsonargparse import lazy_instance
import torch

class MyLightningCLI(LightningCLI):

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

        # EarlyStopping
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        early_stopping_defaults = {
            "early_stopping.monitor": "val/neg_si_sdr",
            "early_stopping.patience": 30,
            "early_stopping.mode": "min",
            "early_stopping.min_delta": 0.0001,
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
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)
        resume_from_checkpoint = self.trainer.checkpoint_callback.best_model_path
        epoch = os.path.basename(resume_from_checkpoint).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(resume_from_checkpoint))
        exp_save_path = os.path.normpath(write_dir + '/' + epoch + '_' + self.config['fit']['data']["test_set"] + '_set')

        self.trainer.logger = TensorBoardLogger(exp_save_path, name="", default_hp_metric=False)
        self.trainer.test(ckpt_path=resume_from_checkpoint, datamodule=self.datamodule)

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


cli = MyLightningCLI(NBSS_ifp, SS_SemiOnlineDataModule, seed_everything_default=None, save_config_overwrite=True)
