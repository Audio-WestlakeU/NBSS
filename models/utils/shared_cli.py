"""
Command Line Interface for different models, provides command line controls for training, test, and inference
"""
import os

os.environ["OMP_NUM_THREADS"] = str(1)  # limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine

import torch

torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from jsonargparse import lazy_instance

from models.utils import MyRichProgressBar as RichProgressBar
# from pytorch_lightning.loggers import TensorBoardLogger
from models.utils.my_logger import MyLogger as TensorBoardLogger
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback


class SharedCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.set_defaults({"trainer.strategy": "ddp_find_unused_parameters_false"})

        # RichProgressBar
        parser.add_lightning_class_args(RichProgressBar, nested_key='progress_bar')
        if pl.__version__.startswith('1.5.'):
            parser.set_defaults({
                "progress_bar.refresh_rate_per_second": 1,
            })
        else:
            parser.set_defaults({"progress_bar.console_kwargs": {
                "force_terminal": True,
                "no_color": True,
                "width": 200,
            }})

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

    def before_fit(self):
        resume_from_checkpoint: str = self.config['fit']['ckpt_path']
        if resume_from_checkpoint is not None and resume_from_checkpoint.endswith('last.ckpt'):
            # log in same dir
            # resume_from_checkpoint example: /mnt/home/quancs/projects/NBSS_pmt/logs/NBSS_ifp/version_29/checkpoints/last.ckpt
            resume_from_checkpoint = os.path.normpath(resume_from_checkpoint)
            splits = resume_from_checkpoint.split(os.path.sep)
            version = int(splits[-3].replace('version_', ''))
            save_dir = os.path.sep.join(splits[:-3])
            self.trainer.logger = TensorBoardLogger(save_dir=save_dir, name="", version=version, default_hp_metric=False)
        else:
            model_name = type(self.model).__name__
            self.trainer.logger = TensorBoardLogger('logs/', name=model_name, default_hp_metric=False)

    def before_test(self):
        if self.config['test']['ckpt_path'] != None:
            ckpt_path = self.config['test']['ckpt_path']
        else:
            raise Exception('You should give --ckpt_path if you want to test')
        epoch = os.path.basename(ckpt_path).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(ckpt_path))

        test_set = 'test'
        if 'test_set' in self.config['test']['data']:
            test_set = self.config['test']['data']["test_set"]
        elif 'init_args' in self.config['test']['data'] and 'test_set' in self.config['test']['data']['init_args']:
            test_set = self.config['test']['data']['init_args']["test_set"]
        exp_save_path = os.path.normpath(write_dir + '/' + epoch + '_' + test_set + '_set')

        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)
        print(self.trainer.log_dir)

    def after_test(self):
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for test is removed: ' + self.trainer.log_dir + '/' + f)


if __name__ == '__main__':
    cli = SharedCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        subclass_mode_data=True,
        subclass_mode_model=False,
        run=True,
    )
