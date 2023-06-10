"""
Command Line Interface for NBSS, provides command line controls for training, test, and inference
"""
import os

os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # enable bf16 in pytorch 1.12, see https://github.com/Lightning-AI/lightning/issues/11933#issuecomment-1181590004
os.environ["OMP_NUM_THREADS"] = str(8)  # limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine

import torch

torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from models.NBSS import NBSS
from models.utils import MyRichProgressBar as RichProgressBar
# from pytorch_lightning.loggers import TensorBoardLogger
from models.utils.my_logger import MyLogger as TensorBoardLogger
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback


class NBSSCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # link args
        parser.link_arguments("model.channels", "model.arch.init_args.input_size", compute_fn=lambda channels: 2 * len(channels), apply_on="parse")
        import importlib
        parser.link_arguments(
            ("model.speaker_num", "model.io.class_path"),
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
        parser.link_arguments("model.speaker_num", "model.io.init_args.spk_num", apply_on="parse")  # when parse config file
        # link functions
        parser.link_arguments("model.collate_func_train", "data.init_args.collate_func_train", apply_on="instantiate")  # after instantiate model
        parser.link_arguments("model.collate_func_val", "data.init_args.collate_func_val", apply_on="instantiate")  # after instantiate model
        parser.link_arguments("model.collate_func_test", "data.init_args.collate_func_test", apply_on="instantiate")  # after instantiate model

        self.add_model_invariant_arguments_to_parser(parser)

    def add_model_invariant_arguments_to_parser(self, parser) -> None:
        # RichProgressBar
        parser.add_lightning_class_args(RichProgressBar, nested_key='progress_bar')
        parser.set_defaults({"progress_bar.console_kwargs": {
            "force_terminal": True,
            "no_color": True,
            "width": 200,
        }})

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
        # parser.add_lightning_class_args(ModelSummary, 'model_summary')
        # model_summary_defaults = {
        #     "model_summary.max_depth": 1,
        # }
        # parser.set_defaults(model_summary_defaults)

    def before_fit(self):
        resume_from_checkpoint: str = self.config['fit']['ckpt_path']
        if resume_from_checkpoint is not None and resume_from_checkpoint.endswith('last.ckpt'):
            # log in same dir
            # ckpt_path example: /mnt/home/quancs/projects/NBSS_pmt/logs/NBSS_ifp/version_29/checkpoints/last.ckpt
            resume_from_checkpoint = os.path.normpath(resume_from_checkpoint)
            splits = resume_from_checkpoint.split(os.path.sep)
            version = int(splits[-3].replace('version_', ''))
            save_dir = os.path.sep.join(splits[:-3])
            self.trainer.logger = TensorBoardLogger(save_dir=save_dir, name="", version=version, default_hp_metric=False)
        else:
            model_name = str(self.model_class).split('\'')[1].split('.')[-1]
            self.trainer.logger = TensorBoardLogger('logs/', name=model_name, default_hp_metric=False)

    def before_test(self):
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)
        if self.config['test']['ckpt_path'] is not None:
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

        import time
        # add 10 seconds for threads to simultaneously detect the next version
        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)
        time.sleep(10)

    def after_test(self):
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for test is removed: ' + self.trainer.log_dir + '/' + f)


if __name__ == '__main__':
    cli = NBSSCLI(
        NBSS,
        pl.LightningDataModule,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        subclass_mode_data=True,
    )
