import sys

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *
from rich.console import Console
from torch import Tensor


class MyRichProgressBar(RichProgressBar):
    """A progress bar prints metrics at the end of each epoch
    """

    def _init_progress(self, trainer):
        if pl.__version__.startswith('1.5.'):
            if self.is_enabled and (self.progress is None or self._progress_stopped):
                self._reset_progress_bar_ids()
                self._console: Console = Console(force_terminal=True, no_color=True, width=200)
                self._console.clear_live()
                self._metric_component = MetricsTextColumn(trainer, self.theme.metrics)
                self.progress = CustomProgress(
                    *self.configure_columns(trainer),
                    self._metric_component,
                    refresh_per_second=self.refresh_rate_per_second,
                    disable=self.is_disabled,
                    console=self._console,
                )
                self.progress.start()
                # progress has started
                self._progress_stopped = False
        else:
            RichProgressBar._init_progress(self, trainer)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        sys.stdout.flush()
        if trainer.is_global_zero:
            metrics = trainer.logged_metrics
            infos = f"Epoch {trainer.current_epoch} metrics: "
            for k, v in metrics.items():
                if k.startswith('train/'):
                    continue
                value = v
                if isinstance(v, Tensor):
                    value = v.item()
                if isinstance(value, float):
                    infos += k + f"={value:.4f}  "
                else:
                    infos += k + f"={value}  "
            if len(metrics) > 0:
                sys.stdout.write(f'{infos}\n')
            sys.stdout.flush()
