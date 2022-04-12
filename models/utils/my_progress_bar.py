from pytorch_lightning.callbacks.progress import TQDMProgressBar
import sys
from torch import Tensor
from pytorch_lightning import Trainer


class MyProgressBar(TQDMProgressBar):
    """print out the metrics on_validation_epoch_end
    """

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        sys.stdout.flush()
        if trainer.is_global_zero:
            metrics = trainer.logged_metrics
            infos = f"\x1B[1A\x1B[K\nEpoch {trainer.current_epoch} metrics: "
            for k, v in metrics.items():
                value = v
                if isinstance(v, Tensor):
                    value = v.item()
                if isinstance(value, float):
                    infos += k + f"={value:.4f}  "
                else:
                    infos += k + f"={value}  "
            if len(metrics) > 0:
                sys.stdout.write(f'{infos}\x1B[K\n')
            sys.stdout.flush()
