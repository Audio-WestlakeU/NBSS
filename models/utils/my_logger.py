from typing import Dict, Optional
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


class MyLogger(TensorBoardLogger):

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            _my_step = step
            if k.startswith('val/'):  # use epoch for val metrics
                _my_step = int(metrics['epoch'])
            super().log_metrics(metrics={k: v}, step=_my_step)
