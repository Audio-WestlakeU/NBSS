import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


class MyEarlyStopping(EarlyStopping):

    def __init__(
        self,
        enable: bool = True, # enable EarlyStopping or not
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable = enable

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        if self.enable == False:
            return True
        else:
            return super()._should_skip_check(trainer)
