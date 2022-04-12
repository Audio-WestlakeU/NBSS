from abc import abstractmethod
from typing import Any, Callable, Dict, Tuple

import torch
from torch import Tensor


class NBIO(torch.nn.Module):
    """Narrow Band Speech Separation Input, Output, and loss
    """

    size_per_spk: int  # feature size per spk per time frame
    loss_name: str  # name of loss function

    def __init__(self, ft_len: int, ft_overlap: int, ref_chn_idx: int, spk_num: int, loss_func: Callable, loss_name=None) -> None:
        """init

        Args:
            ft_len: for STFT
            ft_overlap: for STFT
            ref_chn_idx: the index of the reference channel
            spk_num: the speaker num
            loss_func: loss function
            loss_name: the name of loss function, if is None, the name of `loss_func` will be used
        """
        super().__init__()
        self.ft_len = ft_len
        self.ft_overlap = ft_overlap
        self.ref_chn_idx = ref_chn_idx
        self.loss_func = loss_func
        self.spk_num = spk_num

        self.windows: Dict[str, Tensor] = dict()
        self.window = torch.hann_window(ft_len)
        self.windows[str(self.window.device)] = self.window

        if loss_name is None:
            self.loss_name = loss_func.__name__
        else:
            self.loss_name = loss_name

    def _get_window(self, device) -> Tensor:
        """return the STFT window for device

        Args:
            device: the device of the window needed

        Returns:
            window at device
        """
        if str(device) not in self.windows:
            self.windows[str(device)] = self.window.clone().to(device)
        return self.windows[str(device)]

    @abstractmethod
    def prepare_input(self, x: Tensor, *args, **kwargs) -> Any:
        """prepare input for network

        Args:
            x: time domain mixture signal, shape [batch, channel, time]
        
        returns:
            If multiple items are returned, the first item will be regarded as the input of network
        """

        pass

    @abstractmethod
    def prepare_target(self, ys: Tensor, input: Any, *args, **kwargs) -> Tensor:
        """prepare target for loss function

        Args:
            ys: time domain target signal, shape [batch, speaker, time]
            input: the output of prepare_input
        returns:
            the target for loss function
        """
        pass

    @abstractmethod
    def prepare_prediction(self, o: Tensor, input: Any, *args, **kwargs) -> Tensor:
        """prepare prediction from the output of network for loss function

        Args:
            o: raw output from network
            input: the output of prepare_input

        returns:
            the prediction for loss function
        """
        pass

    @abstractmethod
    def prepare_time_domain(self, o: Tensor, input: Any, preds: Tensor, *args, **kwargs) -> Tensor:
        """prepare time domain prediction

        Args:
            o: raw output from network
            input: the output of prepare_input
            preds: the output of prepare_prediction

        returns:
            the time domain prediction
        """
        pass

    @abstractmethod
    def loss(self, preds: Tensor, target: Tensor, reduce_batch: bool = True, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """loss for preds and target

        Args:
            preds: prediction
            target: target

        Returns:
            loss value(s), shape [batch] if reduce_batch==False, else a single value
            perms: returned by pit
        """
        pass
