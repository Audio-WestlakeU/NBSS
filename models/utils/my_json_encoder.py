import json
import numpy as np
from torch import Tensor
from pytorch_lightning.utilities.rank_zero import rank_zero_warn


class MyJsonEncoder(json.JSONEncoder):
    large_array_size: bool = 100
    ignore_large_array: bool = True

    def default(self, obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            if obj.size == 1:
                return obj.item()
            else:
                if obj.size > self.large_array_size:
                    if self.ignore_large_array:
                        rank_zero_warn('large array is ignored while saved to json file.')
                        return None
                    else:
                        rank_zero_warn('large array detected. saving it in json is slow. please remove it')
                return obj.tolist()
        elif isinstance(obj, Tensor):
            if obj.numel() == 1:
                return obj.item()
            else:
                if obj.numel() > self.large_array_size:
                    if self.ignore_large_array:
                        rank_zero_warn('large array is ignored while saved to json file.')
                        return None
                    else:
                        rank_zero_warn('large array detected. saving it in json is slow. please remove it')
                return obj.detach().cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)
