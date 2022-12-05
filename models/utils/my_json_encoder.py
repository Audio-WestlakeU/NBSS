import json
import numpy as np
from torch import Tensor


class MyJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.size==1:
                return obj.item()
            else:
                return obj.tolist()
        elif isinstance(obj, Tensor):
            if obj.numel()==1:
                return obj.item()
            else:
                return obj.detach().cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)
