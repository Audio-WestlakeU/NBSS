import json
import numpy as np
from torch import Tensor


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Tensor):
            return obj.detach().cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)
