from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor


def default_collate_func(batches: List[Tuple[Tensor, Tensor, Dict[str, Any]]]) -> List[Any]:
    mini_batch = []
    for x in zip(*batches):
        if isinstance(x[0], np.ndarray):
            x = [torch.tensor(x[i]) for i in range(len(x))]
        if isinstance(x[0], Tensor):
            x = torch.stack(x)
        mini_batch.append(x)
    return mini_batch
