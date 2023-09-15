from pathlib import Path
from typing import *
import torch


def ensemble(opts: Union[int, str, List[str]], ckpt: str) -> Tuple[List[str], Dict]:
    """ensemble checkpoints

    Args:
        opts: ensemble last N epochs if opts is int; ensemble globed checkpoints if opts is str; ensemble specified checkpoints if opts is a list.
        ckpt: the current checkpoint path

    Returns:
        ckpts: the checkpoints ensembled
        state_dict
    """
    # parse the ensemble args to obtains the ckpts to ensemble
    if isinstance(opts, int):
        ckpts = []
        if opts > 0:
            epoch = int(Path(ckpt).name.split('_')[0].replace('epoch', ''))
            for epc in range(max(0, epoch - opts), epoch, 1):
                path = list(Path(ckpt).parent.glob(f'epoch{epc}_*'))[0]
                ckpts.append(path)
    elif isinstance(opts, list):
        assert len(opts) > 0, opts
        ckpts = list(set(opts))
    else:  # e.g. logs/SSFNetLM/version_100/checkpoints/epoch* or epoch*
        assert isinstance(opts, str), opts
        ckpts = list(Path(opts).parent.glob(Path(opts).name))
        if len(ckpts) == 0:
            ckpts = list(Path(ckpt).parent.glob(opts))
        assert len(ckpts) > 0, f"checkpoints not found in {opts} or {Path(ckpt).parent/opts}"
    ckpts = ckpts + [ckpt]

    # remove redundant ckpt
    ckpts_ = dict()
    for ckpt in ckpts:
        ckpts_[Path(ckpt).name] = str(ckpt)
    ckpts = list(ckpts_.values())
    ckpts.sort()

    # load weights from checkpoints
    state_dict = dict()
    for path in ckpts:
        data = torch.load(path, map_location='cpu')
        for k, v in data['state_dict'].items():
            if k in state_dict:
                state_dict[k] += (v / len(ckpts))
            else:
                state_dict[k] = (v / len(ckpts))
    return ckpts, state_dict
