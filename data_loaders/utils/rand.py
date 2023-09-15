import torch


def randint(g: torch.Generator, low: int, high: int) -> int:
    """return a value sampled in [low, high)
    """
    if low == high:
        return low
    r = torch.randint(low=low, high=high, size=(1,), generator=g, device='cpu')
    return r[0].item()  # type:ignore


def randfloat(g: torch.Generator, low: float, high: float) -> float:
    """return a value sampled in [low, high)
    """
    if low == high:
        return low
    r = torch.rand(size=(1,), generator=g, device='cpu')[0].item()
    return float(low + r * (high - low))
