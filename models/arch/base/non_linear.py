import torch
from torch import Tensor, nn


class PReLU(nn.PReLU):

    def __init__(self, num_parameters: int = 1, init: float = 0.25, dim: int = 1, device=None, dtype=None) -> None:
        super().__init__(num_parameters, init, device, dtype)
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        if self.dim == 1:
            # [B, Chn, Feature]
            return super().forward(input)
        else:
            return super().forward(input.transpose(self.dim, 1)).transpose(self.dim, 1)


def new_non_linear(non_linear_type: str, dim_hidden: int, seq_last: bool) -> nn.Module:
    if non_linear_type.lower() == 'prelu':
        return PReLU(num_parameters=dim_hidden, dim=1 if seq_last == True else -1)
    elif non_linear_type.lower() == 'silu':
        return nn.SiLU()
    elif non_linear_type.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif non_linear_type.lower() == 'relu':
        return nn.ReLU()
    elif non_linear_type.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif non_linear_type.lower() == 'elu':
        return nn.ELU()
    else:
        raise Exception(non_linear_type)


if __name__ == '__main__':
    x = torch.rand(size=(12, 10, 100))
    prelu = new_non_linear('PReLU', 10, True)
    y = prelu(x)
    print(y.shape)
