import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class BLSTM2_FC1(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            activation: Optional[str] = "",
            hidden_size: Tuple[int, int] = (256, 128),
            n_repeat_last_lstm: int = 1,
            dropout: Optional[float] = None,
    ):
        """Two layers of BiLSTMs & one fully connected layer

        Args:
            input_size: the input size for the features of the first BiLSTM layer
            output_size: the output size for the features of the last BiLSTM layer
            hidden_size: the hidden size of each BiLSTM layer. Defaults to (256, 128).
        """

        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout = dropout

        self.blstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0], batch_first=True, bidirectional=True)  # type:ignore
        self.blstm2 = nn.LSTM(input_size=self.hidden_size[0] * 2, hidden_size=self.hidden_size[1], batch_first=True, bidirectional=True, num_layers=n_repeat_last_lstm)  # type:ignore
        if dropout is not None:
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)

        self.linear = nn.Linear(self.hidden_size[1] * 2, self.output_size)  # type:ignore
        if self.activation is not None and len(self.activation) > 0:  # type:ignore
            self.activation_func = getattr(nn, self.activation)()  # type:ignore
        else:
            self.activation_func = None

    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x: shape [batch, seq, input_size]

        Returns:
            Tensor: shape [batch, seq, output_size]
        """
        x, _ = self.blstm1(x)
        if self.dropout:
            x = self.dropout1(x)
        x, _ = self.blstm2(x)
        if self.dropout:
            x = self.dropout2(x)
        if self.activation_func is not None:
            y = self.activation_func(self.linear(x))
        else:
            y = self.linear(x)

        return y