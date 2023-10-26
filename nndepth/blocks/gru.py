import torch
import torch.nn as nn


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 192 + 128):
        """Initialize the SepConvGRU module.

        Args:
            hidden_dim (int): The hidden dimension of the GRU. Default is 128.
            input_dim (int): The input dimension of the GRU. Default is 192 + 128.
        """
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 192 + 128):
        """Initialize the ConvGRU module.

        Args:
            hidden_dim (int): The hidden dimension of the GRU. Default is 128.
            input_dim (int): The input dimension of the GRU. Default is 192 + 128.
        """
        super(ConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h
