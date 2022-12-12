import torch
import torch.nn as nn
from positional_encoder import PositionalEncoder

class NSFF(nn.Module):
    r"""
    Neural scene flow fields module.
    """
    def __init__(
        self,
        d_input: int,
        n_layers: int = 9,
        n_channels: int = 256,
        skip: tuple[int] = (4,)
    ):
        super.__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, n_channels)] +
            [nn.linear(n_channels + self.d_input, n_channels) if i in skip \
                else nn.Linear(n_channels, n_channels) for i in range(n_layers - 1)]
        )

        