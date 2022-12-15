import torch
from torch import nn
from typing import Optional, Tuple
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self,d_input,n_freqs,log_space=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
    def forward(self,x):
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

class FourierEncoder(nn.Module):
    """
    Fourier positional encoder for input points.
    Inputs are:
    d_input (int) : dimension of input
    n_feats (int) : number of frequencies
    sigma (float) : standard deviation of random Fourier feature matrix
    Outputs are:
    features (torch.Tensor) : input but increased feature dimension from d_input to (2 * n_feats) + d_input,
    since there's sine and cosine, and then the original input.
    """
    def __init__(
        self,
        d_input: int,
        n_feats: int,
        sigma: float = 1.0
    ):
        super().__init__()
        self.d_input = d_input
        self.n_feats = n_feats
        self.d_output = d_input * (1 + 2 * self.n_feats)
        self.sigma = sigma

        # Create random Fourier feature matrix
        self.B = torch.from_numpy(np.random.normal(scale=sigma, size=(n_feats, d_input))).float()


    def forward(
        self,
        x
    ) -> torch.Tensor:
        """
        Apply positional encoding to input.
        """
        Bx = torch.concat([torch.t(torch.matmul(self.B, torch.t(x[i, None]))) for i in range(x.shape[0])])
        print(Bx.shape)
        return torch.concat([x, torch.cos(2*np.pi*Bx), torch.sin(2*np.pi*Bx)], dim=-1)

class NeRF(nn.Module):
    def __init__(
        self,
        d_input: int = 3,
        n_layers: int = 8,
        d_filter: int = 256,
        skip: Tuple[int] = (4,),
        d_viewdirs: Optional[int] = None
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        self.d_viewdirs = d_viewdirs

        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
            else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            self.output = nn.Linear(d_filter, 4)
  
    def forward(self, x, viewdirs = None):
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        if self.d_viewdirs is not None:
            alpha = self.alpha_out(x)

            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            x = torch.concat([x, alpha], dim=-1)
        else:
            x = self.output(x)
        return x
