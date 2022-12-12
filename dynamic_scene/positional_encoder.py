import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """
    def __init__(
        self,
        d_input: int,
        n_freqs: int,
    ):
        super.__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x : x]

        freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(
        self,
        x
    ) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)