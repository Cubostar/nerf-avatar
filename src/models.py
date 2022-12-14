import torch
from torch import nn
from typing import Optional, Tuple



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



class PositionalEncoder(nn.Module):
    """
    Sine-cosine positional encoder for input points.
    Inputs are:
    d_input (int) : dimension of input
    n_freqs (int) : number of frequencies
    Outputs are:
    features (torch.Tensor) : input but increased feature dimension from d_input to (2 * n_freqs) + d_input,
    since there's sine and cosine, and then the original input.
    """
    def __init__(
        self,
        d_input: int,
        n_freqs: int,
        log_space: bool = False
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
    def forward(
        self,
        x
    ) -> torch.Tensor:
        """
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)



class NeRF(nn.Module):
    """
    Neural radiance fields module.
    """
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

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
            else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, 4)
  
    def forward(
        self,
        x: torch.Tensor,
        viewdirs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional view direction.
        """

        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_out(x)

            # Pass through bottleneck to get RGB
            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)
        return x

if __name__ == '__main__':
    from preprocessing import *
    filepath = input("Enter filepath to load data: ")
    images, poses, focal = load_data(filepath)
    origins, dirs = convert_poses(poses)
    
    height, width = images.shape[1:3]
    rays_o, rays_d = get_rays(height, width, focal, torch.from_numpy(poses)[0])

    rays_o = rays_o.view([-1, 3])
    rays_d = rays_d.view([-1, 3])

    pts, z_vals = sample_stratified(rays_o, rays_d, 0.2, 0.6, 8)

    encoder = PositionalEncoder(3, 10)
    viewdirs_encoder = PositionalEncoder(3, 4)

    pts_flattened = pts.reshape(-1, 3)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    flattened_viewdirs = viewdirs[:, None, ...].expand(pts.shape).reshape((-1, 3))

    encoded_points = encoder(pts_flattened)
    encoded_viewdirs = viewdirs_encoder(flattened_viewdirs)

    print('Positionally Encoded Points')
    print(encoded_points.shape)
    print(torch.min(encoded_points), torch.max(encoded_points), torch.mean(encoded_points))
    print('')

    print(encoded_viewdirs.shape)
    print('Positionally Encoded Viewdirs')
    print(torch.min(encoded_viewdirs), torch.max(encoded_viewdirs), torch.mean(encoded_viewdirs))
    print('')

    ffencoder = FourierEncoder(3, 10)
    viewdirs_ffencoder = FourierEncoder(3, 4)

    ffencoded_points = ffencoder(pts_flattened)
    ffencoded_viewdirs = viewdirs_ffencoder(flattened_viewdirs)

    print('Random Fourier Encoded Points')
    print(ffencoded_points.shape)
    print(torch.min(ffencoded_points), torch.max(ffencoded_points), torch.mean(ffencoded_points))
    print('')

    print(encoded_viewdirs.shape)
    print('Random Fourier Encoded Viewdirs')
    print(torch.min(ffencoded_viewdirs), torch.max(ffencoded_viewdirs), torch.mean(ffencoded_viewdirs))
    print('')