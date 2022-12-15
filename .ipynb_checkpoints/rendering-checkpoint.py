import torch
from torch import nn
from typing import Tuple


def raw2outputs(batched, z_points, rays_d):
    up_to_last = batched[..., 3]
    sigma = torch.sigmoid(up_to_last)
    rgb_values = torch.sigmoid(batched[..., :3])
    
    back_to_front = z_points[..., 1:] - z_points[..., :-1]
    broadcasted_vals = torch.broadcast_to(torch.tensor([1e10]), z_points[...,:1].size())

    distances = torch.cat([back_to_front, broadcasted_vals], -1)
    distances = distances * torch.norm(rays_d[..., None, :], dim=-1)

    alpha = 1.0 - torch.exp(-1 * sigma * distances)
    inverse_alphas = 1.0 - alpha + 1e-10
    weights = torch.cumprod(inverse_alphas, -1)
    weights = torch.roll(weights, 1, -1)
    weights[...,0] = 1.0
    weights = alpha * weights
#     print(rgb_values.shape)
    rgb_map = torch.sum(weights[...,None] * rgb_values, -2) 
    depth_map = torch.sum(weights * z_points, -1) 
    acc_map = torch.sum(weights, -1)
#     print(weights.shape)
#     print(alphas.shape)

    return rgb_map, depth_map, acc_map, weights



def sample_pdf(bins, weights, n_samples, perturb = False):

    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1) # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous() # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples # [n_rays, n_samples]

def sample_hierarchical(rays_o, rays_d, z_vals, weights, n_samples, perturb = False):
    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples,
                            perturb=perturb)
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]
    return pts, z_vals_combined, new_z_samples