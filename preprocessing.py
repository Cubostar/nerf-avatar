import os
import numpy as np
from typing import Optional, Tuple
import torch

def load_data(path):
    data = np.load(path)
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    return images, poses, focal

def convert_poses(poses):
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
    return origins, dirs

def get_rays(height, width, focal, camera_frames):
    i, j = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")
#     i, j = torch.meshgrid(torch.range(width), torch.range(height), indexing="ij")
    width_over_focal = (i-width/2.0)/focal
    height_over_focal = -1 * (j - height/ 2.0)/focal
    dirs = torch.stack([width_over_focal, height_over_focal, -1 * torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * camera_frames[:3,:3], -1)
    shape_of_rays_d = rays_d.size()
#     print(type(shape_of_rays_d))

#     print(shape_of_rays_d)
    from_third = torch.tensor(camera_frames[:3,-1])
#     print(type(from_third))
    rays_o = torch.broadcast_to(from_third, shape_of_rays_d)
    return rays_o, rays_d

def sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True, inverse_depth=False):
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    if not inverse_depth:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    
    if perturb:
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals