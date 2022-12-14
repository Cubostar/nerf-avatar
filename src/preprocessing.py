import os
import numpy as np
from typing import Optional, Tuple
import torch



def load_data(
    filepath: str
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loads data.
    """
    data = np.load(filepath)
    images = data['images']
    poses = data['poses']
    focal = data['focal']

    return images, poses, focal



def convert_poses(
    poses: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts poses to positions and viewing directions.
    """
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
    return origins, dirs



def get_rays(
  height: int,
  width: int,
  focal_length: float,
  pose: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Find origin and direction of rays through each pixel from the camera origin.
  Inputs are:
  - height (int) : height of image
  - width (int) : width of image
  - focal_length (float) : focal length of camera
  - pose (torch.Tensor) : single camera pose
  Outputs are:
  - rays_o (torch.Tensor) : origin of each ray for each pixel and camera origin
  - rays_d (torch.Tensor) : direction of each ray for each pixel and camera origin
  """

  # Apply pinhole camera model to gather directions at each pixel
  i, j = torch.meshgrid(
      torch.arange(width, dtype=torch.float32).to(pose),
      torch.arange(height, dtype=torch.float32).to(pose),
      indexing='ij')
  i, j = i.transpose(-1, -2), j.transpose(-1, -2)
  directions = torch.stack([(i - width * .5) / focal_length,
                            -(j - height * .5) / focal_length,
                            -torch.ones_like(i)
                           ], dim=-1)

  # Apply camera pose to directions
  rays_d = torch.sum(directions[..., None, :] * pose[:3, :3], dim=-1)

  # Origin is same for all directions (the optical center)
  rays_o = pose[:3, -1].expand(rays_d.shape)
  return rays_o, rays_d



def sample_stratified(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: Optional[bool] = True,
    inverse_depth: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample along rays from regularly-spaced bins.
    Inputs are:
    - rays_o (torch.Tensor) : points of origin for rays
    - rays_d (torch.Tensor) : directions for rays
    - near (float) : when to start sampling
    - far (float) : when to stop sampling
    - n_samples (int) : number of samples
    - perturb (Optional[bool] = True) : sample randomly from each bin if True or have points evenly spaced
    - inverse_depth (bool) : take inverse of depth
    Outputs are:
    - pts (torch.Tensor) : points along each ray
    - zvals (torch.Tensor) : z-values for points as proportions between `near` and `far`
    """
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between `near` and `far`
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    
    # Draw uniform samples from bins along ray
    if perturb:
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    # Apply scale from `rays_d` and offset from `rays_o` to samples
    # pts: (width, height, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

    

if __name__ == '__main__':
    filepath = input("Enter filepath to load data: ")
    images, poses, focal = load_data(filepath)
    origins, dirs = convert_poses(poses)
    
    height, width = images.shape[1:3]
    rays_o, rays_d = get_rays(height, width, focal, torch.from_numpy(poses)[0])
    
    rays_o = rays_o.view([-1, 3])
    rays_d = rays_d.view([-1, 3])

    pts, z_vals = sample_stratified(rays_o, rays_d, 0.2, 0.6, 8)
    print(pts.shape)
    print(z_vals.shape)