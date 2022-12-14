import os
import numpy as np
from typing import Optional, Tuple, List, Union, Callable
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


if __name__ == '__main__':
    filepath = input("Enter filepath to load data: ")
    images, poses, focal = load_data(filepath)
    origins, dirs = convert_poses(poses)
    
    height, width = images.shape[1:3]
    rays_o, rays_d = get_rays(height, width, focal, torch.from_numpy(poses)[0])
    print(rays_o.shape)
    print(rays_d.shape)