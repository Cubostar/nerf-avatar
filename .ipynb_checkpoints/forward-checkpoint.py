import torch
from torch import nn
from typing import Optional, Tuple, List, Callable

from preprocessing import *
from rendering import *

def get_chunks(inputs,chunksize= 2**15):
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def prepare_chunks(points, encoding_function, chunksize= 2**15):
    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points

def prepare_viewdirs_chunks(points, rays_d, encoding_function, chunksize= 2**15):
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs

def nerf_forward(rays_o, rays_d, near, far, encoding_fn, coarse_model, kwargs_sample_stratified = None,n_samples_hierarchical = 0, kwargs_sample_hierarchical = None, fine_model = None, viewdirs_encoding_fn = None,chunksize = 2**15
):

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}
  
    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, near, far, **kwargs_sample_stratified)

    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                    viewdirs_encoding_fn,
                                                    chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
    # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = {
        'z_vals_stratified': z_vals
    }

    # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = sample_hierarchical(
            rays_o, rays_d, z_vals, weights, n_samples_hierarchical,
            **kwargs_sample_hierarchical)

        # Prepare inputs as before.
        batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                        viewdirs_encoding_fn,
                                                        chunksize=chunksize)
        else:
            batches_viewdirs = [None] * len(batches)

        # Forward pass new samples through fine model.
        fine_model = fine_model if fine_model is not None else coarse_model
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)

        # Store outputs.
        outputs['z_vals_hierarchical'] = z_hierarch
        outputs['rgb_map_0'] = rgb_map_0
        outputs['depth_map_0'] = depth_map_0
        outputs['acc_map_0'] = acc_map_0

    # Store outputs.
    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    return outputs