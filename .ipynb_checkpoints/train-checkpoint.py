import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from models import *
from preprocessing import *
from forward import *
from rendering import *

PATH = 'trained_models/vanilla_nerf_1000.pt'

def crop_center(img, frac= 0.5):
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))
    return img[h_offset:-h_offset, w_offset:-w_offset]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    filepath = input("Enter filepath to load data: ")
    images, poses, focal = load_data(filepath)

    height, width = images.shape[1:3]
    near, far = 2., 6.

    n_training = 100
    testimg_idx = 101

    testimg = torch.from_numpy(images[testimg_idx]).to(device)
    testpose = torch.from_numpy(poses[testimg_idx]).to(device)
    poses = torch.from_numpy(poses).to(device)
    focal = torch.from_numpy(focal).to(device)
    images = torch.from_numpy(images[:n_training]).to(device)

    # Encoders
    d_input = 3           # Number of input dimensions
    n_freqs = 10          # Number of encoding functions for samples
    log_space = True      # If set, frequencies scale in log space
    use_viewdirs = True   # If set, use view direction as input
    n_freqs_views = 4     # Number of encoding functions for views

    # Stratified sampling
    n_samples = 64         # Number of spatial samples per ray
    perturb = True         # If set, applies noise to sample positions
    inverse_depth = False  # If set, samples points linearly in inverse depth

    # Model
    d_filter = 128          # Dimensions of linear layer filters
    n_layers = 2            # Number of layers in network bottleneck
    skip = []               # Layers at which to apply input residual
    use_fine_model = True   # If set, creates a fine model

    # Hierarchical sampling
    n_samples_hierarchical = 64   # Number of samples per ray

    # Optimizer
    lr = 5e-4  # Learning rate

    # Training
    n_iters = 1000
    batch_size = 2**14          # Number of rays per gradient step (power of 2)
    one_image_per_step = True   # One image per gradient step (disables batching)
    chunksize = 2**14           # Modify as needed to fit in GPU memory
    center_crop = True          # Crop the center of image (one_image_per_)
    center_crop_iters = 50      # Stop cropping center after this many epochs
    display_rate = 25          # Display test output every X epochs

    kwargs_sample_stratified = {
        'n_samples': n_samples,
        'perturb': perturb,
        'inverse_depth': inverse_depth
    }
    kwargs_sample_hierarchical = {
        'perturb': perturb
    }

    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    encode = lambda x: encoder(x)

    # View direction encoders
    if use_viewdirs:
        encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,
                                            log_space=log_space)
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models
    model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
                d_viewdirs=d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())
    if use_fine_model:
        fine_model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
                            d_viewdirs=d_viewdirs)
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    # Optimizer
    optimizer = torch.optim.Adam(model_params, lr=lr)

    # Shuffle rays across all images.
    if not one_image_per_step:
        height, width = images.shape[1:3]
        all_rays = torch.stack([torch.stack(get_rays(height, width, focal, p), 0)
                            for p in poses[:n_training]], 0)
        rays_rgb = torch.cat([all_rays, images[:, None]], 1)
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
        i_batch = 0

    train_psnrs = []
    val_psnrs = []
    iternums = []
    for i in trange(n_iters):
        model.train()

        if one_image_per_step:
            # Randomly pick an image as the target.
            target_img_idx = np.random.randint(images.shape[0])
            target_img = images[target_img_idx].to(device)
            if center_crop and i < center_crop_iters:
                target_img = crop_center(target_img)
            height, width = target_img.shape[:2]
            target_pose = poses[target_img_idx].to(device)
            rays_o, rays_d = get_rays(height, width, focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
        else:
            # Random over all images.
            batch = rays_rgb[i_batch:i_batch + batch_size]
            batch = torch.transpose(batch, 0, 1)
            rays_o, rays_d, target_img = batch
            height, width = target_img.shape[:2]
            i_batch += batch_size
            # Shuffle after one epoch
            if i_batch >= rays_rgb.shape[0]:
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                i_batch = 0
        target_img = target_img.reshape([-1, 3])

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        outputs = nerf_forward(rays_o, rays_d,
                                near, far, encode, model,
                                kwargs_sample_stratified=kwargs_sample_stratified,
                                n_samples_hierarchical=n_samples_hierarchical,
                                kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                                fine_model=fine_model,
                                viewdirs_encoding_fn=encode_viewdirs,
                                chunksize=chunksize)

        rgb_predicted = outputs['rgb_map']
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Compute mean-squared error between predicted and target images.
        psnr = -10. * torch.log10(loss)
        train_psnrs.append(psnr.item())

        # Evaluate testimg at given display rate.
        if i % display_rate == 0:
            model.eval()
            height, width = testimg.shape[:2]
            rays_o, rays_d = get_rays(height, width, focal, testpose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            outputs = nerf_forward(rays_o, rays_d,
                                    near, far, encode, model,
                                    kwargs_sample_stratified=kwargs_sample_stratified,
                                    n_samples_hierarchical=n_samples_hierarchical,
                                    kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                                    fine_model=fine_model,
                                    viewdirs_encoding_fn=encode_viewdirs,
                                    chunksize=chunksize)

            rgb_predicted = outputs['rgb_map']
            loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
            print("Loss:", loss.item())
            val_psnr = -10. * torch.log10(loss)
            
            val_psnrs.append(val_psnr.item())
            iternums.append(i)

    torch.save(model.state_dict(), PATH)
    torch.save(train_psnrs, 'trained_models/train_psnrs_1000.pt')
    torch.save(val_psnrs, 'trained_models/val_psnrs_1000.pt')

if __name__ == '__main__':
    main()
