import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from models import *
from preprocessing import *
from forward import *
from rendering import *


def plot_samples(
    z_vals: torch.Tensor,
    z_hierarch: Optional[torch.Tensor] = None,
    ax: Optional[np.ndarray] = None
):
    """
    Plot stratified and (optional) hierarchical samples.
    """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, 'b-o')
    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, 'r-o')
    ax.set_ylim([-1, 2])
    ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)
    return ax



def crop_center(
    img: torch.Tensor,
    frac: float = 0.5
) -> torch.Tensor:
    """
    Crop center square from image.
    """
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))
    return img[h_offset:-h_offset, w_offset:-w_offset]



class EarlyStopping:
    """
    Early stopping helper based on fitness criterion.
    """
    def __init__(
        self,
        patience: int = 30,
        margin: float = 1e-4
    ):
        self.best_fitness = 0.0  # In our case PSNR
        self.best_iter = 0
        self.margin = margin
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop

    def __call__(
        self,
        iter: int,
        fitness: float
    ):
        """
        Check if criterion for stopping is met.
        """
        if (fitness - self.best_fitness) > self.margin:
            self.best_iter = iter
            self.best_fitness = fitness
        delta = iter - self.best_iter
        stop = delta >= self.patience  # stop training if patience exceeded
        return stop



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    filepath = input("Enter filepath to load data: ")
    savepath = input("Enter path to directory to save files: ")
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


    """
    Hyperparameters
    """
    print("Creating hyperparameters...")

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
    d_filter_fine = 128     # Dimensions of linear layer filters of fine network
    n_layers_fine = 6       # Number of layers in fine network bottleneck

    # Hierarchical sampling
    n_samples_hierarchical = 64   # Number of samples per ray
    perturb_hierarchical = False  # If set, applies noise to sample positions

    # Optimizer
    lr = 5e-4  # Learning rate

    # Training
    n_iters = 10000
    batch_size = 2**14          # Number of rays per gradient step (power of 2)
    one_image_per_step = True   # One image per gradient step (disables batching)
    chunksize = 2**14           # Modify as needed to fit in GPU memory
    center_crop = True          # Crop the center of image (one_image_per_)
    center_crop_iters = 50      # Stop cropping center after this many epochs
    display_rate = 25          # Display test output every X epochs

    # Early Stopping
    warmup_iters = 100          # Number of iterations during warmup phase
    warmup_min_fitness = 10.0   # Min val PSNR to continue training at warmup_iters
    n_restarts = 10             # Number of times to restart if training stalls

    # We bundle the kwargs for various functions to pass all at once.
    kwargs_sample_stratified = {
        'n_samples': n_samples,
        'perturb': perturb,
        'inverse_depth': inverse_depth
    }
    kwargs_sample_hierarchical = {
        'perturb': perturb
    }


    """
    init_models()
    """
    print("Making models...")

    # Encoders
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

    # Early Stopping
    warmup_stopper = EarlyStopping(patience=50)


    """
    train()
    """
    print("Training...")

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
    
        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

        # Backprop!
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

            # Plot example outputs
            fig, ax = plt.subplots(1, 4, figsize=(24,4), gridspec_kw={'width_ratios': [1, 1, 1, 3]})
            ax[0].imshow(rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy())
            ax[0].set_title(f'Iteration: {i}')
            ax[1].imshow(testimg.detach().cpu().numpy())
            ax[1].set_title(f'Target')
            ax[2].plot(range(0, i + 1), train_psnrs, 'r')
            ax[2].plot(iternums, val_psnrs, 'b')
            ax[2].set_title('PSNR (train=red, val=blue')
            z_vals_strat = outputs['z_vals_stratified'].view((-1, n_samples))
            z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
            if 'z_vals_hierarchical' in outputs:
                z_vals_hierarch = outputs['z_vals_hierarchical'].view((-1, n_samples_hierarchical))
                z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
            else:
                z_sample_hierarch = None
                _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
            ax[3].margins(0)
            plt.savefig(savepath + str(i))

        # Check PSNR for issues and stop if any are found.
        if i == warmup_iters - 1:
            if val_psnr < warmup_min_fitness:
                print(f'Val PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping...')
                return False, train_psnrs, val_psnrs
        elif i < warmup_iters:
                if warmup_stopper is not None and warmup_stopper(i, psnr):
                    print(f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...')
                    return False, train_psnrs, val_psnrs

    torch.save(model.state_dict(), savepath + "\\model.pt")
    if fine_model is not None:
        torch.save(model.state_dict(), savepath + "\\fine_model.pt")



if __name__ == '__main__':
    main()