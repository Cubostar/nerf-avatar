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
    
    images, poses, focal = load_data("../datasets/tiny_nerf_data.npz")

    height, width = images.shape[1:3]
    near, far = 2., 6.

    n_training = 100
    testimg_idx = 101

    testimg = torch.from_numpy(images[testimg_idx]).to(device)
    testpose = torch.from_numpy(poses[testimg_idx]).to(device)
    poses = torch.from_numpy(poses).to(device)
    focal = torch.from_numpy(focal).to(device)
    images = torch.from_numpy(images[:n_training]).to(device)

    d_input = 3         
    n_freqs = 10         
    log_space = True     
    use_viewdirs = True  
    n_freqs_views = 4    

    n_samples = 64         
    perturb = True        
    inverse_depth = False 

    d_filter = 128          
    n_layers = 2           
    skip = []               
    use_fine_model = True   

    n_samples_hierarchical = 64  

    lr = 5e-4  

    n_iters = 500
    batch_size = 2**14         
    one_image_per_step = True  
    chunksize = 2**14          
    center_crop = True         
    center_crop_iters = 50     
    display_rate = 25         

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

    if use_viewdirs:
        encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,
                                            log_space=log_space)
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

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
            batch = rays_rgb[i_batch:i_batch + batch_size]
            batch = torch.transpose(batch, 0, 1)
            rays_o, rays_d, target_img = batch
            height, width = target_img.shape[:2]
            i_batch += batch_size
            if i_batch >= rays_rgb.shape[0]:
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                i_batch = 0
        target_img = target_img.reshape([-1, 3])

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

        psnr = -10. * torch.log10(loss)
        train_psnrs.append(psnr.item())

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
    torch.save(outputs, 'trained_models/outputs.pt')
    torch.save(model.state_dict(), PATH)
    torch.save(train_psnrs, 'trained_models/train_psnrs_1000.pt')
    torch.save(val_psnrs, 'trained_models/val_psnrs_1000.pt')

if __name__ == '__main__':
    main()
