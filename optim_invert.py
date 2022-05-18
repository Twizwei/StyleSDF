# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import os
import torch
import trimesh
import numpy as np
from munch import *
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from torchvision import transforms
from skimage.measure import marching_cubes
from scipy.spatial import Delaunay
from options import BaseOptions
from model import Generator
from utils import (
    generate_camera_params,
    align_volume,
    extract_mesh_with_marching_cubes,
    xyz2mesh,
)

import PIL.Image
import lpips
import imageio
import pdb

device = "cuda"

sample_cam_extrinsics = torch.tensor([[[ 0.8154, -0.0659, -0.5751, -0.5751],
                                        [-0.0000,  0.9935, -0.1138, -0.1138],
                                        [ 0.5789,  0.0928,  0.8101,  0.8101]]]).to(device)
sample_focals = torch.tensor([[[304.4597]]]).to(device)
sample_near = torch.tensor([[[0.8800]]]).to(device)
sample_far = torch.tensor([[[1.1200]]]).to(device)

def project(
    g_ema,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 1e-3,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    output_dir: str,
    device: torch.device
):
    
    assert target.shape == (3, g_ema.size, g_ema.size)

    def logprint(*args):
        if verbose:
            print(*args)


    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    w_samples = torch.from_numpy(np.random.RandomState(123).randn(w_avg_samples, g_ema.style_dim)).to(device)
    # w_samples = g_ema.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    # noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    # with dnnlib.util.open_url(url) as f:
    #     vgg16 = torch.jit.load(f).eval().to(device)
    
    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    # if target_images.shape[2] > 256:
    #     target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    # target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    target_images = target_images/255. * 2.0 - 1.0  # [0, 255] -> [-1, 1]

    # w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True).repeat(1, 10, 1) # pylint: disable=not-callable
    w_opt = torch.from_numpy(w_avg).detach().clone().repeat(1, 10, 1).to(device)

    w_opt_ = torch.randn((1, 256), dtype=torch.float32, device=device, requires_grad=True)
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(w_opt.shape[0], 1, 1, 1).normal_())

    for noise in noises:
        noise.requires_grad = True

    optimizer = torch.optim.Adam([w_opt] + noises, betas=(0.9, 0.999), lr=initial_learning_rate)

    for param in g_ema.renderer.parameters():
        param.requires_grad = False
    
    # optimizer = torch.optim.Adam([
    #                             {'params': [w_opt], 'lr': float(initial_learning_rate)},
    #                             # {'params': G2.generator.parameters(), 'lr': float(initial_learning_rate)},
    #                             {'params':  list(noise_bufs.values()), 'lr': float(initial_learning_rate)},
    #                             # betas=(0.9, 0.999),
    #                             ])
    
    # lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    l2_fn = torch.nn.MSELoss(reduction='mean')

    # Init noise.
    # for buf in noise_bufs.values():
    #     buf[:] = torch.randn_like(buf)
    #     buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        # w_noise = torch.randn_like(w_opt) * w_noise_scale
        # ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])


        
        # synth_images, _ = g_ema([w_opt], sample_cam_extrinsics, sample_focals, sample_near, sample_far, input_is_latent=True, randomize_noise=False)
        thumb_rgb, features, sdf, mask, xyz, eikonal_term = g_ema.renderer(sample_cam_extrinsics, sample_focals, sample_near, sample_far, styles=w_opt_, return_eikonal=False)
        synth_images, _ = g_ema.decoder(features, [w_opt],
                                        transform=None,
                                        return_latents=False,
                                        inject_index=None, truncation=1.0,
                                        truncation_latent=None, noise=None,
                                        input_is_latent=True, randomize_noise=False, input_is_w_plus=True,
                                        mesh_path=None)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        # synth_images = (synth_images + 1) * (255/2)

        # disable vgg feature for now
        # if synth_images.shape[2] > 256:
        #     synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # # Features for synth images.
        # synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        # dist = (target_features - synth_features).square().sum()

        # LPIPS distance
        lpips_loss = lpips_fn(target_images, synth_images).mean()

        # L2 loss
        l2_loss = l2_fn(target_images, synth_images)

        # Noise regularization.
        reg_loss = 0.0
        # for v in noise_bufs.values():
        #     noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
        #     while True:
        #         reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
        #         reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
        #         if noise.shape[2] <= 8:
        #             break
        #         noise = F.avg_pool2d(noise, kernel_size=2)
        # loss = dist + reg_loss * regularize_noise_weight
        loss = 0.1 * lpips_loss + l2_loss


        # Step
        optimizer.zero_grad(set_to_none=True)
        pdb.set_trace()
        loss.backward()
        optimizer.step()
        # logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
        logprint(f'step {step+1:>4d}/{num_steps}: lpips {lpips_loss.item():<4.2f} l2 {l2_loss.item():4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()
        # save intermediate result
        if (step + 1) % 50 == 0 or step == 0:
            with torch.no_grad():
                synth_images, _ = g_ema([w_opt], sample_cam_extrinsics, sample_focals, sample_near, sample_far)
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
                synth_images = (synth_images + 1) * (255/2)
                synth_images = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                imageio.imwrite(os.path.join(output_dir, 'intermediates', 'step_{:05}.jpg'.format(step+1) ), synth_images)

        # Normalize noise.

    return w_out

#----------------------------------------------------------------------------

def run_projection(
    G,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    # G.eval()
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.size, G.size), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    # Optimize projection.
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'intermediates'), exist_ok=True)
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True,
        output_dir=outdir,
    )

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            # synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')['img']
            synth_image, _ = g_ema([projected_w.unsqueeze(0)], sample_cam_extrinsics, sample_focals, sample_near, sample_far)
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    # synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')['img']
    synth_image, _ = g_ema([projected_w.unsqueeze(0)], sample_cam_extrinsics, sample_focals, sample_near, sample_far)
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    # options for StyleSDF
    opt = BaseOptions()
    opt = opt.parse()
    opt.model.is_test = True
    opt.model.freeze_renderer = False
    opt.rendering.offset_sampling = True
    opt.rendering.static_viewdirs = True
    opt.rendering.force_background = True
    opt.rendering.perturb = 0
    opt.inference.size = opt.model.size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.inference.return_xyz = opt.rendering.return_xyz

    # find checkpoint directory
    # check if there's a fully trained model
    checkpoints_dir = 'full_models'
    checkpoint_path = os.path.join(checkpoints_dir, opt.experiment.expname + '.pt')
    if os.path.isfile(checkpoint_path):
        # define results directory name
        result_model_dir = 'final_model'
    else:
        checkpoints_dir = os.path.join('checkpoint', opt.experiment.expname)
        checkpoint_path = os.path.join(checkpoints_dir,
                                       'models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
        # define results directory name
        result_model_dir = 'iter_{}'.format(opt.experiment.ckpt.zfill(7))

    # create results directory
    results_dir_basename = os.path.join(opt.inference.results_dir, opt.experiment.expname)
    opt.inference.results_dst_dir = os.path.join(results_dir_basename, result_model_dir)
    if opt.inference.fixed_camera_angles:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'fixed_angles')
    else:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'random_angles')
    os.makedirs(opt.inference.results_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.inference.results_dst_dir, 'images'), exist_ok=True)
    if not opt.inference.no_surface_renderings:
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'depth_map_meshes'), exist_ok=True)
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'marching_cubes_meshes'), exist_ok=True)

    # load saved model
    checkpoint = torch.load(checkpoint_path)

    # load image generation model
    g_ema = Generator(opt.model, opt.rendering).to(device)
    pretrained_weights_dict = checkpoint["g_ema"]
    model_dict = g_ema.state_dict()
    for k, v in pretrained_weights_dict.items():
        if v.size() == model_dict[k].size():
            model_dict[k] = v

    g_ema.load_state_dict(model_dict)

    # load a second volume renderer that extracts surfaces at 128x128x128 (or higher) for better surface resolution
    if not opt.inference.no_surface_renderings:
        opt['surf_extraction'] = Munch()
        opt.surf_extraction.rendering = opt.rendering
        opt.surf_extraction.model = opt.model.copy()
        opt.surf_extraction.model.renderer_spatial_output_dim = 128
        opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
        opt.surf_extraction.rendering.return_xyz = True
        opt.surf_extraction.rendering.return_sdf = True
        surface_g_ema = Generator(opt.surf_extraction.model, opt.surf_extraction.rendering, full_pipeline=False).to(device)

        # Load weights to surface extractor
        surface_extractor_dict = surface_g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
                surface_extractor_dict[k] = v

        surface_g_ema.load_state_dict(surface_extractor_dict)
    else:
        surface_g_ema = None
    
    # get the mean latent vector for g_ema
    if opt.inference.truncation_ratio < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(opt.inference.truncation_mean, device)
    else:
        surface_mean_latent = None

    # get the mean latent vector for surface_g_ema
    if not opt.inference.no_surface_renderings:
        surface_mean_latent = mean_latent[0]
    else:
        surface_mean_latent = None
    
    target = '/vulcanscratch/yiranx/codes/StyleSDF/evaluations/ffhq1024x1024/final_model/random_angles/images/0000000.png'
    num_steps = 500
    results_dir = './evaluations/optim_project'

    run_projection(G=g_ema, target_fname=target, outdir=results_dir, save_video=True, seed=102, num_steps=num_steps)
#----------------------------------------------------------------------------
