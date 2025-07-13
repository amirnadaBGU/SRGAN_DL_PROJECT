import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator
import wandb
import torch

from torchvision.models.inception import inception_v3
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
import numpy as np
from timm import create_model
from torchmetrics.image.fid import FrechetInceptionDistance
from model_diffusion import DiffusionModel, Discriminator, sample
from torch.utils.data import Subset
import cProfile

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
parser.add_argument('--time_steps', default=2000, type=int, help='number of diffusion steps')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')

if __name__ == '__main__':
    # Parse learning configuration:
    opt = parser.parse_args()

    # Wandb definition:
    project = "SRGAN_DL_PROJECT"
    wandb.init(project=project)

    # Wandb configuration::
    wandb.config.update({
        "crop_size": opt.crop_size,
        "upscale_factor": opt.upscale_factor,
        "num_epochs": opt.num_epochs,
        "batch_size": opt.batch_size,
        "optimizer": "Adam",
        "loss": "Multiple",
    })

    # Global training variables:
    CROP_SIZE = opt.crop_size  # crop size our of the full image
    UPSCALE_FACTOR = opt.upscale_factor  # upscale factor
    NUM_EPOCHS = opt.num_epochs  # learning epochs
    TIME_STEPS = opt.time_steps
    BATCH_SIZE = opt.batch_size

    # Load train and validation sets:
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR,
                                       diffusion=True)
    val_set = TrainDatasetFromFolder('data/DIV2K_valid_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR,
                                     diffusion=True, val=True)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    # Initialize diffusion model and discriminator
    image_dims = (3, CROP_SIZE, CROP_SIZE)
    netG = DiffusionModel(time_steps=TIME_STEPS, image_dims=image_dims)
    print('# diffusion model parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # Initialize GeneratorLoss class as generator_criterion
    generator_criterion = GeneratorLoss()

    # Move to cuda if available
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    # Optimizers for both generator and discriminator
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    # Initialize results dictionary:
    results = {'d_loss': [], 'g_loss': [],
               'image_loss': [], 'adversarial_loss': [],
               'perception_loss': [], 'tv_loss': [],
               'w_image_loss': [], 'w_adversarial_loss': [],
               'w_perception_loss': [], 'w_tv_loss': [],
               'd_score': [], 'g_score': [], 'train_psnr': [],
               'train_ssim': [], 'val_psnr': [], 'val_ssim': [], 'val_fid': []}

    # Training loop:
    for epoch in range(1, NUM_EPOCHS + 1):

        # tqdm train loader
        train_bar = tqdm(train_loader)

        # initialize single epoch results dictionary:
        running_results = {'batch_sizes': 0,
                           'd_loss': 0, 'g_loss': 0,
                           'image_loss': 0, 'adversarial_loss': 0,
                           'perception_loss': 0, 'tv_loss': 0,
                           'w_image_loss': 0, 'w_adversarial_loss': 0,
                           'w_perception_loss': 0, 'w_tv_loss': 0,
                           'd_score': 0, 'g_score': 0}

        # set NetG and NetD to be trainable:
        netG.train()
        netD.train()

        # loop over all train data:
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
            real_img = target
            if torch.cuda.is_available():
                real_img = real_img.float().cuda()
            z = data
            if torch.cuda.is_available():
                z = z.float().cuda()

            # Pick random timesteps for batch
            bs = real_img.size(0)
            t = torch.randint(1, TIME_STEPS, (bs,), device=real_img.device)

            # Add noise to HR at step t
            gamma = netG.alpha_hats.to(real_img.device)[t]
            noisy_img, less_noised_img, target_noise = netG.add_noise(real_img, t)

            # Denoise one step, condition on z

            model_input = torch.cat([z, noisy_img], dim=1)
            predicted_noise = netG(model_input, gamma)
            alpha_hat = netG.alpha_hats.to(real_img.device)[t].view(-1, 1, 1, 1)
            denoised_img = (noisy_img - (1 - alpha_hat).sqrt() * predicted_noise) / alpha_hat.sqrt()

            # returns average classification score for the batch:
            fake_out = netD(predicted_noise).mean()
            # Generator update
            optimizerG.zero_grad()
            (g_loss, image_loss, adversarial_loss, perception_loss, tv_loss,
             w_image_loss, w_adversarial_loss, w_perception_loss, w_tv_loss) = generator_criterion(fake_out,
                                                                                                   predicted_noise,
                                                                                                   target_noise)
            g_loss.backward()
            optimizerG.step()

            # Discriminator update
            real_out = netD(target_noise).mean()
            fake_out = netD(predicted_noise.detach()).mean()

            # Encourages the discriminator classify real images as 1 and fake images as 0
            d_loss = 1 - real_out + fake_out

            # one backwards step of the discriminator
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # Logging
            running_results['g_loss'] += float(g_loss.item()) * batch_size
            running_results['d_loss'] += float(d_loss.item()) * batch_size
            running_results['image_loss'] += float(image_loss.item()) * batch_size
            running_results['w_image_loss'] += float(w_image_loss.item()) * batch_size
            running_results['adversarial_loss'] += float(adversarial_loss.item()) * batch_size
            running_results['w_adversarial_loss'] += float(w_adversarial_loss.item()) * batch_size
            running_results['perception_loss'] += float(perception_loss.item()) * batch_size
            running_results['w_perception_loss'] += float(w_perception_loss.item()) * batch_size
            running_results['tv_loss'] += float(tv_loss.item()) * batch_size
            running_results['w_tv_loss'] += float(w_tv_loss.item()) * batch_size
            running_results['d_score'] += float(real_out.item()) * batch_size
            running_results['g_score'] += float(fake_out.item()) * batch_size
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        # Save image in wanb
        sample_sr = denoised_img[0].cpu()
        sample_hr = target[0].cpu()
        sample_lr = data[0].cpu()

        wandb_images = [
            wandb.Image(sample_lr, caption="trainLow Resolution (LR)"),
            wandb.Image(sample_sr, caption="train denoised image"),
            wandb.Image(sample_hr, caption="High Resolution (HR)")
        ]

        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['image_loss'].append(running_results['image_loss'] / running_results['batch_sizes'])
        results['w_image_loss'].append(running_results['w_image_loss'] / running_results['batch_sizes'])
        results['adversarial_loss'].append(running_results['adversarial_loss'] / running_results['batch_sizes'])
        results['w_adversarial_loss'].append(running_results['w_adversarial_loss'] / running_results['batch_sizes'])
        results['perception_loss'].append(running_results['perception_loss'] / running_results['batch_sizes'])
        results['w_perception_loss'].append(running_results['w_perception_loss'] / running_results['batch_sizes'])
        results['tv_loss'].append(running_results['tv_loss'] / running_results['batch_sizes'])
        results['w_tv_loss'].append(running_results['w_tv_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])

        # Generator Evaluation
        if epoch % 10 == 0:
            netG.eval()

            # TODO: Delete
            out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # TODO: add validation loss metric

            # # When epoch is finished:
            with torch.no_grad():

                # Over val set
                val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0, 'fid': 0}

                # Initialize FID
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                fid_metric = FrechetInceptionDistance(normalize=True).to(device)

                for val_lr, val_hr in val_bar:
                    batch_size = val_lr.size(0)
                    valing_results['batch_sizes'] += batch_size
                    lr = val_lr
                    hr = val_hr
                    if torch.cuda.is_available():
                        lr = lr.float().cuda()
                        hr = hr.float().cuda()
                    sr = sample(netG, lr, val=True)

                    # Collect MSE over batch
                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size

                    # Collect SSIM over batch
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    valing_results['ssims'] += batch_ssim * batch_size

                    # Collect FID over batch
                    fid_metric.update(hr.to(device), real=True)
                    fid_metric.update(sr.to(device), real=False)

                # Compute FID
                valing_results['fid'] = fid_metric.compute()  # Final score – no division needed

                # Calculate PSNR and SSIM for the entire epoch
                valing_results['psnr'] = 10 * log10(
                    (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

            # save model parameters
            torch.save(netG.state_dict(), 'epochs/netG_diffusion_GAN_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

            results['val_psnr'].append(valing_results['psnr'])
            results['val_ssim'].append(valing_results['ssim'])
            results['val_fid'].append(valing_results['fid'])

            # Save image in wanb
            sample_lr = val_lr[0].cpu()
            sample_sr = sr[0].cpu()
            sample_hr = hr[0].cpu()

            wandb_images = [
                wandb.Image(sample_lr, caption="Low Resolution (LR)"),
                wandb.Image(sample_sr, caption="Super Resolution (SR)"),
                wandb.Image(sample_hr, caption="High Resolution (HR)")
            ]

            wandb.log({
                "example_images_train": wandb_images,
                "epoch": epoch,
                "train/Loss_D": results['d_loss'][-1],
                "train/Loss_G": results['g_loss'][-1],
                "train/image_loss": results['image_loss'][-1],
                "train/w_image_loss": results['w_image_loss'][-1],
                "train/adversarial_loss": results['adversarial_loss'][-1],
                "train/w_adversarial_loss": results['w_adversarial_loss'][-1],
                "train/perception_loss": results['perception_loss'][-1],
                "train/w_perception_loss": results['w_perception_loss'][-1],
                "train/tv_loss": results['tv_loss'][-1],
                "train/w_tv_loss": results['w_tv_loss'][-1],
                "train/Score_D": results['d_score'][-1],
                "train/Score_G": results['g_score'][-1],
                "val/PSNR": results['val_psnr'][-1],
                "val/SSIM": results['val_ssim'][-1],
                "val/FID": results['val_fid'][-1],
                "example_images": wandb_images,
            })

    wandb.finish()