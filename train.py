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

from torchvision.models.inception import inception_v3
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
import numpy as np
from timm import create_model

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')


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
        "batch_size": 1,
        "optimizer": "Adam",
        "loss": "GeneratorLoss + Adversarial",
    })

    # Global training variables:
    CROP_SIZE = opt.crop_size # crop size our of the full image
    UPSCALE_FACTOR = opt.upscale_factor # upscale factor
    NUM_EPOCHS = opt.num_epochs # learning epochs

    # Load train and validation sets:
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    # Initialize generator and discriminator networks
    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # Initialize GeneratorLoss class as generator_criterion
    generator_criterion = GeneratorLoss()

    # Check if cuda is avaliable and apply generator_criterion to be based on it
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    # Initialize Adam optimizers for both generator and discriminator
    # TODO: can check AdamW instead of Adam
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    # Initialize results dictionary:
    results = {'d_loss': [], 'g_loss': [],
               'image_loss':[],'adversarial_loss':[],
               'perception_loss':[],'tv_loss':[],
               'w_image_loss':[], 'w_adversarial_loss':[],
               'w_perception_loss':[], 'w_tv_loss':[],
               'd_score': [], 'g_score': [], 'train_psnr': [],
               'train_ssim': [], 'val_psnr': [], 'val_ssim': []}

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
            g_update_first = True # obsolete? TODO: Remove
            batch_size = data.size(0) # get number of samples in batch
            running_results['batch_sizes'] += batch_size # accumulating number of samples

            ############## Generator Train ##############
            # (1) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            #############################################
            real_img = target # full resolution image
            print("X size: ", data.size()) # print data batch size
            print("target size: ", real_img.size()) # print target batch size

            # convert everything to cuda
            if torch.cuda.is_available():
                real_img = real_img.float().cuda()
            z = data
            if torch.cuda.is_available():
                z = z.float().cuda()

            # generates high resolution image with generator
            fake_img = netG(z)

            # returns average classification score for the batch:
            fake_out = netD(fake_img).mean()

            # one backwards step of the generator
            optimizerG.zero_grad()
            # calls forward generator criterion
            (g_loss, image_loss,   adversarial_loss,  perception_loss,    tv_loss ,
                     w_image_loss, w_adversarial_loss, w_perception_loss, w_tv_loss) \
                = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            ############ Discriminator train ################
            # (2) Update D network: maximize D(x)-1-D(G(z))
            #################################################

            # classification of the real images
            real_out = netD(real_img).mean()
            # classification of the fake images
            fake_out = netD(fake_img.detach()).mean()

            # Encourages the discriminator classify real images as 1 and fake images as 0
            d_loss = 1 - real_out + fake_out

            # one backwards step of the discriminator
            optimizerD.zero_grad()
            d_loss.backward()

            # fake images to calculate g_score
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerD.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size # accumulate generator loss (total score over the epoch)
            running_results['d_loss'] += d_loss.item() * batch_size # accumulate discriminator loss (total score over the epoch)
            running_results['d_score'] += real_out.item() * batch_size # accumulate discriminator's confidence on real images (total score over the epoch)
            running_results['g_score'] += fake_out.item() * batch_size # accumulate discriminator's confidence on fake images (total score over the epoch)

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        # Generator Evaluation
        netG.eval()

        # TODO: Delete
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # TODO: add validation loss metric

        # When epoch is finished:
        with torch.no_grad():

            # Over train set
            train_eval_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            for train_lr, train_hr in tqdm(train_loader):
                batch_size = train_lr.size(0)
                train_eval_results['batch_sizes'] += batch_size
                lr = train_lr # low res images batch
                hr = train_hr # high res images batch
                if torch.cuda.is_available():
                    lr = lr.float().cuda()
                    hr = hr.float().cuda()
                # Generator predict super res image
                sr = netG(lr)

                # Collect MSE over batch
                batch_mse = ((sr - hr) ** 2).data.mean()
                train_eval_results['mse'] += batch_mse * batch_size

                # Collect SSIM over batch
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                train_eval_results['ssims'] += batch_ssim * batch_size

            # Calculate PSNR and SSIM for the entire epoch
            train_eval_results['psnr'] = 10 * log10(
                (hr.max() ** 2) / (train_eval_results['mse'] / train_eval_results['batch_sizes']))
            train_eval_results['ssim'] = train_eval_results['ssims'] / train_eval_results['batch_sizes']

            # Appends psnr and ssim for train set over epoch - duplicate saving
            # results['train_psnr'].append(train_eval_results['psnr'])
            # results['train_ssim'].append(train_eval_results['ssim'])

            # Over val set
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr # low res images batch
                hr = val_hr # high res images batch
                if torch.cuda.is_available():
                    lr = lr.float().cuda()
                    hr = hr.float().cuda()
                # Generator predict super res image
                sr = netG(lr)

                # Collect MSE over batch
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size

                # Collect SSIM over batch
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size

            # Calculate PSNR and SSIM for the entire epoch
            valing_results['psnr'] = 10 * log10(
                (hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

            # Appends psnr and ssim for train set over epoch - duplicate saving
            # results['val_psnr'].append(valing_results['psnr'])
            # results['val_ssim'].append(valing_results['ssim'])

            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))

            # TODO: Obsolite
            val_images.extend(
                [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                 display_transform()(sr.data.cpu().squeeze(0))])

            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['train_psnr'].append(train_eval_results['psnr'])
        results['train_ssim'].append(train_eval_results['ssim'])
        results['val_psnr'].append(valing_results['psnr'])
        results['val_ssim'].append(valing_results['ssim'])

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
            "example_images": wandb_images
        })

        wandb.log({
            "epoch": epoch,
            "train/Loss_D": results['d_loss'][-1],
            "train/Loss_G": results['g_loss'][-1],
            "train/Score_D": results['d_score'][-1],
            "train/Score_G": results['g_score'][-1],
            "train/PSNR": results['train_psnr'][-1],
            "train/SSIM": results['train_ssim'][-1],
            "val/PSNR": results['val_psnr'][-1],
            "val/SSIM": results['val_ssim'][-1],
        })

        # if epoch % 10 == 0 and epoch != 0:
        #     out_path = 'statistics/'
        #     data_frame = pd.DataFrame(
        #         data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
        #               'Score_G': results['g_score'], 'train_PSNR': results['train_psnr'], 'train_SSIM': results['train_ssim'],
        #               'val_PSNR': results['val_psnr'], 'val_SSIM': results['val_ssim']},
        #         index=range(1, epoch + 1))
        #     data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
    wandb.finish()