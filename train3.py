import argparse
import os
from math import log10

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder
from model3 import DiffusionModel
import wandb

from torchmetrics.image.fid import FrechetInceptionDistance

parser = argparse.ArgumentParser(description='Train DDPM Super Resolution Model')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--time_steps', default=1000, type=int, help='number of diffusion steps')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')

if __name__ == '__main__':
    # Parse learning configuration:
    opt = parser.parse_args()

    # Wandb definition:
    project = "SRGAN_DL_PROJECT"
    wandb.init(project=project)

    # Wandb configuration:
    wandb.config.update({
        "crop_size": opt.crop_size,
        "upscale_factor": opt.upscale_factor,
        "num_epochs": opt.num_epochs,
        "batch_size": opt.batch_size,
        "optimizer": "Adam",
        "loss": "DiffusionMSE",
        "time_steps": opt.time_steps,
    })

    # Global training variables:
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    TIME_STEPS = opt.time_steps
    BATCH_SIZE = opt.batch_size

    # Load train and validation sets:
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, diffusion=True)
    val_set = TrainDatasetFromFolder('data/DIV2K_valid_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, diffusion=True)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    # Initialize DDPM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_dims = (3, CROP_SIZE, CROP_SIZE)
    ddpm = DiffusionModel(time_steps=TIME_STEPS, image_dims=image_dims).to(device)
    print('# model parameters:', sum(param.numel() for param in ddpm.parameters()))

    # Optimizer and loss
    optimizer = optim.Adam(ddpm.parameters())
    criterion = torch.nn.MSELoss(reduction="mean")

    # Results dictionary for logging
    results = {'loss': [], 'train_psnr': [], 'train_ssim': [], 'val_psnr': [], 'val_ssim': [], 'val_fid': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        ddpm.train()
        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch}/{NUM_EPOCHS}]")
        running_loss = 0.0
        num_batches = 0

        for x, y in train_bar:
            # x: LR image (already upscaled to HR size), y: HR image
            x, y = x.to(device).float(), y.to(device).float()
            bs = y.shape[0]
            ts = torch.randint(low=1, high=TIME_STEPS, size=(bs,)).to(device)
            gamma = ddpm.alpha_hats.to(device)[ts]
            y_noised, target_noise = ddpm.add_noise(y, ts)
            model_input = torch.cat([x, y_noised], dim=1)
            predicted_noise = ddpm(model_input, gamma)
            loss = criterion(target_noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bs
            num_batches += bs
            train_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / num_batches
        results['loss'].append(avg_loss)

        # --------- Evaluation ---------
        if epoch % 1  == 0:
            ddpm.eval()
            with torch.no_grad():
                # Over train set
                # train_eval_results = {'mse': 0.0, 'ssims': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'batch_sizes': 0}
                # for train_lr, train_hr in tqdm(train_loader, desc='[Train Evaluation]'):
                #     train_lr = train_lr.to(device).float()
                #     train_hr = train_hr.to(device).float()
                #     # DDPM sampling: start from noise, condition on LR
                #     y = torch.randn_like(train_hr, device=device)
                #     for t in range(TIME_STEPS - 1, 0, -1):
                #         alpha_t, alpha_t_hat, beta_t = ddpm.alphas[t], ddpm.alpha_hats[t], ddpm.betas[t]
                #         t_tensor = torch.tensor([t] * train_lr.size(0), device=device).long()
                #         pred_noise = ddpm(torch.cat([train_lr, y], dim=1), alpha_t_hat.to(device).repeat(train_lr.size(0)))
                #         y = (torch.sqrt(1 / alpha_t)) * (y - (1 - alpha_t) / torch.sqrt(1 - alpha_t_hat) * pred_noise)
                #         if t > 1:
                #             noise = torch.randn_like(y)
                #             y = y + torch.sqrt(beta_t) * noise
                #     sr = y
                #     batch_size = train_lr.size(0)
                #     train_eval_results['batch_sizes'] += batch_size
                #     batch_mse = ((sr - train_hr) ** 2).data.mean().item()
                #     train_eval_results['mse'] += batch_mse * batch_size
                #     batch_ssim = pytorch_ssim.ssim(sr, train_hr).item()
                #     train_eval_results['ssims'] += batch_ssim * batch_size
                # train_eval_results['psnr'] = 10 * log10((train_hr.max() ** 2) / (train_eval_results['mse'] / train_eval_results['batch_sizes']))
                # train_eval_results['ssim'] = train_eval_results['ssims'] / train_eval_results['batch_sizes']
                # results['train_psnr'].append(train_eval_results['psnr'])
                # results['train_ssim'].append(train_eval_results['ssim'])

                # Over val set (PSNR/SSIM/FID)
                val_batches = list(val_loader)
                valing_results = {'mse': 0.0, 'ssims': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'batch_sizes': 0, 'fid': 0.0}
                real_images = []

                generated_images = []
                fid_metric = FrechetInceptionDistance(normalize=True).to(device)
                for val_lr, val_hr in tqdm(val_batches[-2:], desc='[Val Evaluation]'):
                    val_lr = val_lr.to(device).float()
                    val_hr = val_hr.to(device).float()
                    # DDPM sampling: start from noise, condition on LR
                    y = torch.randn_like(val_hr, device=device)
                    for t in range(TIME_STEPS - 1, 0, -1):
                        alpha_t, alpha_t_hat, beta_t = ddpm.alphas[t], ddpm.alpha_hats[t], ddpm.betas[t]
                        t_tensor = torch.tensor([t] * val_lr.size(0), device=device).long()
                        pred_noise = ddpm(torch.cat([val_lr, y], dim=1), alpha_t_hat.to(device).repeat(val_lr.size(0)))
                        y = (torch.sqrt(1 / alpha_t)) * (y - (1 - alpha_t) / torch.sqrt(1 - alpha_t_hat) * pred_noise)
                        if t > 1:
                            noise = torch.randn_like(y)
                            y = y + torch.sqrt(beta_t) * noise
                    sr = y
                    batch_size = val_lr.size(0)
                    valing_results['batch_sizes'] += batch_size
                    batch_mse = ((sr - val_hr) ** 2).data.mean().item()
                    valing_results['mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, val_hr).item()
                    valing_results['ssims'] += batch_ssim * batch_size
                    fid_metric.update(val_hr, real=True)
                    fid_metric.update(sr, real=False)
                    real_images.append(val_hr.cpu())
                    generated_images.append(sr.cpu())
                valing_results['fid'] = fid_metric.compute().item()
                valing_results['psnr'] = 10 * log10((val_hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                results['val_psnr'].append(valing_results['psnr'])
                results['val_ssim'].append(valing_results['ssim'])
                results['val_fid'].append(valing_results['fid'])

                # Save example images to wandb
                sample_lr = val_lr[0].cpu()
                sample_sr = sr[0].cpu()
                sample_hr = val_hr[0].cpu()
                wandb_images = [
                    wandb.Image(sample_lr, caption="Low Resolution (LR)"),
                    wandb.Image(sample_sr, caption="Super Resolution (SR)"),
                    wandb.Image(sample_hr, caption="High Resolution (HR)")
                ]

                wandb.log({
                    "epoch": epoch,
                    "train/loss": results['loss'][-1],
                    # "train/PSNR": results['train_psnr'][-1],
                    # "train/SSIM": results['train_ssim'][-1],
                    "val/PSNR": results['val_psnr'][-1],
                    "val/SSIM": results['val_ssim'][-1],
                    "val/FID": results['val_fid'][-1],
                    "example_images": wandb_images,
                })

            # Save model checkpoint
            torch.save(ddpm.state_dict(), f'epochs3/ddpm_epoch_{UPSCALE_FACTOR}_{epoch}.pth')

    wandb.finish()