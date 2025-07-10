import random

import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, Subset

from model_diffusion import DiffusionModel, sample
from data_utils import TrainDatasetFromFolder
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- Config ---- #
MODEL_PATH = 'best_models/diffusion/ddpm_epoch_4_320.pth'
DATA_PATH = 'data/DIV2K_valid_HR'
CROP_SIZE = 128
UPSCALE_FACTOR = 4
TIME_STEPS = 2000

# ---- Load model ---- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = DiffusionModel(time_steps=TIME_STEPS, image_dims=(3, CROP_SIZE, CROP_SIZE)).to(device)
netG.load_state_dict(torch.load(MODEL_PATH, map_location=device))
netG.eval()

# ---- Load single image from dataset ---- #
dataset = TrainDatasetFromFolder(DATA_PATH, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR, diffusion=True, val=True)
subset = Subset(dataset, [random.randint(0, len(dataset) - 1)])
loader = DataLoader(subset, batch_size=1, shuffle=False)

# ---- Inference ---- #
with torch.no_grad():
    for lr_img, hr_img in loader:
        lr_img = lr_img.to(device).float()
        hr_img = hr_img.to(device).float()

        sr_img = sample(netG, lr_img, val=True)
        break

# ---- Convert to PIL for display ---- #
to_pil = ToPILImage()
lr_img_pil = to_pil(lr_img.squeeze().cpu().clamp(0, 1))
sr_img_pil = to_pil(sr_img.squeeze().cpu().clamp(0, 1))
hr_img_pil = to_pil(hr_img.squeeze().cpu().clamp(0, 1))

# ---- Plot ---- #
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(lr_img_pil)
axes[0].set_title("Low Resolution (LR)")
axes[1].imshow(sr_img_pil)
axes[1].set_title("Super Resolution (SR)")
axes[2].imshow(hr_img_pil)
axes[2].set_title("High Resolution (HR)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
