import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Subset
from PIL import Image
from torchvision.transforms import RandomCrop

# Prevent OpenMP crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- Custom Modules ---- #
from model_diffusion import DiffusionModel, sample as diffusion_sample
from model import Generator
from data_utils import AlignedInferenceDataset

# ---- Configuration ---- #
CROP_SIZE = 128
UPSCALE_FACTOR = 4
TIME_STEPS = 2000

DIFFUSION_MODEL_PATH = 'best_models/diffusion/ddpm_epoch_4_320.pth'
GAN_MODEL_PATH = 'C:/Users/ndvam/PycharmProjects/SRGAN_DL_PROJECT/Best_models/basemodel/netG_epoch_4_100.pth'
DATA_PATH = 'data/DIV2K_valid_HR'

# ---- Device ---- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Models ---- #
netG_diff = DiffusionModel(time_steps=TIME_STEPS, image_dims=(3, CROP_SIZE, CROP_SIZE)).to(device)
netG_diff.load_state_dict(torch.load(DIFFUSION_MODEL_PATH, map_location=device))
netG_diff.eval()

netG_gan = Generator(UPSCALE_FACTOR).to(device)
netG_gan.load_state_dict(torch.load(GAN_MODEL_PATH, map_location=device))
netG_gan.eval()

# ---- Select a Random Sample Index ---- #
image_filenames = sorted(os.listdir(DATA_PATH))
sample_index = 1
sample_image_path = os.path.join(DATA_PATH, image_filenames[sample_index])

for k in range(50):
    # Compute shared crop coordinates
    ref_img = Image.open(sample_image_path).convert("RGB")
    i, j, h, w = RandomCrop.get_params(ref_img, output_size=(128, 128))
    shared_crop_coords = (i, j, h, w)

    # ---- Prepare Datasets ---- #
    dataset_gan = AlignedInferenceDataset(DATA_PATH, CROP_SIZE, UPSCALE_FACTOR, crop_coords=shared_crop_coords, diffusion=False)
    dataset_diff = AlignedInferenceDataset(DATA_PATH, CROP_SIZE, UPSCALE_FACTOR, crop_coords=shared_crop_coords, diffusion=True)

    # ---- Load Data for the Selected Sample ---- #
    loader_diff = DataLoader(Subset(dataset_diff, [sample_index]), batch_size=1, shuffle=False)
    loader_gan  = DataLoader(Subset(dataset_gan,  [sample_index]), batch_size=1, shuffle=False)

    # ---- Inference ---- #
    with torch.no_grad():
        for lr_diff, hr_img in loader_diff:
            lr_diff = lr_diff.to(device).float()
            hr_img = hr_img.to(device).float()
            sr_img_diff = diffusion_sample(netG_diff, lr_diff, val=True)
            break

        for lr_gan, _ in loader_gan:
            lr_gan = lr_gan.to(device).float()
            sr_img_gan = netG_gan(lr_gan)
            break

    # ---- Convert to PIL for Display ---- #
    to_pil = ToPILImage()
    lr_img_pil      = to_pil(lr_diff.squeeze().cpu().clamp(0, 1))
    sr_img_gan_pil  = to_pil(sr_img_gan.squeeze().cpu().clamp(0, 1))
    sr_img_diff_pil = to_pil(sr_img_diff.squeeze().cpu().clamp(0, 1))
    hr_img_pil      = to_pil(hr_img.squeeze().cpu().clamp(0, 1))

    # ---- Plot Side-by-Side ---- #
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Low Resolution (LR)", "SR - GAN", "SR - Diffusion", "High Resolution (HR)"]
    images = [lr_img_pil, sr_img_gan_pil, sr_img_diff_pil, hr_img_pil]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
