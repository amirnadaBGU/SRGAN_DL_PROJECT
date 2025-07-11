import os
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, RandomCrop
from torch.utils.data import DataLoader, Subset
from PIL import Image

# Prevent OpenMP crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- Custom Modules ---- #
from model_diffusion import DiffusionModel, sample as diffusion_sample
from data_utils import AlignedInferenceDataset

# ---- Configuration ---- #
CROP_SIZE = 128
UPSCALE_FACTOR = 4
TIME_STEPS = 2000

DIFFUSION_MODEL_PATH_1 = 'best_models/diffusion/ddpm_epoch_4_320.pth'
DIFFUSION_MODEL_PATH_2 = 'best_models/diffusion/diffusion_GAN_noise_prediction_loss_epoch_4_135.pth'
DATA_PATH = 'data/DIV2K_valid_HR'

# ---- Device ---- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Diffusion Models ---- #
netG_diff_1 = DiffusionModel(time_steps=2000, image_dims=(3, CROP_SIZE, CROP_SIZE)).to(device)
netG_diff_1.load_state_dict(torch.load(DIFFUSION_MODEL_PATH_1, map_location=device))
netG_diff_1.eval()

netG_diff_2 = DiffusionModel(time_steps=2000, image_dims=(3, CROP_SIZE, CROP_SIZE)).to(device)
netG_diff_2.load_state_dict(torch.load(DIFFUSION_MODEL_PATH_2, map_location=device))
netG_diff_2.eval()

# ---- Select a Sample Index ---- #
image_filenames = sorted(os.listdir(DATA_PATH))
sample_index = 1
sample_image_path = os.path.join(DATA_PATH, image_filenames[sample_index])

# ---- Loop Over Random Crops ---- #
for k in range(50):
    # Generate same crop coordinates
    ref_img = Image.open(sample_image_path).convert("RGB")
    i, j, h, w = RandomCrop.get_params(ref_img, output_size=(CROP_SIZE, CROP_SIZE))
    shared_crop_coords = (i, j, h, w)

    # ---- Datasets for Each Diffusion ---- #
    dataset_diff1 = AlignedInferenceDataset(DATA_PATH, CROP_SIZE, UPSCALE_FACTOR, crop_coords=shared_crop_coords, diffusion=True)
    dataset_diff2 = AlignedInferenceDataset(DATA_PATH, CROP_SIZE, UPSCALE_FACTOR, crop_coords=shared_crop_coords, diffusion=True)

    loader_diff1 = DataLoader(Subset(dataset_diff1, [sample_index]), batch_size=1, shuffle=False)
    loader_diff2 = DataLoader(Subset(dataset_diff2, [sample_index]), batch_size=1, shuffle=False)

    with torch.no_grad():
        # Diffusion Model 1
        for lr_diff1, hr_img in loader_diff1:
            lr_diff1 = lr_diff1.to(device).float()
            hr_img = hr_img.to(device).float()
            sr_img_diff1 = diffusion_sample(netG_diff_1, lr_diff1, val=True)
            break

        # Diffusion Model 2
        for lr_diff2, _ in loader_diff2:
            lr_diff2 = lr_diff2.to(device).float()
            sr_img_diff2 = diffusion_sample(netG_diff_2, lr_diff2, val=True)
            break

    # ---- Convert to PIL for Display ---- #
    to_pil = ToPILImage()
    lr_img_pil       = to_pil(lr_diff1.squeeze().cpu().clamp(0, 1))
    sr_img_diff1_pil = to_pil(sr_img_diff1.squeeze().cpu().clamp(0, 1))
    sr_img_diff2_pil = to_pil(sr_img_diff2.squeeze().cpu().clamp(0, 1))
    hr_img_pil       = to_pil(hr_img.squeeze().cpu().clamp(0, 1))

    # ---- Plot ---- #
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Low Resolution (LR)", "SR - Diffusion A", "SR - Diffusion B", "High Resolution (HR)"]
    images = [lr_img_pil, sr_img_diff1_pil, sr_img_diff2_pil, hr_img_pil]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
