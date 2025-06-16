import argparse
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import gc
import os
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from model import Generator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Helper functions:

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        Resize([crop_size // upscale_factor,crop_size // upscale_factor], interpolation=Image.BICUBIC),
        ToTensor()
    ])


# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', default='data/DIV2K_valid_HR/0804.png', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_10.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

# ------------------- Setup -------------------
UPSCALE_FACTOR = opt.upscale_factor
USE_GPU = opt.test_mode == 'GPU'
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
CROP_SIZE = 88

# Release any unused memory
gc.collect()
torch.cuda.empty_cache()

# Set device
device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

# Load model
model = Generator(UPSCALE_FACTOR).to(device).eval()
model_path = os.path.join('epochs', MODEL_NAME)
model.load_state_dict(torch.load(model_path, map_location=device))

# Load and prepare image
image = Image.open(IMAGE_NAME).convert('RGB')  # Ensure RGB mode
original_crop = CenterCrop(CROP_SIZE)(image)  # crop the original image to match transform
transform = train_lr_transform(CROP_SIZE, UPSCALE_FACTOR)
image_lr = transform(original_crop)  # Apply transform to image
image_tensor = image_lr.unsqueeze(0).to(device)

#image_tensor = ToTensor()(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():  # No grad = less memory usage
    start = time.perf_counter()
    output = model(image_tensor)
    elapsed = time.perf_counter() - start
    print(f'Inference time: {elapsed:.3f}s')

# Convert and save output image
output_image = ToPILImage()(output.squeeze().cpu())
output_path = f'try_out_srf_{UPSCALE_FACTOR}_{os.path.basename(IMAGE_NAME)}'
output_image.save(output_path)
print(f"Saved output to {output_path}")

import matplotlib.pyplot as plt

# Convert tensors back to PIL images for display

resized_crop = image_lr   # low-res image after downsampling
sr_output = output_image                 # super-resolved image

# Plot all three images
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(original_crop)
axes[0].set_title("Original HR Crop (1024x1024)")
axes[0].axis("off")

axes[1].imshow(ToPILImage()(resized_crop.squeeze().cpu()))
axes[1].set_title(f"Downsampled LR ({1024 // UPSCALE_FACTOR}x{1024 // UPSCALE_FACTOR})")
axes[1].axis("off")

axes[2].imshow(sr_output)
axes[2].set_title("SRGAN Output")
axes[2].axis("off")

plt.tight_layout()
plt.show()
