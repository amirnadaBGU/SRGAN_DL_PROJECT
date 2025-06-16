# test_image_runner.py

import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator
import gc
import torch


def run_inference(upscale_factor, test_mode, image_name, model_name):
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if test_mode == 'GPU' and torch.cuda.is_available() else 'cpu')
    model = Generator(upscale_factor).to(device).eval()
    model.load_state_dict(torch.load('epochs/' + model_name, map_location=device))

    image = Image.open(image_name)
    MAX_RES = 512
    if image.width > MAX_RES or image.height > MAX_RES:
        image = image.resize((MAX_RES, MAX_RES), Image.BICUBIC)

    image = ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.perf_counter()
        out = model(image)
        elapsed = time.perf_counter() - start
        print(f'Inference took {elapsed:.2f} seconds')

    out_img = ToPILImage()(out[0].cpu())
    out_img.save(f'out_srf_{upscale_factor}_{os.path.basename(image_name)}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Single Image')
    parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
    parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
    parser.add_argument('--image_name',default='data/DIV2K_valid_HR/0804.png' ,type=str, help='test low resolution image name')
    parser.add_argument('--model_name', default='netG_epoch_8_10.pth', type=str, help='generator model epoch name')
    opt = parser.parse_args()

    run_inference(opt.upscale_factor, opt.test_mode, opt.image_name, opt.model_name)
