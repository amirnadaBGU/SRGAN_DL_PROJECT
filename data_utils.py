from os import listdir
from os.path import join

from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torchvision.transforms import InterpolationMode
from torchvision.transforms import transforms
import os, cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def val_hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif"])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, diffusion = False, val = False):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        if val:
            self.hr_transform = val_hr_transform(crop_size)
        else:
            self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
        self.diffusion = diffusion
        self.hr_sz = transforms.Resize((crop_size, crop_size), interpolation=InterpolationMode.BICUBIC)
        

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        if self.diffusion:
            return self.hr_sz(lr_image), hr_image #the hr_image is 'y' and low res image scaled to (128, 128) is our 'x'
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

class AlignedInferenceDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, crop_coords=None, diffusion=False):
        super().__init__()
        self.image_filenames = [join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.diffusion = diffusion
        self.crop_coords = crop_coords  # (i, j, h, w)

        self.to_tensor = TF.to_tensor
        self.downsample = lambda img: img.resize((crop_size // upscale_factor, crop_size // upscale_factor), Image.BICUBIC)
        self.upsample = lambda img: img.resize((crop_size, crop_size), Image.BICUBIC)

    def __getitem__(self, index):
        img = Image.open(self.image_filenames[index]).convert("RGB")

        # Use provided crop coordinates
        if self.crop_coords is None:
            i, j, h, w = TF.RandomCrop.get_params(img, output_size=(self.crop_size, self.crop_size))
            self.crop_coords = (i, j, h, w)
        else:
            i, j, h, w = self.crop_coords

        hr_crop = TF.crop(img, i, j, h, w)
        lr = self.downsample(hr_crop)

        if self.diffusion:
            lr = self.upsample(lr)

        return self.to_tensor(lr), self.to_tensor(hr_crop)

    def __len__(self):
        return len(self.image_filenames)


