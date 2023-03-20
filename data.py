from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import os

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return transforms.Compose([
        transforms.RandomCrop(crop_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=90, interpolation = transforms.InterpolationMode.BICUBIC)], p=0.5)
    ])


def train_lr_transform(crop_size, upscale_factor):
    return transforms.Compose([
        transforms.Resize(crop_size // upscale_factor, interpolation=Image.Resampling.BICUBIC)
    ])


def _to_tensor(img):
    img = img.transpose(2, 0, 1)
    return torch.FloatTensor(img)


class TrainDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDataset, self).__init__()
        self.image_filenames = []
        for ds_path in dataset_dir:
            for x in os.listdir(ds_path):
                self.image_filenames.append(os.path.join(ds_path, x))

        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
        self.repeat = 10

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index % len(self.image_filenames)]).convert('RGB'))
        lr_image = self.lr_transform(hr_image)
        lr_image = np.asarray(lr_image)
        hr_image = np.asarray(hr_image)

        lr_image = _to_tensor(lr_image)
        hr_image = _to_tensor(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames) * self.repeat


class TrainWholeDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainWholeDataset, self).__init__()
        print("Loading the whole Training Dataset")
        self.image_lr = []
        self.image_hr = []
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        hr_transform = train_hr_transform(crop_size)
        lr_transform = train_lr_transform(crop_size, upscale_factor)
        rgb_range = 255
        self.repeat = 10

        for ds_path in dataset_dir:
            for x in os.listdir(ds_path):
                hr_image = hr_transform(Image.open(os.path.join(ds_path, x)).convert('RGB'))
                lr_image = lr_transform(hr_image)
                lr_image = np.asarray(lr_image)
                hr_image = np.asarray(hr_image)
                lr_image = _to_tensor(lr_image)
                hr_image = _to_tensor(hr_image)
                self.image_lr.append(lr_image)
                self.image_hr.append(hr_image)
        print("Done loading the whole Training Dataset")

    def __getitem__(self, index):
        return self.image_lr[index % len(self.image_lr)], self.image_hr[index % len(self.image_hr)]

    def __len__(self):
        return len(self.image_lr) * self.repeat


class ValDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(ValDataset, self).__init__()
        self.image_filenames = []
        for ds_path in dataset_dir:
            for x in os.listdir(ds_path):
                self.image_filenames.append(os.path.join(ds_path, x))
        
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.upscale_factor = upscale_factor
        self.lr_scale = transforms.Resize(self.crop_size // self.upscale_factor, interpolation=Image.Resampling.BICUBIC)

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')

        image_width = (hr_image.width // self.upscale_factor) * self.upscale_factor
        image_height = (hr_image.height // self.upscale_factor) * self.upscale_factor
        hr_scale = transforms.Resize((image_height, image_width), interpolation=Image.Resampling.BICUBIC)
        lr_scale = transforms.Resize((image_height // self.upscale_factor, image_width // self.upscale_factor), interpolation=Image.Resampling.BICUBIC)

        lr_image = lr_scale(hr_image)
        hr_image = hr_scale(hr_image)
        lr_image = np.asarray(lr_image)
        hr_image = np.asarray(hr_image)

        lr_image = _to_tensor(lr_image)
        hr_image = _to_tensor(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)