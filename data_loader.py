from configparser import Interpolation
import os
import random
from random import shuffle
from sqlite3 import SQLITE_RECURSIVE
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import random
from utils import *
import albumentations as A
import albumentations.pytorch
import cv2


class ImageFolder(data.Dataset):
    def __init__(self, config):
        """Initializes image paths and preprocessing module."""

        self.bg_dir = config.bg_dir
        self.ped_dir = config.ped_dir
        self.mask_dir = config.mask_dir

        # fmt: off
        self.bg_paths = sorted(list(map(lambda x: os.path.join(self.bg_dir, x), os.listdir(self.bg_dir))))
        self.ped_paths = sorted(list(map(lambda x: os.path.join(self.ped_dir, x), os.listdir(self.ped_dir))))
        self.mask_paths = sorted(list(map(lambda x: os.path.join(self.mask_dir, x), os.listdir(self.mask_dir))))
        # fmt: on

        self.total = len(self.ped_paths)

        self.flip = config.flip  # prob of horizontally flipping inputs
        self.bg_jitter_b = config.bg_jitter_b  # how much to jitter bg brightness
        self.bg_jitter_c = config.bg_jitter_c  # how much to jitter bg contrast
        self.bg_jitter_s = config.bg_jitter_s  # how much to jitter bg saturation
        self.bg_jitter_h = config.bg_jitter_h  # how much to jitter bg hue

        self.ped_scale_min = config.ped_scale_min
        self.ped_scale_max = config.ped_scale_max

        self.bg_size = config.bg_load_size

        print(f"Successfully initialized dataset of size: {self.total}.")

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        # get pipeline input paths
        bg_path = random.choice(self.bg_paths)
        ped_path = self.ped_paths[index]
        mask_path = self.mask_paths[index]

        # get basename
        basename = os.path.basename(bg_path).split(".")[0]

        # load pipeline inputs as images
        im_bg = cv2.imread(bg_path)
        im_ped = cv2.imread(ped_path)
        im_mask = cv2.imread(mask_path)

        im_bg = cv2.cvtColor(im_bg, cv2.COLOR_BGR2RGB)
        im_ped = cv2.cvtColor(im_ped, cv2.COLOR_BGR2RGB)
        im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2RGB)

        bg_transform = []
        bg_transform.append(
            A.RandomResizedCrop(
                height=self.bg_size,
                width=self.bg_size,
                scale=(0.5, 1),
                ratio=(0.75, 1.33),
                p=1,
            )
        )
        bg_transform.append(A.HorizontalFlip(self.flip))
        bg_transform.append(
            A.ColorJitter(
                brightness=self.bg_jitter_b,
                contrast=self.bg_jitter_c,
                saturation=self.bg_jitter_s,
                hue=self.bg_jitter_h,
                p=0.5,
            )
        )
        # bg_transform.append(A.pytorch.ToTensorV2())

        ped_transform = []
        ped_scale = random.uniform(self.ped_scale_min, self.ped_scale_max)
        if ped_scale <= 1:
            new_size = int((1 / ped_scale) * max(*im_ped.shape[:2]))
            ped_transform.append(
                A.PadIfNeeded(
                    min_height=new_size,
                    min_width=new_size,
                    position="random",
                    border_mode=cv2.BORDER_REPLICATE,
                )
            )
        elif ped_scale > 1:
            new_size = int((1 / ped_scale) * max(*im_ped.shape[:2]))
            ped_transform.append(A.RandomCrop(height=new_size, width=new_size))

        ped_transform.append(A.Resize(height=self.bg_size, width=self.bg_size))
        ped_transform.append(A.HorizontalFlip(self.flip))
        ped_transform.append(
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0,
                p=1.0,
            )
        )

        norm_transform = []
        norm_transform.append(
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0,
                p=1.0,
            )
        )

        torch_transform = []
        torch_transform.append(A.pytorch.ToTensorV2())

        #     def squish_colormask(colormask):
        # colormask = torch.sum(
        #     colormask, dim=0, keepdim=True
        # )  # average across rgb channels
        # obj_ids = torch.unique(colormask)  # get unique values in mask
        # obj_ids = obj_ids[obj_ids != 0]
        # masks = colormask == obj_ids[:, None, None]
        # bool_mask = torch.any(masks, dim=0, keepdim=True)
        # out = torch.zeros_like(bool_mask, dtype=torch.float)
        # out[bool_mask] = 1
        # return out

        # compose transforms
        bg_transform = A.Compose(bg_transform)
        ped_transform = A.Compose(ped_transform)
        norm_transform = A.Compose(norm_transform)
        torch_transform = A.Compose(torch_transform)

        # apply transforms
        bg = bg_transform(image=im_bg)["image"]  # unnormalized, arr
        bg_norm = norm_transform(image=bg)["image"]  # normalized, arr
        ped_out = ped_transform(image=im_ped, mask=im_mask)
        ped = ped_out["image"]  # normalized, arr
        colormask = ped_out["mask"]  # unnormalized, arr

        # get distinct masks from colormask
        colors = get_unique_colors(colormask)
        colors = colors[np.sum(colors, axis=1) != 0]
        ped_masks = torch.zeros((len(colors), colormask.shape[0], colormask.shape[1]))
        combined_mask = torch.zeros((1, colormask.shape[0], colormask.shape[1]))
        for i, color in enumerate(colors):
            bool_mask = mask_from_rgb_threshold(color, np.array(colormask)[:, :, :3])
            bool_mask = torch.from_numpy(bool_mask)
            ped_mask = torch.zeros_like(bool_mask, dtype=torch.float)
            ped_mask[bool_mask] = 1
            combined_mask[0, bool_mask] = 1
            ped_masks[i] = ped_mask

        bg = torch_transform(image=bg)["image"]  # unnormalized tensor
        bg_norm = torch_transform(image=bg_norm)["image"]  # normalized tensor
        ped = torch_transform(image=ped)["image"]  # normalized tensor
        combined_mask = combined_mask  # unnormalized tensor
        all_masks = ped_masks  # unnormalized tensor

        return bg, bg_norm, ped, combined_mask, all_masks

    def __len__(self):
        """Returns the total number of files."""
        return self.total


def get_loader(
    config,
    batch_size,
    num_workers=2,
    mode="train",
):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(config)
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return data_loader
