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

        self.total = len(self.bg_paths)

        # assert length of paths is the same
        assert (
            self.total
            == len(self.bg_paths)
            == len(self.ped_paths)
            == len(self.mask_paths)
        )

        # assert all paired paths have same filenames
        assert all(
            [
                os.path.basename(self.bg_paths[i])
                == os.path.basename(self.ped_paths[i])
                == os.path.basename(self.mask_paths[i])
                for i in range(self.total)
            ]
        )

        self.flip = config.flip  # prob of horizontally flipping inputs
        self.bg_jitter_b = config.bg_jitter_b  # how much to jitter bg brightness
        self.bg_jitter_c = config.bg_jitter_c  # how much to jitter bg contrast
        self.bg_jitter_s = config.bg_jitter_s  # how much to jitter bg saturation
        self.bg_jitter_h = config.bg_jitter_h  # how much to jitter bg hue

        # self.image_size = image_size
        print(f"Successfully initialized dataset of size: {self.total}.")

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        # get pipeline input paths
        bg_path = self.bg_paths[index]
        ped_path = self.ped_paths[index]
        mask_path = self.mask_paths[index]

        # load pipeline inputs as images
        im_bg = Image.open(bg_path).convert("RGB")
        im_ped = Image.open(ped_path).convert("RGB")
        im_mask = Image.open(mask_path).convert("RGB")

        # construct base transform
        base_transform = []
        base_transform.append(T.PILToTensor())
        base_transform.append(T.ConvertImageDtype(dtype=torch.float))

        if random.random() < self.flip:
            base_transform.append(T.RandomHorizontalFlip(1))

        # construct bg-specific transform
        bg_transform = base_transform.copy()
        bg_transform.append(
            T.ColorJitter(
                brightness=self.bg_jitter_b,
                contrast=self.bg_jitter_c,
                saturation=self.bg_jitter_s,
                hue=self.bg_jitter_h,
            )
        )

        # construct ped-specific transform
        ped_transform = base_transform.copy()

        # construct mask-specific transform
        mask_transform = base_transform.copy()

        def squish_colormask(colormask):
            colormask = torch.sum(
                colormask, dim=0, keepdim=True
            )  # average across rgb channels
            obj_ids = torch.unique(colormask)  # get unique values in mask
            obj_ids = obj_ids[obj_ids != 0]
            masks = colormask == obj_ids[:, None, None]
            bool_mask = torch.any(masks, dim=0, keepdim=True)
            out = torch.zeros_like(bool_mask, dtype=torch.float)
            out[bool_mask] = 1
            return out

        mask_transform.append(T.Lambda(squish_colormask))

        # compose transforms
        bg_transform = T.Compose(bg_transform)
        bg_norm_transform = T.Compose(
            [T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )
        ped_transform = T.Compose(ped_transform)
        mask_transform = T.Compose(mask_transform)

        # apply transforms
        bg = bg_transform(im_bg)
        bg_norm = bg_norm_transform(bg)
        ped = ped_transform(im_ped)
        mask = mask_transform(im_mask)

        return bg, bg_norm, mask, ped

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