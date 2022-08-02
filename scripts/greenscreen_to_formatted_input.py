from PIL import Image
import numpy as np
from utils import *
import os
from os import listdir
from tqdm import tqdm

raw_dir = "/home/ubuntu/data/pandaset_synthclip/raw"
ped_dir = "/home/ubuntu/data/pandaset_synthclip/ped"
colormask_dir = "/home/ubuntu/data/pandaset_synthclip/colormask"

valid_count = 0
invalid_count = 0
confused_count = 0
confused = []
with tqdm(os.listdir(raw_dir)) as pbar:
    for image in pbar:
        if image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg"):

            pbar.set_postfix(
                {
                    "valid": valid_count,
                    "invalid": invalid_count,
                    "confused": confused_count,
                }
            )

            raw = Image.open(os.path.join(raw_dir, image)).convert("RGB")
            raw_arr = np.array(raw)[:, :, :3]
            colors = raw.getcolors(maxcolors=10000)
            colors = [color for count, color in colors]
            if len(colors) < 10:
                confused_count += 1
                confused.append(image)
                print(confused)
            if len(colors) == 2:
                invalid_count += 1
                continue
            else:
                valid_count += 1

            bg_mask = mask_from_rgb_target(
                (55, 255, 20), raw_arr, colors, dist_threshold=70
            )

            ped = raw_arr.copy()
            ped[bg_mask] = (0, 0, 0)

            colormask = raw_arr.copy()
            color = list(np.random.choice(range(1, 256), size=3))
            colormask[bg_mask] = (0, 0, 0)
            colormask[~bg_mask] = color

            # HWC
            flattened_colormask = np.sum(colormask, axis=2, keepdims=False)
            bbox_vals = bbox_from_mask_np(flattened_colormask)
            bbox_vals = enforce_bbox_size(
                flattened_colormask.shape, *bbox_vals, min_width=256, min_height=256
            )
            ymin, ymax, xmin, xmax = bbox_vals

            # if int(ymax) - int(ymin) > 256:
            #     invalid_count += 1
            #     continue
            # if int(xmax) - int(xmin) > 256:
            #     invalid_count += 1
            #     continue

            ped = ped[ymin:ymax, xmin:xmax]
            ped = Image.fromarray(ped)
            # ped.save(os.path.join(ped_dir, image))

            colormask = colormask[ymin:ymax, xmin:xmax]
            colormask = Image.fromarray(colormask)
            # colormask.save(os.path.join(colormask_dir, image))
print(confused)
