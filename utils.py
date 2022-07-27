import torch
import numpy as np


def get_model_input_from_loader(bg_norm, mask, ped):
    return torch.cat(bg_norm, mask, ped, dim=0)


def bbox_from_mask_torch(ped_mask):
    y, x = torch.where(ped_mask != 0)
    xmin = torch.min(x)
    ymin = torch.min(y)
    xmax = torch.max(x)
    ymax = torch.max(y)

    return ymin, ymax, xmin, xmax


def enforce_bbox_size(im_shape, rmin, rmax, cmin, cmax, min_width=250, min_height=250):
    rpadding = 0
    cpadding = 0

    # compute ideal padding (equal on both sides)
    if rmax - rmin < min_height:
        rpadding = int((min_height - (rmax - rmin)) / 2)
    if cmax - cmin < min_width:
        cpadding = int((min_width - (cmax - cmin)) / 2)

    rmin -= rpadding
    rmax += rpadding
    cmin -= cpadding
    cmax += cpadding

    # account for negative indexing (subtract padding from one side and add it to other)
    if rmin < 0:
        rmax = min(rmax - rmin, im_shape[0])
        rmin = 0
    if cmin < 0:
        cmax = min(cmax - cmin, im_shape[1])
        cmin = 0
    if rmax >= im_shape[0]:
        rmin = max(0, rmin - (rmax - im_shape[0] + 1))
        rmax = im_shape[0] - 1
    if cmax >= im_shape[1]:
        cmin = max(0, cmin - (cmax - im_shape[1] + 1))
        cmax = im_shape[1] - 1

    return int(rmin), int(rmax), int(cmin), int(cmax)


def mask_from_rgb_threshold(rgb, arr_im):
    """Returns a boolean (binary) mask that is
    True for pixels in the image that are the
    same color as the rgb param.
    """
    mask = np.full(arr_im.shape[:2], False)
    idx = (arr_im[..., :3] == np.array(rgb)).all(axis=-1)
    mask[idx] = True
    return mask
