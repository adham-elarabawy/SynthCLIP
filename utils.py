import torch


def get_model_input_from_loader(bg_norm, mask, ped):
    return torch.cat(bg_norm, mask, ped, dim=0)
