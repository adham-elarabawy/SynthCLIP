import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from network import AttU_Net
import csv


def main(config):
    # setup dataloaders
    train_loader = None

    # setup model
    unet = AttU_Net(img_ch=4, output_ch=3)
    optimizer = optim.Adam(
        list(unet.parameters()), config.lr, [config.beta1, config.beta2]
    )

    # prepare for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device)

    # train loop
    for epoch in range(config.num_epochs):
        unet.train(True)


if __name__ == "main":
    # fmt: off
    parser = argparse.ArgumentParser()

    # Model Hyper-parameters
    parser.add_argument("--image_size", type=int, default=224)

    # Training Hyper-parameters
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_epochs_decay", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument("--beta2", type=float, default=0.999)  # momentum2 in Adam
    # parser.add_argument('--augmentation_prob', type=float, default=0.4)

    # parser.add_argument('--log_step', type=int, default=2)
    # parser.add_argument('--val_step', type=int, default=2)

    # Data

    # if passing in directories of paired background/pedestrians/masks
    parser.add_argument("--bg_dir",type=str,help="Path to directory containing (paired) background images.")
    parser.add_argument("--ped_dir",type=str,help="Path to directory containing (paired) synthetic pedestrian images.")
    parser.add_argument("--mask_dir",type=str,help="Path to directory containing (paired) pedestrian masks.")
    parser.add_argument("--num_workers", type=int, default=8)

    # Data Loading
    parser.add_arugment("--flip", type=float, default=0.5, help="[0, 1] Probability that random inputs get horizontally flipped.")
    parser.add_argument("--bg_jitter_b", type=float, default=0.3, help="[0, 1] How much to randomly jitter background brightness.")
    parser.add_argument("--bg_jitter_c", type=float, default=0.3, help="[0, 1] How much to randomly jitter background contrast.")
    parser.add_argument("--bg_jitter_s", type=float, default=0.3, help="[0, 1] How much to randomly jitter background saturation.")
    parser.add_argument("--bg_jitter_h", type=float, default=0.3, help="[0, 1] How much to randomly jitter background hue.")


    # Miscellanious
    parser.add_argument("--mode", type=str, default="train", help="train,test")
    parser.add_argument("--model_type", type=str, default="AttU_Net", help="U_Net/R2U_Net/AttU_Net/R2AttU_Net")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--result_path", type=str, default="./result/")
    parser.add_argument("--cuda_idx", type=int, default=1)

    # fmt: on
    config = parser.parse_args()
