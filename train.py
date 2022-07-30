import os
import numpy as np
import time
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from network import AttU_Net
from data_loader import get_loader
from utils import get_model_input_from_loader
import argparse
from utils import *
import clip as clip_api
import mlflow


def main(config):
    # setup config params

    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(uri=config.mlflow_tracking_uri)
    else:
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    mlflow.set_experiment(experiment_name="synthclip")

    with mlflow.start_run() as run:

        # log run parameters to mlflow
        mlflow.log_params(vars(config))

        # setup logging dir
        run_dir = os.path.join("checkpoints", config.name)
        os.makedirs(run_dir, exist_ok=False)
        img_dir = os.path.join(run_dir, "img")
        os.makedirs(img_dir, exist_ok=False)

        # setup dataloaders
        train_loader = get_loader(config, config.batch_size)

        # setup model
        unet = AttU_Net(img_ch=7, output_ch=3)
        optimizer = optim.Adam(
            list(unet.parameters()),
            lr=config.lr,
            betas=[config.beta1, config.beta2],
            weight_decay=config.weight_decay,
        )
        scheduler = ExponentialLR(optimizer, gamma=config.lr_scheduler_gamma)

        # setup criterion
        if config.bg_dc_criterion == "L1":
            bg_dc_criterion = torch.nn.L1Loss(reduction="none")
        else:
            bg_dc_criterion = torch.nn.MSELoss(reduction="none")
        clip_sim_criterion = torch.nn.CosineSimilarity()

        # prepare for computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet.to(device)

        # setup CLIP model
        clip, clip_preprocess = clip_api.load(config.clip_variant, device=device)

        # train loop
        for epoch in range(config.num_epochs):
            unet.train(True)
            for curr_iter, (
                basename,
                bg,
                bg_norm,
                ped,
                combined_mask,
                all_masks,
            ) in enumerate(train_loader):
                total_iter = epoch * len(train_loader) + curr_iter

                # prepare input for model
                model_input = get_model_input_from_loader(bg_norm, combined_mask, ped)
                model_input = model_input.to(device)

                # forward pass
                out = unet(model_input)

                # compute losses
                # background data consistency loss
                bg_dc_loss = bg_dc_criterion(out, bg) * (~combined_mask.bool()).float()
                bg_dc_loss = torch.mean(bg_dc_loss)

                # CLIP similarity loss
                clip_sim_loss = 0
                for iter in range(all_masks.shape[0]):
                    iter_start_time = time.time()
                    for ped_mask in all_masks[iter]:
                        # get bounding box around target object
                        bbox_vals = bbox_from_mask_torch(ped_mask)
                        # expand bounding box to desired patch size
                        bbox_vals = enforce_bbox_size(
                            ped_mask.shape,
                            *bbox_vals,
                            min_width=config.clip_patch_size,
                            min_height=config.clip_patch_size,
                        )
                        ymin, ymax, xmin, xmax = bbox_vals

                        # TODO: Add random transforms to compute average CLIP similarity for more robust convergence.
                        # TODO: Add pedestrian background change options

                        # encode model output patch as CLIP embedding
                        merged_patch = out[iter, ymin:ymax, xmin:xmax]
                        enc_merged_patch = clip_preprocess(merged_patch)
                        enc_merged_patch = enc_merged_patch.unsqueeze(0).to(device)
                        enc_merged_patch = clip.encode_image(enc_merged_patch)

                        # encode pedestrian patch as CLIP embedding
                        synth_patch = ped[iter, ymin:ymax, xmin:xmax]
                        enc_synth_patch = clip_preprocess(synth_patch)
                        enc_synth_patch = enc_synth_patch.unsqueeze(0).to(device)
                        enc_synth_patch = clip.encode_image(enc_synth_patch)

                        clip_sim_loss += 1 - clip_sim_criterion(
                            enc_merged_patch, enc_synth_patch
                        )
                # normalize CLIP similarity loss to number of patches
                clip_sim_loss /= all_masks.shape[1]

                # compute full loss
                loss = bg_dc_loss + clip_sim_loss

                # backprop + optimize
                unet.zero_grad()
                clip.zero_grad()
                loss.backward()
                optimizer.step()

                # print losses and log to dashboards
                iter_end_time = time.time()
                if total_iter % config.log_freq == 0:
                    message = f"[epoch {epoch}|curr_iter {curr_iter}|iter time {iter_end_time - iter_start_time}] - loss {loss}|bg_dc_loss {bg_dc_loss}|clip_sim_loss {clip_sim_loss}"
                    print(message)

                    mlflow.log_metric("bg_dc_loss", bg_dc_loss, step=total_iter)
                    mlflow.log_metric("clip_im_loss", clip_sim_loss, step=total_iter)
                    mlflow.log_metric("loss", loss, step=total_iter)

                # save sample train images and TODO: upload to dashboards
                if total_iter % config.vis_freq == 0:
                    im_out = F.to_pil_image(out[0, :, :, :])
                    im_bg = F.to_pil_image(bg[0, :, :, :])
                    im_mask = F.to_pil_image(combined_mask[0, :, :, :])
                    im_ped = F.to_pil_image(ped[0, :, :, :])

                    # fmt: off
                    im_out.save(os.path.join(img_dir, f"epoch{epoch}_" + basename[0] + "_out.png"))
                    im_bg.save(os.path.join(img_dir, f"epoch{epoch}_" + basename[0] + "_bg.png"))
                    im_mask.save(os.path.join(img_dir, f"epoch{epoch}_" + basename[0] + "_mask.png"))
                    im_ped.save(os.path.join(img_dir, f"epoch{epoch}_" + basename[0] + "_ped.png"))
                    # fmt: on

            # update learning rate
            scheduler.step()
            # save model checkpoints and TODO: upload to mlflow
            if epoch % config.save_freq == 0:
                state = {
                    "epoch": epoch,
                    "model_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.stat_dict(),
                    "loss": loss,
                }
                torch.save(state, os.path.join(run_dir, f"epoch{epoch}_model.pth"))


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()

    # Model Hyper-parameters
    # parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--clip_variant", type=str, default="ViT-B/32")

    # Training Hyper-parameters
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_epochs_decay", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument("--beta2", type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.99)

    # Custom ML hyper-parameters
    parser.add_argument("--bg_load_size", type=int, default=256)
    parser.add_argument("--clip_patch_size", type=int, default=128)
    parser.add_argument("--bg_dc_criterion", type=str, default="L2", help="L1|L2")

    # Data

    # if passing in directories of paired background/pedestrians/masks
    parser.add_argument("--bg_dir", type=str, required=True, help="Path to directory containing (paired) background images.")
    parser.add_argument("--ped_dir", type=str, required=True, help="Path to directory containing (paired) synthetic pedestrian images.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to directory containing (paired) pedestrian masks.")
    parser.add_argument("--num_workers", type=int, default=8)

    # Data Loading
    parser.add_argument("--flip", type=float, default=0.5, help="[0, 1] Probability that random inputs get horizontally flipped.")
    parser.add_argument("--bg_jitter_b", type=float, default=0.3, help="[0, 1] How much to randomly jitter background brightness.")
    parser.add_argument("--bg_jitter_c", type=float, default=0.3, help="[0, 1] How much to randomly jitter background contrast.")
    parser.add_argument("--bg_jitter_s", type=float, default=0.3, help="[0, 1] How much to randomly jitter background saturation.")
    parser.add_argument("--bg_jitter_h", type=float, default=0.15, help="[0, 1] How much to randomly jitter background hue.")


    # Miscellanious
    parser.add_argument("--mode", type=str, default="train", help="train,test")
    parser.add_argument("--model_type", type=str, default="AttU_Net", help="U_Net/R2U_Net/AttU_Net/R2AttU_Net")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--result_path", type=str, default="./result/")
    parser.add_argument("--cuda_idx", type=int, default=1)

    # Logging params 
    parser.add_argument("--log_freq", type=int, default=100) # iter
    parser.add_argument("--vis_freq", type=int, default=100) # iter
    parser.add_argument("--save_freq", type=int, default=5) # epoch

    parser.add_argument("--mlflow_tracking_uri", type=str, default=None)

    # fmt: on
    config = parser.parse_args()

    main(config)
