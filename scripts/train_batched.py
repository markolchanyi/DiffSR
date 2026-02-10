import os
import sys
import math
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

# update if need be
sys.path.append("/autofs/space/nicc_003/users/olchanyi/DiffSR_testing")

from ResSR.generators import hr_lr_random_res_generator
from ResSR.models import S2UNetGlobalL2
from ResSR.utils import (fibonacci_sphere, forward_blur_down_up, sh_angular_mse_loss)


writer = SummaryWriter("./runs_monitor")

# obviously update
training_data_dir = "/autofs/space/nicc_005/users/olchanyi/DiffSR/training_data/sshell_sh/"
output_directory  = "/autofs/space/nicc_005/users/olchanyi/DiffSR/models_attentionunet/model_test_l2_graph_v1/"

device_generator = "cuda:1" if torch.cuda.is_available() else "cpu"
device_training  = device_generator

os.makedirs(output_directory, exist_ok=True)

####################
crop_size = 32
n_epochs = 5000
n_its_per_epoch = 20
batch_size = 4
lowres_min = 1.5
lowres_max = 4.0
njobs = 64
#####################

initial_model = None 

# schedules
noise_schedule = [0.025, 0.03, 0.03, 0.035, 0.04, 0.045, 0.045, 0.05]
lowres_max_schedule = [2.5, 2.5,  2.5, 2.75, 3.0, 3.25, 3.5, 4.0]
rotate_prob_schedule = [0.0, 0.05, 0.10, 0.15, 0.20, 0.2, 0.2, 0.20]
rotate_bounds_schedule = [5, 5, 5, 10, 15, 20, 20, 25]
patch_schedule = [0.0]
dropout_schedule = [0.0]
ang_subsample_schedule = [0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2]

# loss weights
l1_mult = 1000.0
l2_mult = 500.0
ang_mult = 100.0
lambda_dc = 250.0

# initial generator
gen = hr_lr_random_res_generator(
    training_data_dir,
    crop_size=crop_size,
    prob_rotate=rotate_prob_schedule[0],
    rotation_bounds=rotate_bounds_schedule[0],
    prob_patch=patch_schedule[0],
    prob_dropout=dropout_schedule[0],
    device=device_generator,
    noise_std_max=noise_schedule[0],
    lowres_min=lowres_min,
    lowres_max=lowres_max_schedule[0],
    njobs=njobs,
    return_params=True,
    debug_dir=None,
    debug_ang_subsample=False,
    prob_ang_subsample=ang_subsample_schedule[0],
)

LR = 1e-4
model = S2UNetGlobalL2().to(device_training)
optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

warm = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100)
cos = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30_000, eta_min=2e-6)
scheduler = lr_scheduler.SequentialLR(optimizer, [warm, cos], [600])

epoch_ini = 0
if initial_model is not None:
    print("Loading weights from " + initial_model)
    checkpoint = torch.load(initial_model, map_location=device_training)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_ini = 1 + checkpoint.get("epoch", 0)
else:
    print("Starting from scratch")


directions = fibonacci_sphere(48, device=device_training)


l1_loss_fn = nn.L1Loss()

# training loop
global_step = 0
fresh_step  = 0

##################################################
for j in range(n_epochs - epoch_ini):
    model.train()
    epoch = epoch_ini + j

    print(f"\nEpoch {epoch+1} / {n_epochs}")
    print("Current LR:", optimizer.param_groups[0]["lr"])

    loss_epoch_acc    = 0.0
    loss_epoch_acc_l1 = 0.0

    # update gen every 50 epochs
    if (epoch % 50 == 0 and epoch > 10) or (fresh_step == 0):
        idx = epoch // 50

        new_noise_std   = noise_schedule[min(idx, len(noise_schedule) - 1)]
        new_lowres_max  = lowres_max_schedule[min(idx, len(lowres_max_schedule) - 1)]
        new_rotate_prob = rotate_prob_schedule[min(idx, len(rotate_prob_schedule) - 1)]
        new_rotate_bounds = rotate_bounds_schedule[min(idx, len(rotate_bounds_schedule) - 1)]
        new_patch_prob  = patch_schedule[min(idx, len(patch_schedule) - 1)]
        new_dropout_prob= dropout_schedule[min(idx, len(dropout_schedule) - 1)]
        new_ang_prob    = ang_subsample_schedule[min(idx, len(ang_subsample_schedule) - 1)]

        gen = hr_lr_random_res_generator(
            training_data_dir,
            crop_size=crop_size,
            prob_rotate=new_rotate_prob,
            rotation_bounds=new_rotate_bounds,
            prob_patch=new_patch_prob,
            prob_dropout=new_dropout_prob,
            device=device_generator,
            noise_std_max=new_noise_std,
            lowres_min=lowres_min,
            lowres_max=new_lowres_max,
            njobs=njobs,
            return_params=True,
            debug_dir=None,
            debug_ang_subsample=False,
            prob_ang_subsample=new_ang_prob,
        )

        print("Updated noise std to:", new_noise_std)
        print("Updated lowres max to:", new_lowres_max)
        print("Updated rotation prob to:", new_rotate_prob)
        print("Updated rotation bounds to:", new_rotate_bounds)
        print("Updated dropout prob to:", new_dropout_prob)
        print("Updated angular subsample prob to:", new_ang_prob)

        fresh_step += 1


    for iteration in range(n_its_per_epoch):
        inputs_list  = []
        targets_list = []
        ratios_list  = []

        # draw batch from generator
        for _ in range(batch_size):
            single_input, single_target, single_ratios = next(gen)
            inputs_list.append(single_input)
            targets_list.append(single_target)
            ratios_list.append(single_ratios)

        input_batch  = torch.stack(inputs_list,  dim=0).to(device_training)
        target_batch = torch.stack(targets_list, dim=0).to(device_training)
        ratios_batch = torch.stack(ratios_list,  dim=0).to(device_training)

        optimizer.zero_grad()


        pred_batch = model(input_batch)

 
        low_diff = pred_batch[:, 0:2, ...] - target_batch[:, 0:2, ...]  # (B,2,D,H,W)
        sh_diff  = pred_batch[:, 2:,  ...] - target_batch[:, 2:,  ...]  # (B,5,D,H,W)

        # L2 on b0 + l0
        l2_loss_low = (low_diff ** 2).mean()

        # L1 on SH
        l1_loss_sh = sh_diff.abs().mean()

        # angular loss
        ang_loss = sh_angular_mse_loss(
            pred_batch,
            target_batch,
            directions=directions,
            lmax=2,
            gamma=10.0,
            b0_index=0,
        )

        # data consistency loss
        input_pred = forward_blur_down_up(pred_batch, ratios_batch)

        dc_low  = (input_pred[:, 0:2, ...] - input_batch[:, 0:2, ...]).pow(2).mean()
        dc_high = (input_pred[:, 2:,  ...] - input_batch[:, 2:,  ...]).pow(2).mean()

        alpha_low  = 0.3   # trust DC less for low-b/l0
        alpha_high = 1.0

        dc_loss = (alpha_low * dc_low) + (alpha_high * dc_high)

        # all loss
        total_loss = (
            l1_mult * l1_loss_sh
            + l2_mult * l2_loss_low
            + ang_mult * ang_loss
            + lambda_dc * dc_loss
        )

        if not torch.isfinite(total_loss):
            print(
                f"[epoch {epoch} iter {iteration}] non-finite loss "
                f"(total={total_loss}, l2 (lowb/l0)={l2_loss_low}, "
                f"l2(l>=2)={l1_loss_sh}, ang={ang_loss}, dc={dc_loss}), skipping batch"
            )
            del total_loss, l1_loss_sh, l2_loss_low, ang_loss, dc_loss, pred_batch, input_batch, target_batch, input_pred
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()


        loss_epoch_acc += total_loss.detach().cpu().item()
        loss_epoch_acc_l1 += (l1_mult * l1_loss_sh).detach().cpu().item()

        cumul_loss_epoch = loss_epoch_acc / (iteration + 1)
        cumul_loss_epoch_l1 = loss_epoch_acc_l1 / (iteration + 1)

        L1_scaled = l1_mult * l1_loss_sh.item()
        L2_scaled = l2_mult * l2_loss_low.item()
        Ang_scaled = ang_mult * ang_loss.item()
        DC_scaled  = lambda_dc * dc_loss.item()

        print(
            f"Iter {iteration+1}/{n_its_per_epoch} | "
            f"Total: {total_loss.item():.4f} | "
            f"L2 (lowb/l0): {l2_loss_low.item():.4f} (x{l2_mult} = {L2_scaled:.2f}) | "
            f"L2(l>=2): {l1_loss_sh.item():.4f} (x{l1_mult} = {L1_scaled:.2f}) | "
            f"Angular: {ang_loss.item():.4f} (x{ang_mult} = {Ang_scaled:.2f}) | "
            f"DC: {dc_loss.item():.4f} (x{lambda_dc} = {DC_scaled:.2f})",
            end="\r",
        )


        writer.add_scalar("loss/total",   total_loss.item(),   global_step)
        writer.add_scalar("loss/l1",      l1_loss_sh.item(),   global_step)
        writer.add_scalar("loss/l2",      l2_loss_low.item(),  global_step)
        writer.add_scalar("loss/angular", ang_loss.item(),     global_step)
        writer.add_scalar("loss/dc",      dc_loss.item(),      global_step)

        global_step += 1

        # dump batch to look at
        if epoch % 20 == 0 and iteration == 5:
            os.makedirs("./tmp_epoch_output", exist_ok=True)
            print("\nSaving a batch for checking...")

            input_npy = input_batch[0, ...].detach().permute(1, 2, 3, 0).cpu().numpy()
            input_pred_npy = input_pred[0, ...].detach().permute(1, 2, 3, 0).cpu().numpy()
            target_npy = target_batch[0, ...].detach().permute(1, 2, 3, 0).cpu().numpy()
            pred_npy = pred_batch[0, ...].detach().permute(1, 2, 3, 0).cpu().numpy()

            nib.save(nib.Nifti1Image(input_npy, affine=np.eye(4)), "./tmp_epoch_output/input.nii.gz")
            nib.save(nib.Nifti1Image(input_pred_npy, affine=np.eye(4)), "./tmp_epoch_output/input_pred.nii.gz")
            nib.save(nib.Nifti1Image(target_npy, affine=np.eye(4)), "./tmp_epoch_output/target.nii.gz")
            nib.save(nib.Nifti1Image(pred_npy, affine=np.eye(4)), "./tmp_epoch_output/pred.nii.gz")

    print(f"\nEnd of epoch {epoch+1}; mean total loss: {cumul_loss_epoch:.4f}")

    fresh_step += 1
    if scheduler is not None:
        scheduler.step()

    # checkpoint
    #####################################
    if (epoch + 1) % 5 == 0:
        ckpt_path = os.path.join(output_directory, f"checkpoint_{epoch+1:04d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": cumul_loss_epoch,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

print("Training complete!")
writer.close()

