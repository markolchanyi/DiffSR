import os
import sys
import torch
import numpy as np
import nibabel as nib
from torch.optim import Adam,AdamW
from torch.nn import L1Loss, MSELoss
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

sys.path.append('/autofs/space/nicc_003/users/olchanyi/DiffSR')
from ResSR.generators import hr_lr_random_res_generator
from ResSR.models_attention_diffm import AttnUNetPartialDiff
from ResSR.utils import mixed_loss, fibonacci_sphere, _mrtrix_real_sh_basis, evaluate_mrtrix_sh, laplacian_loss, load_volume

# Parameters
training_data_dir = '/autofs/space/nicc_005/users/olchanyi/DiffSR/training_data/sshell_sh/'
device_generator = 'cuda:1'
# device_generator = 'cpu'
device_training = 'cuda:0'
num_filters = 256
num_residual_blocks = 24
crop_size = 64
kernel_size = 3
prob_dropout=0.2,
prob_sh_rotate_deform=0.25,
use_global_residual = False
n_epochs = 2000
n_its_per_epoch = 10
output_directory = '/autofs/space/nicc_005/users/olchanyi/DiffSR/models_attentionunet/model_testing/'
#initial_model = '/autofs/space/nicc_005/users/olchanyi/DiffSR/models_attentionunet/model_v5/checkpoint_0457.pth'
initial_model = None
#noise_std_max=0.06
#noise_std_max_base=0.01
lowres_min=2
lowres_max=3.5
njobs = 64

#noise_schedule=[0.02,0.03,0.04,0.05,0.06]
noise_schedule=[0.025,0.025,0.025,0.025,0.025]

# @TODO any more than 5 and gpu mem overflows
batch_size = 1

# Create output directory if needed
if os.path.exists(output_directory) is False:
    os.mkdir(output_directory)

# Prepare generator
gen = hr_lr_random_res_generator(training_data_dir,
                                crop_size=crop_size,
                                device=device_generator,
                                noise_std_max=noise_schedule[0],
                                prob_dropout=prob_dropout,
                                prob_sh_rotate_deform=prob_sh_rotate_deform,
                                lowres_min=lowres_min,
                                lowres_max=lowres_max,
                                njobs=njobs)


model = AttnUNetPartialDiff().to(device_training)

LR=1e-3
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0)

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-6, last_epoch=-1)   # <- default -1
###

# Initialize scheduler
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-6)

l1_loss_fn = L1Loss()
l2_loss_fn = MSELoss()

# Load weights if provided
if initial_model is None:
    print('Starting from scratch')
    epoch_ini = 0
else:
    print('Loading weights from ' + initial_model)
    checkpoint = torch.load(initial_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for pg in optimizer.param_groups:          # reset learning-rate fields
        pg["lr"] = LR                # runtime LR
        pg["initial_lr"] = LR                # some schedulers use this

    epoch_ini = 1 + checkpoint['epoch']
# Initialize scheduler (after state dict loading)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2, eta_min=1e-6)


## ONLY FOR ANGULAR LOSS ##
# precompute dispersed directions to sample SH with
print("precommputing angular samples and basis tensor...")
sphere_dirs = fibonacci_sphere(samples=500, device=device_training)
#sh_basis_tensor = _mrtrix_real_sh_basis(sphere_dirs, lmax=6, device=device_training)
print("done")
####################################



# Train!
for j in range(n_epochs - epoch_ini):

    epoch = epoch_ini + j

    print('Epoch ' + str(epoch+1) + ' of ' + str(n_epochs))
    print('Current LR: ', optimizer.param_groups[0]['lr'])
    loss_epoch_acc = 0.0

    loss_epoch_acc_l1 = 0.0
    loss_epoch_acc_l2 = 0.0
    loss_epoch_acc_ang = 0.0

    if epoch % 20 == 0 and  epoch > 5:
        os.makedirs("./tmp_epoch_output", exist_ok=True)
        print("saving a batch for checking...")
        #if input_batch in locals() and target_batch in locals() and pred_batch in locals():
        print("found tensors!")
        input_npy = input_batch[0,...].detach().cpu().numpy()
        target_npy = target_batch[0,...].detach().cpu().numpy()
        pred_npy = pred_batch[0,...].detach().cpu().numpy()

        input_npy = np.transpose(input_npy, (1, 2, 3, 0))
        target_npy = np.transpose(target_npy, (1, 2, 3, 0))
        pred_npy = np.transpose(pred_npy, (1, 2, 3, 0))

        nib.save(nib.Nifti1Image(input_npy, affine=np.eye(4)), './tmp_epoch_output/input.nii.gz')
        nib.save(nib.Nifti1Image(target_npy, affine=np.eye(4)), './tmp_epoch_output/target.nii.gz')
        nib.save(nib.Nifti1Image(pred_npy, affine=np.eye(4)), './tmp_epoch_output/pred.nii.gz')
        #else:
        #    print("no active batches found...skipping")

    for iteration in range(n_its_per_epoch):

        inputs_list = []
        targets_list = []
        for _ in range(batch_size):
            single_input, aff = load_volume('./test_sample/input.nii.gz')  # Load the SH coeffs
            single_input = single_input.astype(float)
            single_target, aff = load_volume('./test_sample/target.nii.gz')  # Load the SH coeffs
            single_target = single_target.astype(float)

            single_input = torch.tensor(single_input, device=device_generator).float()
            single_target = torch.tensor(single_target, device=device_generator).float()
            #single_input = single_input.permute(3, 0, 1, 2)
            #single_target = single_target.permute(3, 0, 1, 2)

            #print("input shape: ", single_input.shape)
            #print("target shape: ", single_target.shape)
            inputs_list.append(single_input)
            targets_list.append(single_target)
            inputs_list.append(single_input)
            targets_list.append(single_target)
            inputs_list.append(single_input)
            targets_list.append(single_target)
            inputs_list.append(single_input)
            targets_list.append(single_target)

        input_batch = torch.stack(inputs_list, dim=0).to(device_training)
        target_batch = torch.stack(targets_list, dim=0).to(device_training)


        optimizer.zero_grad()
        #pred_batch = model(input_batch)
        pred_batch = model(input_batch)
        loss, l2_loss = mixed_loss(pred_batch, target_batch)

        if not torch.isfinite(loss):
            print(f"[step {iteration}]  ...bad batch...skipping")
            # clear gen workers so corrupted tensors are freed
            del input_batch, target_batch, pred_batch
            torch.cuda.empty_cache()
            continue          # <-- discard this iteration

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        loss_epoch_acc = loss_epoch_acc + loss.detach().cpu().numpy()
        cumul_loss_epoch = loss_epoch_acc / (iteration + 1)

        #loss_epoch_acc_l1 = loss_epoch_acc_l1 + l1_loss.detach().cpu().numpy()
        #cumul_loss_epoch_l1 = loss_epoch_acc_l1 / (iteration + 1)

        loss_epoch_acc_l2 = loss_epoch_acc_l2 + l2_loss.detach().cpu().numpy()
        cumul_loss_epoch_l2 = loss_epoch_acc_l2 / (iteration + 1)

        #loss_epoch_acc_ang = loss_epoch_acc_ang + ang_loss.detach().cpu().numpy()
        #cumul_loss_epoch_ang = loss_epoch_acc_ang / (iteration + 1)

        #print('   Iteration ' + str(1+iteration) + ' of ' + str(n_its_per_epoch) + ', tot loss = ' + str(cumul_loss_epoch) + ', laplacian loss: ' + str(cumul_loss_epoch_ang), end="\r")
        print('   Iteration ' + str(1+iteration) + ' of ' + str(n_its_per_epoch) + ', tot loss = ' + str(cumul_loss_epoch) + '  l2 loss: ' + str(cumul_loss_epoch_l2), end="\r")
    print('\n   End of epoch ' + str(epoch+1) + '; saving model... \n')

    #scheduler.step(cumul_loss_epoch)
    scheduler.step()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': cumul_loss_epoch,
    }, '%s/checkpoint_%.4d.pth' % (output_directory, 1+epoch))

print('Training complete!')
