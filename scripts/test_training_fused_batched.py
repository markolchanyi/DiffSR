import os
import sys
import torch
import math
import numpy as np
import nibabel as nib
from torch.optim import Adam,AdamW
from torch.nn import L1Loss, MSELoss
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append('/autofs/space/nicc_003/users/olchanyi/DiffSR')
from ResSR.generators import hr_lr_random_res_generator
from ResSR.models_s2equivtrans_unet import S2UNetGlobal
from ResSR.utils import mixed_loss, fibonacci_sphere, _mrtrix_real_sh_basis, evaluate_mrtrix_sh, laplacian_loss, load_volume

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./runs_monitor")

# Parameters
training_data_dir = '/autofs/space/nicc_005/users/olchanyi/DiffSR/training_data/sshell_sh/'
device_generator = 'cuda:0'
device_training = 'cuda:1'
crop_size = 64
n_epochs = 5000
n_its_per_epoch = 20
output_directory = '/autofs/space/nicc_005/users/olchanyi/DiffSR/models_attentionunet/model_test_v2/'
initial_model = None
lowres_min=1.3
lowres_max=3.5
njobs = 64

noise_schedule=[0.025,0.03,0.03,0.035,0.04,0.045,0.05,0.05]
lowres_max_schedule=[2.5,2.5,2.5,2.75,3,3.25,3.5,3.5]
rotate_prob_schedule=[0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.3]
rotate_bounds_schedule=[5,5,5,10,15,20,20,20]
patch_schedule=[0.0,0.05,0.1,0.15,0.5,0.3,0.35,0.4]
dropout_schedule=[0.0,0.05,0.05,0.05,0.1,0.1,0.15,0.15]

# @TODO any more than 5 and gpu mem overflows
batch_size = 4

# Create output directory if needed
if os.path.exists(output_directory) is False:
    os.mkdir(output_directory)

# Prepare generator
gen = hr_lr_random_res_generator(training_data_dir,
                                crop_size=crop_size,
                                prob_rotate=rotate_prob_schedule[0],
                                rotation_bounds=rotate_bounds_schedule[0],
                                prob_patch=patch_schedule[0],
                                prob_dropout=dropout_schedule[0],
                                device=device_generator,
                                noise_std_max=noise_schedule[0],
                                lowres_min=lowres_min,
                                lowres_max=lowres_max_schedule[0],
                                njobs=njobs)

# Prepare model
model = S2UNetGlobal().to(device_training)

LR=1e-4
optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0)

warm  = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100)
cos   = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30_000, eta_min=2e-6)
scheduler = lr_scheduler.SequentialLR(optimizer, [warm, cos], [200])

# Load weights if provided
if initial_model is None:
    print('Starting from scratch')
    epoch_ini = 0
else:
    print('Loading weights from ' + initial_model)
    checkpoint = torch.load(initial_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch_ini = 1 + checkpoint['epoch']


# Train!
global_step = 0
for j in range(n_epochs - epoch_ini):
    model.train()
    epoch = epoch_ini + j

    print('Epoch ' + str(epoch+1) + ' of ' + str(n_epochs))
    print('Current LR: ', optimizer.param_groups[0]['lr'])
    loss_epoch_acc = 0.0

    loss_epoch_acc_l1 = 0.0
    loss_epoch_acc_l2 = 0.0
    loss_epoch_acc_ang = 0.0

    if epoch % 50 == 0 and epoch > 10:
        idx = epoch // 50
        #idx = 0 ### obviously remove when not testing pls
        new_noise_std = noise_schedule[min(idx, len(noise_schedule)-1)]
        new_lowres_max = lowres_max_schedule[min(idx, len(lowres_max_schedule)-1)]
        new_rotate_prob = rotate_prob_schedule[min(idx, len(rotate_prob_schedule)-1)]
        new_rotate_bounds = rotate_bounds_schedule[min(idx, len(rotate_bounds_schedule)-1)]
        new_patch_prob = patch_schedule[min(idx, len(patch_schedule)-1)]
        new_dropout_prob = dropout_schedule[min(idx, len(dropout_schedule)-1)]

        gen = hr_lr_random_res_generator(training_data_dir,
                                crop_size=crop_size,
                                prob_rotate=new_rotate_prob,
                                rotation_bounds=new_rotate_bounds,
                                prob_patch=new_patch_prob,
                                prob_dropout=new_dropout_prob,
                                device=device_generator,
                                noise_std_max=new_noise_std,
                                lowres_min=lowres_min,
                                lowres_max=new_lowres_max,
                                njobs=njobs)

        print("Updated noise std to: ", new_noise_std)
        print("Updated lowres max to: ", new_lowres_max)
        print("Updated rotate prob to: ", new_rotate_prob)
        print("Updated rotation bounds to: ", new_rotate_bounds)
        print("Updated patch probability to: ", new_patch_prob)

    for iteration in range(n_its_per_epoch):

        inputs_list = []
        targets_list = []

        for _ in range(batch_size):
            # Grab one sample (input, target) from the generator
            single_input, single_target = next(gen)
            inputs_list.append(single_input)
            targets_list.append(single_target)

        ###### IF TESTING SINGLE CASE #######################################
        #single_input, aff = load_volume('./test_sample/input.nii.gz')
        #single_input = single_input.astype(float)
        #single_target, aff = load_volume('./test_sample/target.nii.gz')
        #single_target = single_target.astype(float)

        #single_input = torch.tensor(single_input, device=device_generator).float()
        #single_target = torch.tensor(single_target, device=device_generator).float()
        #single_input = single_input.permute(3, 0, 1, 2)
        #single_target = single_target.permute(3, 0, 1, 2)

        #inputs_list.append(single_input)
        #targets_list.append(single_target)
        #####################################################################

        input_batch = torch.stack(inputs_list, dim=0).to(device_training)
        target_batch = torch.stack(targets_list, dim=0).to(device_training)


        optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=torch.float16):

            pred_batch = model(input_batch)
            loss, l2_loss = mixed_loss(pred_batch, target_batch, multiplier=1000.0, ang_multiplier=5000)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()


        loss_epoch_acc = loss_epoch_acc + loss.detach().cpu().numpy()
        cumul_loss_epoch = loss_epoch_acc / (iteration + 1)


        loss_epoch_acc_l2 = loss_epoch_acc_l2 + l2_loss.detach().cpu().numpy()
        cumul_loss_epoch_l2 = loss_epoch_acc_l2 / (iteration + 1)

        print('   Iteration ' + str(1+iteration) + ' of ' + str(n_its_per_epoch) + ', tot loss = ' + str(cumul_loss_epoch) + '  l2 loss: ' + str(cumul_loss_epoch_l2), end="\r")

        if epoch % 10 == 0 and iteration == 5:
            os.makedirs("./tmp_epoch_output", exist_ok=True)
            print("saving a batch for checking...")

            input_npy = input_batch[0,...].detach().permute(1, 2, 3, 0).cpu().numpy()
            target_npy = target_batch[0,...].detach().permute(1, 2, 3, 0).cpu().numpy()
            pred_npy = pred_batch[0,...].detach().permute(1, 2, 3, 0).cpu().numpy()

            nib.save(nib.Nifti1Image(input_npy, affine=np.eye(4)), './tmp_epoch_output/input.nii.gz')
            nib.save(nib.Nifti1Image(target_npy, affine=np.eye(4)), './tmp_epoch_output/target.nii.gz')
            nib.save(nib.Nifti1Image(pred_npy, affine=np.eye(4)), './tmp_epoch_output/pred.nii.gz')

    print('\n   End of epoch ' + str(epoch+1) + '; saving model... \n')

    #scheduler.step(cumul_loss_epoch)
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': cumul_loss_epoch,
        }, '%s/checkpoint_%.4d.pth' % (output_directory, 1+epoch))

print('Training complete!')
