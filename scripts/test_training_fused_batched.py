import os
import sys
import torch
import numpy as np
import nibabel as nib
from torch.optim import Adam
from torch.nn import L1Loss, MSELoss
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append('/autofs/space/nicc_003/users/olchanyi/DiffSR')
from ResSR.generators import hr_lr_random_res_generator
from ResSR.models_fused import SRmodel
from ResSR.utils import mixed_loss, fibonacci_sphere, _mrtrix_real_sh_basis, evaluate_mrtrix_sh

# Parameters
training_data_dir = '/autofs/space/nicc_005/users/olchanyi/DiffSR/training_data/sshell_sh/'
device_generator = 'cuda:1'
# device_generator = 'cpu'
device_training = 'cuda:0'
num_filters = 256
num_residual_blocks = 24
crop_size = 32
kernel_size = 3
prob_dropout=0.2,
prob_sh_rotate_deform=0.25,
use_global_residual = False
n_epochs = 2000
n_its_per_epoch = 20
output_directory = '/autofs/space/nicc_005/users/olchanyi/DiffSR/models_fused/model_fused_v5/'
#initial_model = '/autofs/space/nicc_005/users/olchanyi/DiffSR/models_fused/model_fused_v3/checkpoint_0067.pth'
initial_model = None
noise_std_max=0.06
lowres_min=1.5
lowres_max=4
njobs = 64

# @TODO any more than 5 and gpu mem overflows
batch_size = 5

# Create output directory if needed
if os.path.exists(output_directory) is False:
    os.mkdir(output_directory)

# Prepare generator
gen = hr_lr_random_res_generator(training_data_dir,
                                crop_size=crop_size,
                                device=device_generator,
                                noise_std_max=noise_std_max,
                                prob_dropout=prob_dropout,
                                prob_sh_rotate_deform=prob_sh_rotate_deform,
                                lowres_min=lowres_min,
                                lowres_max=lowres_max,
                                njobs=njobs)

# Prepare model
#model = SRmodel(num_filters, num_residual_blocks, kernel_size, use_global_residual).to(device_training)
model = SRmodel(num_filters=num_filters,
                num_residual_blocks=num_residual_blocks,
                kernel_size=kernel_size,
                use_global_residual=use_global_residual,
                num_filters_nonang=64,
                num_residual_blocks_nonang=12).to(device_training)

optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

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
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_ini = 1 + checkpoint['epoch']


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
    loss_epoch_acc = 0.0

    loss_epoch_acc_l1 = 0.0
    loss_epoch_acc_l2 = 0.0
    loss_epoch_acc_ang = 0.0

    for iteration in range(n_its_per_epoch):

        inputs_list = []
        targets_list = []
        for _ in range(batch_size):
            # Grab one sample (input, target) from the generator
            single_input, single_target = next(gen)
            # single_input shape => (28, X, Y, Z)
            # single_target shape => (28, X, Y, Z)

            inputs_list.append(single_input)
            targets_list.append(single_target)

        input_batch = torch.stack(inputs_list, dim=0).to(device_training)
        target_batch = torch.stack(targets_list, dim=0).to(device_training)


        optimizer.zero_grad()
        pred_batch = model(input_batch)

        loss, l1_loss, l2_loss, ang_loss = mixed_loss(pred_batch, target_batch, l1_loss_fn, l2_loss_fn, ang_multiplier=0.025, ang_dirs=sphere_dirs, alpha=0.6, beta=1, multiplier=5000, l0_multiplier=None)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_epoch_acc = loss_epoch_acc + loss.detach().cpu().numpy()
        cumul_loss_epoch = loss_epoch_acc / (iteration + 1)

        loss_epoch_acc_l1 = loss_epoch_acc_l1 + l1_loss.detach().cpu().numpy()
        cumul_loss_epoch_l1 = loss_epoch_acc_l1 / (iteration + 1)

        loss_epoch_acc_l2 = loss_epoch_acc_l2 + l2_loss.detach().cpu().numpy()
        cumul_loss_epoch_l2 = loss_epoch_acc_l2 / (iteration + 1)

        loss_epoch_acc_ang = loss_epoch_acc_ang + ang_loss.detach().cpu().numpy()
        cumul_loss_epoch_ang = loss_epoch_acc_ang / (iteration + 1)

        print('   Iteration ' + str(1+iteration) + ' of ' + str(n_its_per_epoch) + ', tot loss = ' + str(cumul_loss_epoch) + ', angular loss: ' + str(cumul_loss_epoch_ang), end="\r")

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
