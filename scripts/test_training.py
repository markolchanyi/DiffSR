import os
import sys
import torch
from torch.optim import Adam
from torch.nn import L1Loss, MSELoss
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append('/autofs/space/nicc_003/users/olchanyi/DiffSR')
from ResSR.generators import hr_lr_random_res_generator
from ResSR.models import SRmodel
from ResSR.utils import mixed_loss

# Parameters
training_data_dir = '/autofs/space/nicc_005/users/olchanyi/DiffSR/training_data_prerotated/fod/'
device_generator = 'cuda:0'
# device_generator = 'cpu'
device_training = 'cuda:0'
num_filters = 256
num_residual_blocks = 16
crop_size = 64
kernel_size = 3
use_global_residual = True
n_epochs = 2000
n_its_per_epoch = 100
output_directory = '/autofs/space/nicc_005/users/olchanyi/DiffSR/model_outputs_v11_test/'
#initial_model = '/autofs/space/nicc_005/users/olchanyi/DiffSR/model_outputs_v10/checkpoint_0073.pth'
initial_model = None
# noise_std_max=0.10
noise_std_max=0.08
lowres_min=1.5
lowres_max=3.5

# Create output directory if needed
if os.path.exists(output_directory) is False:
    os.mkdir(output_directory)

# Prepare generator
gen = hr_lr_random_res_generator(training_data_dir, crop_size=crop_size, device=device_generator, noise_std_max=noise_std_max, lowres_min=lowres_min, lowres_max=lowres_max)

# Prepare model
model = SRmodel(num_filters, num_residual_blocks, kernel_size, use_global_residual).to(device_training)
optimizer = Adam(model.parameters(), lr=2e-5, weight_decay=2e-6)
# Initialize scheduler
#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

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

# Train!
for j in range(n_epochs - epoch_ini):

    epoch = epoch_ini + j

    print('Epoch ' + str(epoch+1) + ' of ' + str(n_epochs))
    loss_epoch_acc = 0.0

    for iteration in range(n_its_per_epoch):

        input, target = next(gen)
        input = input[None, :].to(device_training)
        target = target[None, :].to(device_training)

        pred = model(input)
        #loss = loss_fn(pred, target)
        loss = mixed_loss(pred, target, l1_loss_fn, l2_loss_fn, alpha=0.1, beta=10.0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_epoch_acc = loss_epoch_acc + loss.detach().cpu().numpy()
        cumul_loss_epoch = loss_epoch_acc / (iteration + 1)

        print('   Iteration ' + str(1+iteration) + ' of ' + str(n_its_per_epoch) + ', loss = ' + str(cumul_loss_epoch), end="\r")

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
