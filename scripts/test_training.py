import os
import torch
from torch.optim import Adam
from torch.nn import L1Loss

sys.path.append('/autofs/space/nicc_003/users/olchanyi/DiffSR')
from ResSR.generators import hr_lr_random_res_generator
from ResSR.models import SRmodel


# Parameters
training_data_dir = '/autofs/space/panamint_005/users/iglesias/data/HCPunpacked/data/'
device_generator = 'cuda:0'
# device_generator = 'cpu'
device_training = 'cuda:0'
num_filters = 128
num_residual_blocks = 16
crop_size = 96
kernel_size = 3
use_global_residual = True
n_epochs = 1000
n_its_per_epoch = 100
output_directory = '/autofs/space/panamint_001/users/iglesias/models_temp/ResSR_test_less_noise/'
# initial_model = '/autofs/space/panamint_001/users/iglesias/models_temp/ResSR_test/checkpoint_0675.pth'
initial_model = None
# noise_std_max=0.10
noise_std_max=0.05
lowres_min=1.5
lowres_max=3.0

# Create output directory if needed
if os.path.exists(output_directory) is False:
    os.mkdir(output_directory)

# Prepare generator
gen = hr_lr_random_res_generator(training_data_dir, crop_size=crop_size, device=device_generator, noise_std_max=noise_std_max, lowres_min=lowres_min, lowres_max=lowres_max)

# Prepare model
model = SRmodel(num_filters, num_residual_blocks, kernel_size, use_global_residual).to(device_training)
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = L1Loss()

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
        input = input[None, None, :].to(device_training)
        target = target[None, None, :].to(device_training)

        pred = model(input)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch_acc = loss_epoch_acc + loss.detach().cpu().numpy()
        cumul_loss_epoch = loss_epoch_acc / (iteration + 1)

        print('   Iteration ' + str(1+iteration) + ' of ' + str(n_its_per_epoch) + ', loss = ' + str(cumul_loss_epoch), end="\r")

    print('\n   End of epoch ' + str(epoch+1) + '; saving model... \n')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': cumul_loss_epoch,
    }, '%s/checkpoint_%.4d.pth' % (output_directory, 1+epoch))

print('Training complete!')
