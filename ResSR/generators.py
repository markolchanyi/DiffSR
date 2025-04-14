import glob
import os
import shutil
import numpy as np
import nibabel as nib
import torch
import random
from ResSR.utils import load_volume, make_rotation_matrix, myzoom_torch, fast_3D_interp_torch, rand_lowrank_mix
from ResSR.utils import make_gaussian_kernel, random_crop, random_rotate_sh, batch_rotate_sh, percentile_scaling, sh_norm


def hr_lr_random_res_generator(training_dir,
                               crop_size=64,
                               rotation_bounds=20,
                               scaling_bounds=0.15,
                               nonlin_maxsize=8,
                               nonlin_std_max=3.0,
                               prob_dropout=0.1,
                               prob_sh_rotate_deform=0.01,
                               lowres_min=1,
                               lowres_max=3,
                               gamma_std=0.1,
                               bf_maxsize=4,
                               bf_std_max=0.2,
                               noise_std_min=0.00,
                               noise_std_max=0.10,
                               device='cpu',
                               njobs=1):


    # List images
    image_list = glob.glob(os.path.join(training_dir, '*/sh_coefficients_b*_masked.nii.gz'))
    #image_list = glob.glob(os.path.join(training_dir, '*/fod.nii.gz'))
    #print("Found training images: ", image_list,"\n")
    n_training = len(image_list)
    print('Found %d cases for training' % n_training)

    # Create grid we'll reuse all the time
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size, crop_size]

    xx, yy, zz = np.meshgrid(range(crop_size[0]), range(crop_size[1]), range(crop_size[2]), sparse=False, indexing='ij')
    cx, cy, cz = (np.array(crop_size) - 1) / 2
    xc = xx - cx
    yc = yy - cy
    zc = zz - cz
    xc = torch.tensor(xc, device=device)
    yc = torch.tensor(yc, device=device)
    zc = torch.tensor(zc, device=device)

    # Generate!
    while True:
        # randomly pick an image and read it
        index = np.random.randint(n_training)
        hr, aff = load_volume(image_list[index])  # Load the FOD
        hr = hr.astype(float)
        hr = np.squeeze(hr)  # Ensure it's the correct shape (28, x, y, z)

        orig_shape = hr.shape[:-1]  # Shape of the 3D volume (x, y, z)
        if hr.shape[-1] != 28:
            raise ValueError("Expected FOD with 28 channels (lmax=6), but got shape: {}".format(hr.shape))

        orig_center = (np.array(orig_shape) - 1) / 2
        hr = torch.tensor(hr, device=device)

        # Replace NaNs, +inf, and -inf with 0
        hr[torch.isnan(hr)] = 0.0
        hr[torch.isinf(hr)] = 0.0

        # random view cropping
        hr_cropped = random_crop(hr, crop_size).float()
        #hr_cropped_copy = hr_cropped.detach().clone()

        ###########################################################
          # SH rotation or deformation (either or to save time) #
        ###########################################################
        if type(prob_sh_rotate_deform) != float:
            prob_sh_rotate_deform = prob_sh_rotate_deform[0] ## TODO: weird bug
        if random.random() < prob_sh_rotate_deform:
            alpha = np.random.uniform(-rotation_bounds, rotation_bounds)
            beta = np.random.uniform(-rotation_bounds, rotation_bounds)
            gamma = np.random.uniform(-rotation_bounds, rotation_bounds)

            spline_spacing = random.randint(8, 15)
            deform_mag = random.uniform(2, 3)

            os.makedirs('./tmp', exist_ok=True)
            nib.save(nib.Nifti1Image(hr_cropped.cpu().numpy(), affine=aff), './tmp/sh_raw.nii.gz')

            if random.random() < 0.05:
                #print("ROTATING!!")
                cmd = "python ../ResSR/sh_rotation.py"
                cmd += " -i ./tmp/sh_raw.nii.gz"
                cmd += " -o ./tmp/sh_dwig.nii.gz"
                cmd += " --alpha " + str(alpha)
                cmd += " --beta " + str(beta)
                cmd += " --gamma " + str(gamma)
                cmd += " --n_jobs " + str(njobs)
                os.system(cmd)
                #print("done!!")

            else:
                #print("starting random deformation...")
                cmd = "python ../ResSR/sh_deformation.py"
                cmd += " --in_sh ./tmp/sh_raw.nii.gz"
                cmd += " --out_sh ./tmp/sh_dwig.nii.gz"
                cmd += " --spacing " + str(spline_spacing)
                cmd += " --warp_scale " + str(deform_mag)
                cmd += " --check_global_jacobian False"
                cmd += " --patch_only 0"
                os.system(cmd)
                #print("done!!")

            hr_rot_def, _ = load_volume('./tmp/sh_dwig.nii.gz')
            hr_rot_def = hr_rot_def.astype(float)
            hr_rot_def = np.squeeze(hr_rot_def)
            hr_rot_def = torch.tensor(hr_rot_def, device=device).float()
            hr_rot_def[torch.isnan(hr_rot_def)] = 0.0
            hr_rot_def[torch.isinf(hr_rot_def)] = 0.0

            hr_cropped=hr_rot_def
            shutil.rmtree('./tmp')

        # IQR scale the l=0 isotropic component
        #hr_cropped = adc_sh_norm(hr_cropped, l0_index=0, k=2.0, new_min=-1.0, new_max=1.0, threshold=0.01)
        hr_cropped = sh_norm(hr_cropped,l0_index=0)
        hr_cropped = torch.clamp(hr_cropped, min=-1, max=1)

        # Add random bias field and gamma transform
        # ONLY introduce these ops to the l=0 SH coeff
        # since all higher-order coeffs are purely in angular domain
        gamma = torch.exp(torch.tensor(gamma_std) * torch.randn([1], device=device)).float()

        hr_gamma = hr_cropped.detach().clone()
        hr_gamma[...,0] = ((hr_gamma[...,0] / torch.max(hr_gamma[...,0])) ** gamma)


        ############################################
                     # BIAS FIELD(S) #
        ############################################
        npoints = np.random.randint(1 + bf_maxsize)
        if npoints==0:
            bias = torch.ones(1, device=device)
        else:
            stddev = bf_std_max * torch.rand([1], device=device)
            lr_bf = stddev * torch.randn([npoints, npoints, npoints], device=device)
            factor = np.array(crop_size) / npoints
            bias = torch.exp(myzoom_torch(lr_bf, factor, device=device)).float()

        # Only apply to zeroth-order harmonic
        hr_bias = hr_gamma.detach().clone()
        hr_bias[...,0] = hr_gamma[...,0] * bias


        ## Dir-specific bias ##
        ## approximated with low-rank mixing for now TODO
        if random.random() < 0.25:
            #print("applying bias")
            hr_bias[...,1:] = rand_lowrank_mix(hr_bias[...,1:], rank=4, scale=0.05)
            #print("done")

        ### RANDOM DROPOUT
        sh_mapping = {
            0: [0],
            2: [1, 2, 3, 4, 5],
            4: [6, 7, 8, 9, 10, 11, 12, 13, 14],
            6: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        }

        # Randomly drop-out higher-order SH l's
        if type(prob_dropout) != float:
            prob_dropout = prob_dropout[0] # TODO again...weird
        #print("dropout: ", prob_dropout)
        rand = random.random()
        if rand < prob_dropout/2:
            hr_bias[...,sh_mapping[4]] = 0
            hr_bias[...,sh_mapping[6]] = 0
        elif rand > 1-(prob_dropout/2):
            hr_bias[...,sh_mapping[6]] = 0
        else:
            pass

        # Now simulate low resolution
        # The theoretical blurring sigma to blur the resolution depends on the fraction by which we want to
        # divide the power at the cutoff frequency. I use [0.45,0.85]
        hr_bias_clone = hr_bias.detach().clone()
        blurred = hr_bias_clone[None, None, :]

        blurred[torch.isnan(blurred)] = 0.0
        blurred[torch.isinf(blurred)] = 0.0

        ratios = lowres_min + (lowres_max - lowres_min) * np.random.rand(3)
        ratios = crop_size / (np.round(crop_size / ratios))  # we make sure that the ratios lead to an integer size
        for d in range(3):
            ratio = ratios[d]
            blurred = blurred.permute([0,1,4,2,3,5]) # keep last SH dimension in-place
            if ratio>1:
                fraction = 0.45 + 0.4 * np.random.rand(1)
                sigma = fraction * ratio
                kernel = torch.tensor(make_gaussian_kernel(sigma), dtype=torch.float32, device=device)[None, None, :, None, None]
                for c in range(blurred.shape[-1]):
                    blurred[...,c] = torch.conv3d(blurred[...,c], kernel, stride=1, padding=[int((kernel.shape[2] - 1) / 2), 0, 0] )
        blurred = torch.squeeze(blurred)
        lr = myzoom_torch(blurred, 1 / ratios, device=device)

        # Now we add noise (at low resolution, as will happen at test time) 50 50 gaussian or Rician
        noise_std = noise_std_min + (noise_std_max - noise_std_min) * torch.rand([1], device=device)

        # Rician
        #noise_real = noise_std * torch.randn(lr.shape, device=device)
        #noise_imag = noise_std * torch.randn(lr.shape, device=device)

        #lr_noisy = torch.sqrt((lr + noise_real)**2 + (noise_imag)**2)

        lr_noisy = lr + noise_std * torch.randn(lr.shape, device=device)

        # We also renormalize here (as we do at test time!)
        target = hr_bias

        # Finally, we go back to the original resolution
        input = myzoom_torch(lr_noisy, ratios, device=device)

        input_nopatched = input.detach().clone()

        ## RANDOM PATCHING
        if random.random() < 0.25:
            os.makedirs('./tmp', exist_ok=True)
            nib.save(nib.Nifti1Image(input.cpu().numpy(), affine=aff), './tmp/sh_raw.nii.gz')
            #print("starting patch")
            cmd = "python ../ResSR/sh_deformation.py"
            cmd += " --in_sh ./tmp/sh_raw.nii.gz"
            cmd += " --out_sh ./tmp/sh_dwig.nii.gz"
            cmd += " --spacing 1"
            cmd += " --warp_scale 1"
            cmd += " --check_global_jacobian False"
            cmd += " --patch_only 1"
            os.system(cmd)
            #print("done")
            input_patched, _ = load_volume('./tmp/sh_dwig.nii.gz')
            input_patched = input_patched.astype(float)
            input_patched = np.squeeze(input_patched)
            input_patched = torch.tensor(input_patched, device=device).float()
            input_patched[torch.isnan(input_patched)] = 0.0
            input_patched[torch.isinf(input_patched)] = 0.0
            input=input_patched
            shutil.rmtree('./tmp')

        input = input.float()
        target = target.float()

        input[torch.isnan(input)] = 0.0
        input[torch.isinf(input)] = 0.0
        input = torch.clamp(input, min=-1, max=1)

        target[torch.isnan(target)] = 0.0
        target[torch.isinf(target)] = 0.0
        target = torch.clamp(target, min=-1, max=1)

        ##### TEST SAVE
        #print("Saving intermediates...")
        #os.makedirs("./tmp",exist_ok=True)
        #input_npy = input_nopatched.cpu().numpy()
        #nib.save(nib.Nifti1Image(input_npy, affine=np.eye(4)), './tmp/input_nopatched.nii.gz')
        #input_npy = input.cpu().numpy()
        #nib.save(nib.Nifti1Image(input_npy, affine=np.eye(4)), './tmp/input.nii.gz')
        #target_npy = target.cpu().numpy()
        #nib.save(nib.Nifti1Image(target_npy, affine=np.eye(4)), './tmp/target.nii.gz')
        #print("done")
        #####

        input = input.permute(3, 0, 1, 2)
        target = target.permute(3, 0, 1, 2)

        yield input, target
