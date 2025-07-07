import glob
import os
import shutil
import numpy as np
import nibabel as nib
import torch
import random
from ResSR.utils import load_volume, make_rotation_matrix, myzoom_torch, fast_3D_interp_torch, rand_lowrank_mix, unpad_tensor
from ResSR.utils import make_gaussian_kernel, random_crop, random_rotate_sh, batch_rotate_sh, percentile_scaling, sh_norm
from ResSR.sh_utils import rotate_sh_volume

def hr_lr_random_res_generator(training_dir,
                               crop_size=64,
                               prob_rotate=0.1,
                               rotation_bounds=20,
                               prob_patch=0.1,
                               prob_dropout=0.1,
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

    ### if padding to avoid edge effects ###
    padsize=4
    crop_size += 2*padsize

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
        hr, aff = load_volume(image_list[index])  # Load the SH coeffs
        hr = hr.astype(float)
        hr = np.squeeze(hr)  # Ensure it's the correct shape (x, y, z, 28)

        parentdir = os.path.dirname(image_list[index])
        lowb, aff_lowb = load_volume(os.path.join(parentdir,'mean_b0_synthstripped.nii.gz'))  # Load the mean lowb
        lowb = lowb.astype(float)
        lowb = np.squeeze(lowb)  # (x, y, z)

        lowb = lowb[..., np.newaxis]
        hr = np.concatenate([lowb, hr], axis=-1) # now lowb is first channel (REMEMBER!)

        orig_shape = hr.shape[:-1]  # Shape of the 3D volume (x, y, z)
        if hr.shape[-1] != 29:
            raise ValueError("Expected SH coeffs + 1 mean b0 with 28 + 1 channels (lmax=6), but got shape: {}".format(hr.shape))

        orig_center = (np.array(orig_shape) - 1) / 2
        hr = torch.tensor(hr, device=device)

        # Replace NaNs, +inf, and -inf with 0
        hr[torch.isnan(hr)] = 0.0
        hr[torch.isinf(hr)] = 0.0

        # random view cropping
        hr_cropped = random_crop(hr, crop_size).float()

        ###########################################################
                 # SH ROTATION (either or to save time) #
        ###########################################################
        if random.random() < prob_rotate:
            alpha = np.random.uniform(-rotation_bounds, rotation_bounds)
            beta = np.random.uniform(-rotation_bounds, rotation_bounds)
            gamma = np.random.uniform(-rotation_bounds, rotation_bounds)

            # deformation params
            patch_low = 10
            patch_high = 32
            px, py, pz =  (np.random.randint(patch_low, patch_high),
                           np.random.randint(patch_low, patch_high),
                           np.random.randint(patch_low, patch_high))

            patch_size=(px,py,pz)
            spacing=np.random.uniform(1, np.max((px,py,pz))/2)
            warp_scale=np.random.uniform(1,np.max((px,py,pz))/6)

            hr_rot_def = rotate_sh_volume(hr_cropped.cpu().numpy(),
                                         (alpha,beta,gamma),
                                         rotate=True,
                                         deform_patch=True,
                                         add_noise=False,
                                         patch_size=patch_size,
                                         spacing=spacing,
                                         warp_scale=spacing)

            hr_rot_def = hr_rot_def.astype(float)
            hr_rot_def = torch.tensor(hr_rot_def, device=device).float()
            hr_rot_def[torch.isnan(hr_rot_def)] = 0.0
            hr_rot_def[torch.isinf(hr_rot_def)] = 0.0

            hr_cropped=hr_rot_def

        # IQR scale the l=0 isotropic component
        hr_cropped = sh_norm(hr_cropped,l0_index=1)
        hr_cropped = torch.clamp(hr_cropped, min=-1, max=1)


        # Add random bias field and gamma transform
        # ONLY introduce these ops to the l=0 SH coeff and lowb
        # since all higher-order coeffs are purely in angular domain
        # in theory if there are b-specific biases, they could be different in lowb and l0
        gamma_l0 = torch.exp(torch.tensor(gamma_std) * torch.randn([1], device=device)).float()
        gamma_lowb = torch.exp(torch.tensor(gamma_std) * torch.randn([1], device=device)).float()

        hr_gamma = hr_cropped.detach().clone()
        hr_gamma[...,0] = ((hr_gamma[...,0] / torch.max(hr_gamma[...,0])) ** gamma_lowb)
        hr_gamma[...,1] = ((hr_gamma[...,1] / torch.max(hr_gamma[...,1])) ** gamma_l0)


        ############################################
                     # BIAS FIELD(S) #
        ############################################
        npoints = np.random.randint(1 + bf_maxsize)
        if npoints==0:
            bias_lowb = torch.ones(1, device=device)
            bias_l0 = bias_lowb
        else:
            stddev_lowb = bf_std_max * torch.rand([1], device=device)
            stddev_l0 = bf_std_max * torch.rand([1], device=device)

            lr_bf_lowb = stddev_lowb * torch.randn([npoints, npoints, npoints], device=device)
            lr_bf_l0 = stddev_l0 * torch.randn([npoints, npoints, npoints], device=device)

            factor = np.array(crop_size) / npoints
            bias_lowb = torch.exp(myzoom_torch(lr_bf_lowb, factor, device=device)).float()
            bias_l0 = torch.exp(myzoom_torch(lr_bf_l0, factor, device=device)).float()

        # Only apply to zeroth-order harmonic and lowb
        hr_bias = hr_gamma.detach().clone()
        hr_bias[...,0] = hr_gamma[...,0] * bias_lowb
        hr_bias[...,1] = hr_gamma[...,1] * bias_l0
        #hr_bias[...,0] = hr_gamma[...,0] * 1
        #hr_bias[...,1] = hr_gamma[...,1] * 1

        ## Dir-specific bias ##
        ## approximated with low-rank mixing for now TODO
        if random.random() < 0.1:
            #print("applying bias")
            hr_bias[...,1:] = rand_lowrank_mix(hr_bias[...,1:], rank=2, scale=0.04)
            #print("done")

        ### RANDOM DROPOUT
        sh_mapping = {
            0: [1],
            2: [2, 3, 4, 5, 6],
            4: [7, 8, 9, 10, 11, 12, 13, 14, 15],
            6: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        }

        # Randomly drop-out higher-order SH l's
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
        lr_noisy = lr + noise_std * torch.randn(lr.shape, device=device)

        # We also renormalize here (as we do at test time!)
        target = hr_bias

        # Finally, we go back to the original resolution
        input = myzoom_torch(lr_noisy, ratios, device=device)

        input_nopatched = input.detach().clone()

        ### MISMATCHED PATCHING ###
        if random.random() < prob_patch:
            drift_alpha = np.random.uniform(0.0,360.0)
            drift_beta = np.random.uniform(0.0,360.0)
            drift_gamma = np.random.uniform(0.0,180.0)

            # deformation params
            patch_low = 10
            patch_high = 20
            px, py, pz =  (np.random.randint(patch_low, patch_high),
                           np.random.randint(patch_low, patch_high),
                           np.random.randint(patch_low, patch_high))

            patch_size=(px,py,pz)
            spacing=np.random.uniform(1, np.max((px,py,pz))/2)
            warp_scale=np.random.uniform(1,np.max((px,py,pz))/6)


            #### drift parameters (set high) ####
            drift_patch_low = 20
            drift_patch_high = 45
            dpx, dpy, dpz =  (np.random.randint(drift_patch_low, drift_patch_high),
                           np.random.randint(drift_patch_low, drift_patch_high),
                           np.random.randint(drift_patch_low, drift_patch_high))

            drift_patch_size=(dpx,dpy,dpz)

            prob_drift=0.5
            apply_random_drift=False

            if random.random() < prob_drift:
                apply_random_drift=True

            input_patched = rotate_sh_volume(input.cpu().numpy(),
                                             (drift_alpha,drift_beta,drift_gamma),
                                             rotate=False,
                                             deform_patch=True,
                                             apply_random_drift=apply_random_drift,
                                             add_noise=True,
                                             patch_size=patch_size,
                                             drift_patch_size=drift_patch_size,
                                             spacing=spacing,
                                             warp_scale=spacing)

            input_patched = input_patched.astype(float)
            input_patched = torch.tensor(input_patched, device=device).float()
            input_patched[torch.isnan(input_patched)] = 0.0
            input_patched[torch.isinf(input_patched)] = 0.0
            input=input_patched


        ##### random mismatched patch-wise dropout
        if random.random() < 0:
            block_size_x = np.random.randint(0, 12)
            block_size_y = np.random.randint(0, 12)
            block_size_z = np.random.randint(0, 12)
            X,Y,Z = input.shape[:3]
            z = np.random.randint(0, Z - block_size_z)
            y = np.random.randint(0, Y - block_size_y)
            x = np.random.randint(0, X - block_size_x)
            input[x:x+block_size_x, y:y+block_size_y, z:z+block_size_z,:] = 0

        input = input.float()
        target = target.float()

        input[torch.isnan(input)] = 0.0
        input[torch.isinf(input)] = 0.0
        input = torch.clamp(input, min=-1, max=1)

        target[torch.isnan(target)] = 0.0
        target[torch.isinf(target)] = 0.0
        target = torch.clamp(target, min=-1, max=1)

        input = input.permute(3, 0, 1, 2)
        target = target.permute(3, 0, 1, 2)

        ### unpad edges ###
        input = unpad_tensor(input, padsize)
        target = unpad_tensor(target, padsize)

        ##### TEST SAVE
        #print("Saving intermediates...")
        #os.makedirs("./tmp_gen",exist_ok=True)
        #input_npy = input.permute(1, 2, 3, 0).cpu().numpy()
        #nib.save(nib.Nifti1Image(input_npy, affine=np.eye(4)), './tmp_gen/input.nii.gz')
        #target_npy = target.permute(1, 2, 3, 0).cpu().numpy()
        #nib.save(nib.Nifti1Image(target_npy, affine=np.eye(4)), './tmp_gen/target.nii.gz')
        #print("done")
        #####

        yield input, target
