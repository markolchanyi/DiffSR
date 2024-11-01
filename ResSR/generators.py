import glob
import os
import numpy as np
import nibabel as nib
import torch
from ResSR.utils import load_volume, make_rotation_matrix, myzoom_torch, fast_3D_interp_torch, make_gaussian_kernel, random_crop, random_rotate_sh, batch_rotate_sh


def hr_lr_random_res_generator(training_dir,
                               crop_size=64,
                               rotation_bounds=10,
                               scaling_bounds=0.15,
                               nonlin_maxsize=8,
                               nonlin_std_max=3.0,
                               lowres_min=1,
                               lowres_max=3,
                               gamma_std=0.1,
                               bf_maxsize=4,
                               bf_std_max=0.3,
                               noise_std_min=0.00,
                               noise_std_max=0.10,
                               device='cpu'):


    # List images
    image_list = glob.glob(os.path.join(training_dir, '*/fod.nii.gz'))
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
        #print(f"Rotated FOD min: {hr.min()}, max: {hr.max()}, mean: {hr.mean()}")
        # Replace NaNs, +inf, and -inf with 0
        hr[torch.isnan(hr)] = 0.0
        hr[torch.isinf(hr)] = 0.0
        hr = torch.clamp(hr, min=-1, max=1)

        #### OLD CODE without rotations of SH ####
        '''
        # Sample augmentation parameters
        rotations = (2 * rotation_bounds * np.random.rand(3) - rotation_bounds) / 180.0 * np.pi
        R = torch.tensor(make_rotation_matrix(rotations), device=device)
        s = torch.tensor(1 + (2 * scaling_bounds * np.random.rand(1) - scaling_bounds), device=device)
        t = (np.random.rand(3) - 0.5) * (np.array(orig_shape) - np.array(crop_size))
        npoints =  np.random.randint(1 + nonlin_maxsize)
        if npoints==0:
            hr_field = torch.zeros([1,1,1,3], device=device)
        else:
            stddev = nonlin_std_max * torch.rand([1], device=device)
            lr_field = stddev * torch.randn([npoints, npoints, npoints, 3], device=device)
            factor = np.array(crop_size) / npoints
            hr_field = myzoom_torch(lr_field, factor, device=device)

        # Interpolate!  There is no need to interpolate everywhere; only in the area we will (randomly) crop
        # Essentially, we crop and interpolate at the same time
        xx2 = orig_center[0] + s * (R[0, 0] * xc + R[0, 1] * yc + R[0, 2] * zc) + hr_field[:,:,:,0] + t[0]
        yy2 = orig_center[1] + s * (R[1, 0] * xc + R[1, 1] * yc + R[1, 2] * zc) + hr_field[:,:,:,1] + t[1]
        zz2 = orig_center[2] + s * (R[2, 0] * xc + R[2, 1] * yc + R[2, 2] * zc) + hr_field[:,:,:,2] + t[2]
        '''
        # random view cropping
        hr_cropped = random_crop(hr, crop_size).float()

        #hr_rot = batch_rotate_sh(hr_cropped,probability=1.0)

        # multichannel (i.e., SH coeffs) ok
        #hr_def = fast_3D_interp_torch(hr, xx2, yy2, zz2, 'linear', device=device)

        # Add random bias field and gamma transform
        # ONLY introduce these ops to the l=0 SH coeff
        # since all higher-order coeffs are purely in angular domain
        gamma = torch.exp(torch.tensor(gamma_std) * torch.randn([1], device=device)).float()

        hr_gamma = hr_cropped.detach().clone()
        hr_gamma[...,0] = ((hr_gamma[...,0] / torch.max(hr_gamma[...,0])) ** gamma)

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

        # Now simulate low resolution
        # The theoretical blurring sigma to blur the resolution depends on the fraction by which we want to
        # divide the power at the cutoff frequency. I use [0.45,0.85]
        hr_bias_clone = hr_bias.detach().clone()
        blurred = hr_bias_clone[None, None, :]

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

        # Now we add noise (at low resolution, as will happen at test time)
        noise_std = noise_std_min + (noise_std_max - noise_std_min) * torch.rand([1], device=device)
        lr_noisy = lr + noise_std * torch.randn(lr.shape, device=device)

        # We also renormalize here (as we do at test time!)
        # And also keep the target at the same scale
        #maxi = torch.max(lr_noisy)
        #lr_noisy = lr_noisy / maxi
        #target = hr_bias / maxi
        target = hr_bias

        # Finally, we go back to the original resolution
        input = myzoom_torch(lr_noisy, ratios, device=device)

        input = input.float()
        target = target.float()

        input[torch.isnan(input)] = 0.0
        input[torch.isinf(input)] = 0.0
        input[input < -1] = 0.0
        input[input > 1] = 0.0
        input = torch.clamp(input, min=-1, max=1)

        target[torch.isnan(target)] = 0.0
        target[torch.isinf(target)] = 0.0
        target = torch.clamp(target, min=-1, max=1)

        ##### TEST SAVE
        #print("Saving intermediates...")
        #os.makedirs("./tmp",exist_ok=True)
        #input_npy = input.cpu().numpy()
        #nib.save(nib.Nifti1Image(input_npy, affine=np.eye(4)), './tmp/input.nii.gz')
        #target_npy = target.cpu().numpy()
        #nib.save(nib.Nifti1Image(target_npy, affine=np.eye(4)), './tmp/target.nii.gz')
        #####

        input = input.permute(3, 0, 1, 2)
        target = target.permute(3, 0, 1, 2)

        yield input, target
