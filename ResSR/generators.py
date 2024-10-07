import glob
import numpy as np
import torch
from ResSR.utils import load_volume, make_rotation_matrix, myzoom_torch, fast_3D_interp_torch, make_gaussian_kernel


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
    image_list = glob.glob(training_dir + '/*gz')
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
        hr, aff = load_volume(image_list[index])
        hr = np.squeeze(hr)
        orig_shape = hr.shape
        orig_center = (np.array(orig_shape) - 1) / 2
        hr = torch.tensor(hr, device=device)

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
        hr_def = fast_3D_interp_torch(hr, xx2, yy2, zz2, 'linear', device=device)

        # Add random bias field and gamma transform
        gamma = torch.exp(torch.tensor(gamma_std) * torch.randn([1], device=device))
        hr_gamma = ((hr_def / torch.max(hr_def)) ** gamma)
        npoints = np.random.randint(1 + bf_maxsize)
        if npoints==0:
            bias = torch.ones(1, device=device)
        else:
            stddev = bf_std_max * torch.rand([1], device=device)
            lr_bf = stddev * torch.randn([npoints, npoints, npoints], device=device)
            factor = np.array(crop_size) / npoints
            bias = torch.exp(myzoom_torch(lr_bf, factor, device=device))
        hr_bias = hr_gamma * bias

        # Now simulate low resolution
        # The theoretical blurring sigma to blur the resolution depends on the fraction by which we want to
        # divide the power at the cutoff frequency. I use [0.45,0.85]
        blurred = hr_bias[None, None, :]
        ratios = lowres_min + (lowres_max - lowres_min) * np.random.rand(3)
        ratios = crop_size / (np.round(crop_size / ratios))  # we make sure that the ratios lead to an integer size
        for d in range(3):
            ratio = ratios[d]
            blurred = blurred.permute([0,1,4,2,3])
            if ratio>1:
                fraction = 0.45 + 0.4 * np.random.rand(1)
                sigma = fraction * ratio
                kernel = torch.tensor(make_gaussian_kernel(sigma), dtype=torch.float32, device=device)[None, None, :, None, None]
                blurred = torch.conv3d(blurred, kernel, stride=1, padding=[int((kernel.shape[2] - 1) / 2), 0, 0] )
        blurred = torch.squeeze(blurred)
        lr = myzoom_torch(blurred, 1 / ratios, device=device)

        # Now we add noise (at low resolution, as will happen at test time)
        noise_std = noise_std_min + (noise_std_max - noise_std_min) * torch.rand([1], device=device)
        lr_noisy = lr + noise_std * torch.randn(lr.shape, device=device)

        # We also renormalize here (as we do at test time!)
        # And also keep the target at the same scale
        maxi = torch.max(lr_noisy)
        lr_noisy = lr_noisy / maxi
        target = hr_bias / maxi

        # Finally, we go back to the original resolution
        input = myzoom_torch(lr_noisy, ratios, device=device)

        yield input, target
