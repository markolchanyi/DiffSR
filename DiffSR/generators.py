import glob
import os
import numpy as np
import nibabel as nib
import torch
import random

from ResSR.utils import (
    load_volume,
    make_rotation_matrix,
    myzoom_torch,
    myzoom_torch_better,
    fast_3D_interp_torch,
    rand_lowrank_mix,
    unpad_tensor,
    make_gaussian_kernel,
    random_crop,
    percentile_scaling,
    sh_norm,
    _mrtrix_real_sh_basis,
)
from ResSR.sh_utils import rotate_sh_volume



try:
    from e3nn.util.grid import icosahedral_sphere as _e3nn_icosphere
except Exception:
    _e3nn_icosphere = None


# generates all needed vertices for the icosphere grid (for angular subsampling) (pretty much same as used for model blocks)
# N=1 gives level 1 icogrid: 42 vertices
def icosahedral_sphere(level=1):

    if _e3nn_icosphere is not None:
        verts = _e3nn_icosphere(level)
        if isinstance(verts, torch.Tensor):
            verts = verts.detach().cpu().numpy()
        return verts.astype(np.float32)

    # recursive subdivision IF NEEDED
    phi = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [[-1,  phi, 0],
            [ 1,  phi, 0],
            [-1, -phi, 0],
            [ 1, -phi, 0],
            [0, -1,  phi],
            [0,  1,  phi],
            [0, -1, -phi],
            [0,  1, -phi],
            [ phi, 0, -1],
            [ phi, 0,  1],
            [-phi, 0, -1],
            [-phi, 0,  1],
        ],
        dtype=np.float32,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array(
        [[0, 11, 5], [0, 5, 1],  [0, 1, 7],  [0, 7, 10], [0, 10, 11],
            [1, 5, 9],  [5, 11, 4], [11, 10, 2],[10, 7, 6], [7, 1, 8],
            [3, 9, 4],  [3, 4, 2],  [3, 2, 6],  [3, 6, 8],  [3, 8, 9],
            [4, 9, 5],  [2, 4, 11], [6, 2, 10],[8, 6, 7],  [9, 8, 1],
        ],
        dtype=np.int64,
    )

    for _ in range(level):
        vlist = verts.tolist()
        flist = []
        mid = {}

        def midpoint(a: int, b: int):
            key = tuple(sorted((a, b)))
            if key in mid:
                return mid[key]
            v = (verts[a] + verts[b]) * 0.5
            v /= np.linalg.norm(v)
            mid[key] = len(vlist)
            vlist.append(v)
            return mid[key]

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            flist += [
                [a, ab, ca],
                [b, bc, ab],
                [c, ca, bc],
                [ab, bc, ca],
            ]

        verts = np.asarray(vlist, dtype=np.float32)
        faces = np.asarray(flist, dtype=np.int64)

    return verts


# --------------------------------------
# MAIN GENERATOR
# ----------------------------------

def hr_lr_random_res_generator(
    training_dir,
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
    device="cpu",
    njobs=1,
    return_params=False,
    prob_ang_subsample=0.3,
    ang_min_dirs=4,
    ang_max_dirs=6,
    ang_reg_lambda=1e-3,
    debug_ang_subsample=False,
    debug_dir=None,
    max_debug_save=4,
):


    image_list = glob.glob(
        os.path.join(training_dir, "*/sh_coefficients_b*_masked.nii.gz")
    )
    n_training = len(image_list)
    print(f"Found {n_training} cases for training in {training_dir}")
    if n_training == 0:
        raise ValueError(f"No training SH files found in {training_dir}")

    # Padding around crops to avoid edge effects
    padsize = 4
    crop_size = crop_size + 2 * padsize

    # Make crop size iterable
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size, crop_size]
    crop_size = np.array(crop_size, dtype=np.int32)

    # Precompute coord grid
    xx, yy, zz = np.meshgrid(
        range(crop_size[0]),
        range(crop_size[1]),
        range(crop_size[2]),
        sparse=False,
        indexing="ij",
    )
    cx, cy, cz = (crop_size - 1) / 2.0
    xc = torch.tensor(xx - cx, device=device)
    yc = torch.tensor(yy - cy, device=device)
    zc = torch.tensor(zz - cz, device=device)
    _ = (xc, yc, zc)  # silence lints if unused


    # Precompute icosphere
    ico_level = 1
    ico_dirs_np = icosahedral_sphere(level=ico_level)  # (42,3)
    ico_dirs = torch.tensor(ico_dirs_np, dtype=torch.float32, device=device)
    Y_full = _mrtrix_real_sh_basis(ico_dirs, lmax=2, device=device)
    n_dirs_full, n_sh = Y_full.shape
    if n_sh != 6:
        raise RuntimeError(f"_mrtrix_real_sh_basis(lmax=2) returned n_sh={n_sh}, expected 6.")

    #clamp angular dir counts
    ang_min_dirs = max(1, int(ang_min_dirs))
    ang_max_dirs = max(ang_min_dirs, int(ang_max_dirs))

    debug_counter = 0

    #################
    while True:

        index = np.random.randint(n_training)
        sh_vol, aff = load_volume(image_list[index])
        sh_vol = np.squeeze(sh_vol).astype(np.float32)

        parentdir = os.path.dirname(image_list[index])
        lowb, _ = load_volume(os.path.join(parentdir, "mean_b0_synthstripped.nii.gz"))
        lowb = np.squeeze(lowb).astype(np.float32)[..., np.newaxis]

        if sh_vol.ndim != 4 or sh_vol.shape[-1] != 28:
            raise ValueError(f"Expected SH coeffs shape (*,28), got {sh_vol.shape}")

        # Kdiscard b0 for SH operations
        sh_l2 = sh_vol[..., :6]


        vol = np.concatenate([lowb, sh_l2], axis=-1)
        vol = torch.tensor(vol, device=device)

        # @TODO it shouldn't be spitting out NaNs
        vol[torch.isnan(vol)] = 0.0
        vol[torch.isinf(vol)] = 0.0

        # random crop
        vol_cropped = random_crop(vol, crop_size.tolist()).float()

        # random SH rotation
        if prob_rotate > 0 and random.random() < prob_rotate:
            alpha = np.random.uniform(-rotation_bounds, rotation_bounds)
            beta  = np.random.uniform(-rotation_bounds, rotation_bounds)
            gamma = np.random.uniform(-rotation_bounds, rotation_bounds)

            patch_low = 15
            patch_high = 32
            px = np.random.randint(patch_low, patch_high)
            py = np.random.randint(patch_low, patch_high)
            pz = np.random.randint(patch_low, patch_high)
            patch_size = (px, py, pz)

            spacing = np.random.uniform(2.0, float(max(px, py, pz)) / 2.0)
            warp_scale = np.random.uniform(1.0, float(max(px, py, pz)) / 7.0)

            vol_rot_np = rotate_sh_volume(
                vol_cropped.detach().cpu().numpy(),
                (alpha, beta, gamma),
                rotate=True,
                deform_patch=True,
                apply_random_drift=True,
                add_noise=False,
                patch_size=patch_size,
                drift_patch_size=(30, 30, 30),
                spacing=spacing,
                warp_scale=warp_scale,
            )
            vol_cropped = torch.tensor(
                vol_rot_np.astype(np.float32), device=device
            ).float()
            vol_cropped[torch.isnan(vol_cropped)] = 0.0
            vol_cropped[torch.isinf(vol_cropped)] = 0.0

        # noramlize with interquartile range (z-scoring too unstable)
        vol_cropped = sh_norm(vol_cropped, l0_index=1)
        vol_cropped = torch.clamp(vol_cropped, min=-1.0, max=1.0)

        # random gamma field injection
        gamma_l0 = torch.exp(
            torch.tensor(gamma_std, device=device) * torch.randn([1], device=device)
        ).float()
        gamma_lowb = torch.exp(
            torch.tensor(gamma_std, device=device) * torch.randn([1], device=device)
        ).float()

        hr_gamma = vol_cropped.detach().clone()
        max_lowb = torch.max(hr_gamma[..., 0]).clamp(min=1e-6)
        max_l0   = torch.max(hr_gamma[..., 1]).clamp(min=1e-6)
        hr_gamma[..., 0] = (hr_gamma[..., 0] / max_lowb) ** gamma_lowb
        hr_gamma[..., 1] = (hr_gamma[..., 1] / max_l0) ** gamma_l0


        # simple global intensity scale (i.e., glob bias)
        npoints = np.random.randint(1 + bf_maxsize)
        if npoints == 0:
            bias_lowb = torch.ones(1, device=device)
            bias_l0   = bias_lowb
        else:
            stddev_lowb = bf_std_max * torch.rand([1], device=device)
            stddev_l0   = bf_std_max * torch.rand([1], device=device)

            lr_bf_lowb = stddev_lowb * torch.randn([npoints, npoints, npoints], device=device)
            lr_bf_l0 = stddev_l0 * torch.randn([npoints, npoints, npoints], device=device)

            factor = crop_size.astype(np.float32) / float(npoints)
            bias_lowb = torch.exp(myzoom_torch(lr_bf_lowb, factor, device=device)).float()
            bias_l0 = torch.exp(myzoom_torch(lr_bf_l0, factor, device=device)).float()

        hr_bias = hr_gamma.detach().clone()
        hr_bias[..., 0] = hr_gamma[..., 0] * bias_lowb
        hr_bias[..., 1] = hr_gamma[..., 1] * bias_l0


        # rand direction-specific bias with low-rank mixing
        if random.random() < 0.15:
            hr_bias[..., 1:] = rand_lowrank_mix(hr_bias[..., 1:], rank=2, scale=0.04)


        # random SH channel dropout (don't need for onlu l=2!)
        randd = random.random()
        if randd < prob_dropout / 2.0:
            hr_bias[..., 2:] = 0.0
        elif randd > 1.0 - (prob_dropout / 2.0):
            hr_bias[..., 1:] = 0.0


        # Angular subsampling with icosphere projection/de-projection
        target = hr_bias.detach().clone()
        hr_for_lr = hr_bias

        if prob_ang_subsample > 0.0 and random.random() < prob_ang_subsample:
            sh = hr_for_lr[..., 1:]
            V = sh.shape[0] * sh.shape[1] * sh.shape[2]
            sh_flat = sh.reshape(V, n_sh)

            # choose random dir subset on ico
            n_dirs = random.randint(ang_min_dirs, ang_max_dirs)
            n_dirs = max(1, min(n_dirs, n_dirs_full))
            idx = torch.randperm(n_dirs_full, device=device)[:n_dirs]
            Y_sub = Y_full[idx, :]  # (n new dears,6)

            signal = sh_flat @ Y_sub.T

            # sprinkle in a wee bit of noise in signal space because why not
            # since for large enough numbers of subset directions (>=6) the problem is 
            # overdetermined anyways and we pretty but get back the same SH signal that we put in!
            if random.random() < 0.25:
                ang_noise_std = 0.02
                signal = signal + ang_noise_std * torch.randn_like(signal)

            # regularized least-squares refit
            YtY = Y_sub.T @ Y_sub  # (6,6)
            regI = ang_reg_lambda * torch.eye(
                n_sh, device=device, dtype=YtY.dtype
            )
            P = torch.linalg.solve(YtY + regI, Y_sub.T)
            sh_crappy_flat = signal @ P.T
            sh_crappy = sh_crappy_flat.reshape_as(sh)

            ##################################################
            if (debug_ang_subsample and debug_dir is not None and debug_counter < max_debug_save):
                os.makedirs(debug_dir, exist_ok=True)

                vol_pre = torch.cat(
                    [hr_for_lr[..., 0:1], sh], dim=-1
                ).detach().cpu().numpy()
                vol_post = torch.cat(
                    [hr_for_lr[..., 0:1], sh_crappy], dim=-1
                ).detach().cpu().numpy()

                nib.save(
                    nib.Nifti1Image(vol_pre, aff),
                    os.path.join(
                        debug_dir, f"debug_sh_pre_ang_{debug_counter:03d}.nii.gz"
                    ),
                )
                nib.save(
                    nib.Nifti1Image(vol_post, aff),
                    os.path.join(
                        debug_dir, f"debug_sh_post_ang_{debug_counter:03d}.nii.gz"
                    ),
                )
                debug_counter += 1

            # Update SH part used for LR simulation
            hr_for_lr = torch.cat(
                [hr_for_lr[..., 0:1], sh_crappy], dim=-1
            )

        #############
        # blur + resample of LR sample
        hr_lr_clone = hr_for_lr.detach().clone()
        blurred = hr_lr_clone[None, None, :]

        blurred[torch.isnan(blurred)] = 0.0
        blurred[torch.isinf(blurred)] = 0.0

        # per-axis ratios chosen uniformly in [lowres_min, lowres_max],
        ratios = lowres_min + (lowres_max - lowres_min) * np.random.rand(3)
        ratios = crop_size / (np.round(crop_size / ratios))

        for d in range(3):
            ratio = ratios[d]
            # bring the spatial axis d to the front so the kernel acts along it
            blurred = blurred.permute(0, 1, 4, 2, 3, 5)
            if ratio > 1:
                fraction = 0.45 + 0.4 * np.random.rand(1)
                sigma = float(fraction * ratio)
                kernel = torch.tensor(make_gaussian_kernel(sigma), dtype=torch.float32, device=device)[None, None, :, None, None]
                for c in range(blurred.shape[-1]):
                    blurred[..., c] = torch.conv3d(
                        blurred[..., c],
                        kernel,
                        stride=1,
                        padding=[int((kernel.shape[2] - 1) / 2), 0, 0],
                    )
        blurred = torch.squeeze(blurred)

        # Downsample -> LR
        lr = myzoom_torch(blurred, 1 / ratios, device=device)

        # Noise in low-res domain
        noise_std = noise_std_min + (noise_std_max - noise_std_min) * torch.rand([1], device=device)
        lr_noisy = lr + noise_std * torch.randn(lr.shape, device=device)

        # HR target is the bias-corrupted but angularly clean SH (7-ch)
        # Upsample back to HR grid for network input
        input_vol = myzoom_torch(lr_noisy, ratios, device=device)

        input_vol = input_vol.float()
        target = target.float()

        input_vol[torch.isnan(input_vol)] = 0.0
        input_vol[torch.isinf(input_vol)] = 0.0
        input_vol = torch.clamp(input_vol, min=-1.0, max=1.0)

        target[torch.isnan(target)] = 0.0
        target[torch.isinf(target)] = 0.0
        target = torch.clamp(target, min=-1.0, max=1.0)

        # to (C, X, Y, Z)
        input_vol = input_vol.permute(3, 0, 1, 2)
        target = target.permute(3, 0, 1, 2)

        # remove pad
        input_vol = unpad_tensor(input_vol, padsize)
        target = unpad_tensor(target, padsize)

        if return_params:
            ratios_t = torch.tensor(ratios, dtype=torch.float32, device=device)
            yield input_vol, target, ratios_t
        else:
            yield input_vol, target

