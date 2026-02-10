import math
import random
import time
from typing import Tuple

import numpy as np
import nibabel as nib
import pyshtools as pysh
import scipy.linalg
import scipy.ndimage as ndi
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation

########################
# globs
SQRT2 = math.sqrt(2.0)
INV_SQRT2 = 1.0 / SQRT2

# We now work in an l2 world by default
LMAX = 2
INTERP_ORDER = 3
PAD_FRACTION = 0.1
N_JOBS = 30



# MRtrix real SH to pyshtools "cilm" reps

#number of even-l real SH coeff
def _n_even_sh_coeffs(lmax):
    n = 0
    for l in range(0, lmax + 1, 2):
        n += 2 * l + 1
    return n


# Convert real SH coefficients (MRtrix ordering) to pyshtools cilm
def mrtrix2cilm(vec, lmax=LMAX):

    vec = np.asarray(vec, dtype=np.float32)
    ncoef = _n_even_sh_coeffs(lmax)
    if vec.shape[0] != ncoef:
        raise ValueError(
            f"mrtrix2cilm: expected {ncoef} coefficients for lmax={lmax}, "
            f"got {vec.shape[0]}"
        )

    out = np.zeros((2, lmax + 1, lmax + 1), np.float32)
    k = 0
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            v = vec[k]
            k += 1
            if m == 0:
                out[0, l, 0] = v
            elif m > 0:
                out[0, l, m] = v * INV_SQRT2
            else:  #m<0
                out[1, l, -m] = INV_SQRT2 * (-1) ** (m + 1) * v
    return out


# converts pyshtools cilm back to mrtrix real SH
def cilm2mrtrix(cilm, lmax=LMAX):

    if cilm.shape[0] != 2 or cilm.shape[1] < lmax + 1 or cilm.shape[2] < lmax + 1:
        raise ValueError(f"cilm2mrtrix: unexpected cilm shape {cilm.shape} for lmax={lmax}")

    ncoef = _n_even_sh_coeffs(lmax)
    out = np.empty(ncoef, np.float32)
    k = 0
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            if m == 0:
                out[k] = cilm[0, l, 0]
            elif m > 0:
                out[k] = SQRT2 * cilm[0, l, m]
            else:
                out[k] = SQRT2 * (-1) ** (m + 1) * cilm[1, l, -m]
            k += 1
    return out


# Jacobian mapping for warps for U helper

def _jacobian_field(disp):

    # forward diff except far (backward there)
    du_dx = np.empty_like(disp[..., 0])
    dv_dx = np.empty_like(disp[..., 0])
    dw_dx = np.empty_like(disp[..., 0])

    du_dx[:-1] = disp[1:,  ..., 0] - disp[:-1, ..., 0]
    dv_dx[:-1] = disp[1:,  ..., 1] - disp[:-1, ..., 1]
    dw_dx[:-1] = disp[1:,  ..., 2] - disp[:-1, ..., 2]
    du_dx[-1]  = disp[-1, ..., 0] - disp[-2, ..., 0]
    dv_dx[-1]  = disp[-1, ..., 1] - disp[-2, ..., 1]
    dw_dx[-1]  = disp[-1, ..., 2] - disp[-2, ..., 2]

    du_dy = np.empty_like(disp[..., 0])
    dv_dy = np.empty_like(disp[..., 0])
    dw_dy = np.empty_like(disp[..., 0])

    du_dy[:, :-1] = disp[:, 1:, ..., 0] - disp[:, :-1, ..., 0]
    dv_dy[:, :-1] = disp[:, 1:, ..., 1] - disp[:, :-1, ..., 1]
    dw_dy[:, :-1] = disp[:, 1:, ..., 2] - disp[:, :-1, ..., 2]
    du_dy[:, -1]  = disp[:, -1, ..., 0] - disp[:, -2, ..., 0]
    dv_dy[:, -1]  = disp[:, -1, ..., 1] - disp[:, -2, ..., 1]
    dw_dy[:, -1]  = disp[:, -1, ..., 2] - disp[:, -2, ..., 2]

    du_dz = np.empty_like(disp[..., 0])
    dv_dz = np.empty_like(disp[..., 0])
    dw_dz = np.empty_like(disp[..., 0])

    du_dz[:, :, :-1] = disp[:, :, 1:, 0] - disp[:, :, :-1, 0]
    dv_dz[:, :, :-1] = disp[:, :, 1:, 1] - disp[:, :, :-1, 1]
    dw_dz[:, :, :-1] = disp[:, :, 1:, 2] - disp[:, :, :-1, 2]
    du_dz[:, :, -1]  = disp[:, :, -1, 0] - disp[:, :, -2, 0]
    dv_dz[:, :, -1]  = disp[:, :, -1, 1] - disp[:, :, -2, 1]
    dw_dz[:, :, -1]  = disp[:, :, -1, 2] - disp[:, :, -2, 2]


    J = np.stack(
        [
            np.stack([du_dx, du_dy, du_dz], axis=-1),
            np.stack([dv_dx, dv_dy, dv_dz], axis=-1),
            np.stack([dw_dx, dw_dy, dw_dz], axis=-1),
        ],
        axis=-2,
    )

    return J  # (X, Y, Z, 3, 3)


# check if jacobian is not very negative (i.e., no folding in)
def _jacobian_ok(disp, eps=-10.0):

    J = _jacobian_field(disp)
    det = np.linalg.det(J + np.eye(3)) # broadcasts

    return np.all(det > eps)


# cosine tapering
def _cos_taper(length, blend):
    ramp = np.ones(length, np.float32)
    if blend > 0:
        k = np.arange(blend, dtype=np.float32) / float(blend)
        ramp[:blend] = 0.5 * (1.0 - np.cos(math.pi * k))
        ramp[-blend:] = ramp[:blend][::-1]
    return ramp


# Build a local displacement field for a random patch inside the volume
def make_patch_displacement(
    vol_shape,
    patch_size=(12, 12, 12),
    spacing=4.0,
    warp_scale=2.0,
    blend=2,
    mode="both",
    max_tries=30,
):

    X, Y, Z = vol_shape
    px, py, pz = patch_size

    for _ in range(max_tries):
        x0 = random.randint(0, max(X - px, 1))
        x1 = x0 + px
        y0 = random.randint(0, max(Y - py, 1))
        y1 = y0 + py
        z0 = random.randint(0, max(Z - pz, 1))
        z1 = z0 + pz

        disp_patch = np.zeros((px, py, pz, 3), np.float32)

        # random low-res field upsampled with cubic interp
        if mode in ("curvy", "both"):
            ctrl_shape = (
                max(1, int(px / spacing)),
                max(1, int(py / spacing)),
                max(1, int(pz / spacing)),
            )
            ctrl = warp_scale * np.random.randn(*ctrl_shape, 3).astype(np.float32)

            # coords in control grid
            cx = np.linspace(0, ctrl_shape[0] - 1, px, dtype=np.float32)
            cy = np.linspace(0, ctrl_shape[1] - 1, py, dtype=np.float32)
            cz = np.linspace(0, ctrl_shape[2] - 1, pz, dtype=np.float32)
            gz, gy, gx = np.meshgrid(cz, cy, cx, indexing="ij")
            coords = np.stack([gz, gy, gx], axis=0)

            for c in range(3):
                vol_c = np.transpose(ctrl[..., c], (2, 1, 0))
                disp_patch[..., c] += ndi.map_coordinates(
                    vol_c, coords, order=3, mode="nearest"
                ).reshape(pz, py, px).transpose(2, 1, 0)

        if mode in ("shear", "both"):
            sx, sy, sz = [random.uniform(-0.25, 0.25) for _ in range(3)]
            S = np.array([[1, sx, sx], [sy, 1, sy], [sz, sz, 1]], np.float32)

            # coords in patch-native ordering (px, py, pz, 3)
            xx, yy, zz = np.meshgrid(np.arange(px), np.arange(py), np.arange(pz), indexing="ij")
            coords = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
            coords_flat = coords.reshape(-1, 3)

            # shear
            sheared_flat = coords_flat @ S.T
            diff = (sheared_flat - coords_flat).reshape(px, py, pz, 3)

            # same layout as disp_patch
            disp_patch += diff


        # taper at patch boundaries (zero displacement)
        rx = _cos_taper(px, blend)
        ry = _cos_taper(py, blend)
        rz = _cos_taper(pz, blend)
        mask = rx[:, None, None] * ry[None, :, None] * rz[None, None, :]
        disp_patch *= mask[..., None]

        if _jacobian_ok(disp_patch):
            disp = np.zeros((X, Y, Z, 3), np.float32)
            disp[x0:x1, y0:y1, z0:z1] = disp_patch
            return disp, (x0, x1, y0, y1, z0, z1)

    # give zero field if everythign fails
    return np.zeros((X, Y, Z, 3), np.float32), (0, 0, 0, 0, 0, 0)


# Bulk affine rotation

def bulk_affine_rotate(
    vol4d,
    euler_deg,
    order=3,
    pad_frac=0.15,
    n_jobs=N_JOBS,
):

    a, b, g = map(math.radians, euler_deg)
    Nx, Ny, Nz, C = vol4d.shape
    pad = int(pad_frac * max(Nx, Ny, Nz))
    volP = np.pad(vol4d, ((pad, pad), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0)

    #rotation matrix
    Rx = [
        [1, 0, 0],
        [0, math.cos(a), -math.sin(a)],
        [0, math.sin(a), math.cos(a)],
    ]
    Ry = [
        [math.cos(b), 0, math.sin(b)],
        [0, 1, 0],
        [-math.sin(b), 0, math.cos(b)],
    ]
    Rz = [
        [math.cos(g), -math.sin(g), 0],
        [math.sin(g), math.cos(g), 0],
        [0, 0, 1],
    ]
    Rvox = np.matmul(Rz, np.matmul(Ry, Rx))
    Avox = Rvox.T

    NxP, NyP, NzP, _ = volP.shape
    centreP = np.array([(NxP - 1) / 2, (NyP - 1) / 2, (NzP - 1) / 2])
    offsetP = centreP - Avox @ centreP

    def warp_one(chan):
        return ndi.affine_transform(chan, Avox, offset=offsetP, order=order, mode="constant", cval=0.0)

    t0 = time.perf_counter()
    warped = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(warp_one)(volP[..., k]) for k in range(C))
    _ = time.perf_counter() - t0  # slow for some reason

    vol_rot = np.stack(warped, axis=-1)
    return vol_rot[pad: pad + Nx, pad: pad + Ny, pad: pad + Nz, :]


# Rotate SH coefficients in-place (no spatial transform)
def sh_rotate_coeff(sh4d, euler_deg, lmax=LMAX):

    a, b, g = map(math.radians, euler_deg)
    ncoef = _n_even_sh_coeffs(lmax)
    if sh4d.shape[-1] != ncoef:
        raise ValueError(
            f"sh_rotate_coeff: last axis size {sh4d.shape[-1]} "
            f"!= expected {ncoef} for lmax={lmax}"
        )

    # Build rotation matrix in MRtrix basis via pyshtools once
    mr2 = [(l, m) for l in range(0, lmax + 1, 2) for m in range(-l, l + 1)]
    dj = pysh.rotate.djpi2(lmax)
    R = np.zeros((ncoef, ncoef), np.float32)

    for i, (l, m) in enumerate(mr2):
        e = np.zeros((2, lmax + 1, lmax + 1), np.float32)
        if m == 0:
            e[0, l, 0] = 1.0
        elif m > 0:
            e[0, l, m] = INV_SQRT2
        else:  # m < 0
            e[1, l, -m] = INV_SQRT2 * (-1) ** (m + 1)

        r = pysh.rotate.SHRotateRealCoef(e, [a, b, g], dj)
        col = np.empty(ncoef, np.float32)
        for j, (l2, m2) in enumerate(mr2):
            if m2 == 0:
                col[j] = r[0, l2, 0]
            elif m2 > 0:
                col[j] = SQRT2 * r[0, l2, m2]
            else:
                col[j] = SQRT2 * (-1) ** (m2 + 1) * r[1, l2, -m2]
        R[:, i] = col

    flat = sh4d.reshape(-1, ncoef)
    t0 = time.perf_counter()
    flat_rot = flat @ R.T
    _ = time.perf_counter() - t0  # SLOW

    return flat_rot.reshape(sh4d.shape)


# Apply local drift (i.e., mimic some weird biases) by just perturbing the rotated coeffs and unrotating

def drift_sh_coeffs(
    vol,
    EULER_DEG,
    LMAX,
    patch_size=(30, 30, 30),
    eps_range=(0.05, 0.15),
):

    vol_out = vol.copy()  # chans are [lowb, l0, l2...]
    a, b, g = map(math.radians, EULER_DEG)
    X, Y, Z, _ = vol.shape
    px, py, pz = patch_size

    x0 = random.randint(0, max(X - px, 1))
    x1 = x0 + px
    y0 = random.randint(0, max(Y - py, 1))
    y1 = y0 + py
    z0 = random.randint(0, max(Z - pz, 1))
    z1 = z0 + pz

    # Take SH part 
    sh_patch = vol[x0:x1, y0:y1, z0:z1, 1:]

    # bulk rotate to a random frame
    sh_patch_rot = sh_rotate_coeff(sh_patch, EULER_DEG, lmax=LMAX)

    # small perturbation to l=2 band
    eps = random.uniform(*eps_range)
    if random.random() < 0.5:
        # nudge m = +-2
        sh_patch_rot[..., 1] += eps
        sh_patch_rot[..., 5] += -eps
    else:
        # nudge m = +-1
        sh_patch_rot[..., 2] += eps
        sh_patch_rot[..., 4] += -eps

    # rotate back
    sh_patch_drift = sh_rotate_coeff(sh_patch_rot, [-e for e in EULER_DEG], lmax=LMAX)

    vol_out[x0:x1, y0:y1, z0:z1, 1:] = sh_patch_drift
    return vol_out


# MAIN Warp + SH re-orientation
def apply_patch_warp_and_reorient(
    vol,
    disp,
    *,
    lmax=LMAX,
    noise_std_range=(0.0, 0.02),
):

    X, Y, Z, C = vol.shape

    # spatial warp (linear)
    gz, gy, gx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    coords = np.array(
        [
            gz - disp[..., 2],
            gy - disp[..., 1],
            gx - disp[..., 0],
        ]
    )

    warped = np.empty_like(vol)
    for c in range(C):
        vol_c = np.transpose(vol[..., c], (2, 1, 0))
        warped[..., c] = ndi.map_coordinates(vol_c, coords, order=1, mode="nearest").transpose(2, 1, 0)

    mask = np.any(disp != 0, axis=-1)
    if np.any(mask):
        J = _jacobian_field(disp)
        dj = pysh.rotate.djpi2(lmax)

        for (i, j, k) in zip(*np.where(mask)):
            R = scipy.linalg.polar(np.eye(3) + J[i, j, k])[0]  # rotation
            ang = Rotation.from_matrix(R).as_euler("ZYZ", degrees=False)
            # SH coeffs at voxel (excluding b0)
            sh_vec = warped[i, j, k, 1:]
            cilm = mrtrix2cilm(sh_vec, lmax=lmax)
            cilm_r = pysh.rotate.SHRotateRealCoef(cilm, ang, dj)
            warped[i, j, k, 1:] = cilm2mrtrix(cilm_r, lmax=lmax)

    # 3 add Gaussian noise only inside ROI
    if noise_std_range is not None:
        low, high = noise_std_range
        sigma = random.uniform(low, high)
        warped[mask] += sigma * np.random.randn(*warped[mask].shape).astype(np.float32)

    return warped


# deform + drift help

def rotate_sh_volume(
    vol_input,
    EULER_DEG,
    rotate=True,
    deform_patch=True,
    apply_random_drift=True,
    add_noise=False,
    patch_size=(20, 20, 20),
    drift_patch_size=(30, 30, 30),
    spacing=8.0,
    warp_scale=2.0,
):

    vol = vol_input.copy().astype(np.float32)

    if rotate:
        # 1) global affine rotation of entire 7-channel volume
        vol_rot = bulk_affine_rotate(
            vol,
            EULER_DEG,
            order=INTERP_ORDER,
            pad_frac=PAD_FRACTION,
            n_jobs=N_JOBS,
        )

        # rotate SH coefficients 
        sh_rot = sh_rotate_coeff(vol_rot[..., 1:], EULER_DEG, lmax=LMAX)
        vol_rot[..., 1:] = sh_rot
    else:
        vol_rot = vol

    # random local warp + SH reorientation in patch
    truly_add_noise = add_noise and (random.random() < 0.1)

    if deform_patch:
        noise_std_range = (0.0, 0.02) if truly_add_noise else None
        disp, _ = make_patch_displacement(
            vol_rot.shape[:3],
            patch_size=patch_size,
            spacing=spacing,
            warp_scale=warp_scale,
            mode="both",
        )
        vol_rot = apply_patch_warp_and_reorient(vol_rot, disp, lmax=LMAX, noise_std_range=noise_std_range)

    # 4) Optional "drift" of l=2 band in another patch
    rand_drift = apply_random_drift and (random.random() < 0.1)
    if rand_drift:
        vol_rot = drift_sh_coeffs(
            vol_rot,
            EULER_DEG,
            LMAX,
            patch_size=drift_patch_size,
        )

    return vol_rot

