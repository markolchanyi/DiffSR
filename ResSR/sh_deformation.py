#!/usr/bin/env python3
import argparse
import numpy as np
import random
import nibabel as nib
import spherical
import quaternionic
import scipy.linalg
from scipy.ndimage import map_coordinates

def make_1d_ramp(size, boundary_blend):
    ramp = np.ones(size, dtype=np.float32)
    for i in range(boundary_blend):
        if i < size:
            ramp[i] = i / boundary_blend
            ramp[size - 1 - i] = i / boundary_blend
    # cont be greater than 1
    return np.clip(ramp, 0.0, 1.0)

def all_lm_pairs_up_to_lmax(lmax):
    """List (l,m) for l in [0..lmax], m in [-l..l]."""
    pairs = []
    for l in range(lmax+1):
        for m in range(-l, l+1):
            pairs.append((l,m))
    return pairs

def even_lm_pairs_up_to_lmax(lmax):
    """List (l,m) but only for even l in [0..lmax]."""
    pairs = []
    for l in range(0, lmax+1, 2):
        for m in range(-l, l+1):
            pairs.append((l,m))
    return pairs

def build_even_l_index_mappings(lmax):
    """
    Build arrays to map from the 28 even-l indices -> 49 full-l indices (and vice versa).
    Returns:
      map_even_to_full (length=28),
      map_full_to_even (length=49, or -1 if that slot is odd-l).
    """
    all_lm = all_lm_pairs_up_to_lmax(lmax)   # e.g. 49 for lmax=6
    even_lm = even_lm_pairs_up_to_lmax(lmax) # e.g. 28

    # Map (l,m)-> index for the full-l set
    dict_lm_to_full = { lm: i for i, lm in enumerate(all_lm) }

    map_even_to_full = []
    for i, lm in enumerate(even_lm):
        full_idx = dict_lm_to_full[lm]
        map_even_to_full.append(full_idx)

    # Optional reverse map
    map_full_to_even = [-1]*len(all_lm)
    for even_idx, lm in enumerate(even_lm):
        full_idx = dict_lm_to_full[lm]
        map_full_to_even[full_idx] = even_idx

    return map_even_to_full, map_full_to_even


def generate_even_l_arrays(lmax):
    """
    Return l_array_even, m_array_even for the *28* even-l modes
    (the typical MRtrix ordering).
    """
    lm_pairs = even_lm_pairs_up_to_lmax(lmax)
    l_array = np.array([lm[0] for lm in lm_pairs])
    m_array = np.array([lm[1] for lm in lm_pairs])
    return l_array, m_array

def real_to_complex_sh_coeffs_evenl(real_coeffs, m_array):
    """
    Convert real SH (even-l) to complex SH (even-l).
    real_coeffs: shape (..., 28)
    m_array: shape (28,)
    """
    sqrt2 = np.sqrt(2)
    cplx = np.zeros_like(real_coeffs, dtype=np.complex128)

    m_neg_mask = (m_array < 0)
    m_zero_mask = (m_array == 0)
    m_pos_mask = (m_array > 0)

    # Broadcast
    shape_diff = real_coeffs.ndim - 1
    m_brd = m_array.reshape((1,)*shape_diff + (len(m_array),))

    # Same logic you used before
    cplx += (1j * real_coeffs / sqrt2) * (m_neg_mask == True)
    cplx += real_coeffs * (m_zero_mask == True)
    cplx += (real_coeffs / sqrt2) * (m_pos_mask == True)

    return cplx

def complex_to_real_sh_coeffs_evenl(cplx_coeffs, m_array):
    """
    Convert complex SH (even-l) back to real SH (even-l).
    cplx_coeffs: shape (..., 28)
    m_array: shape (28,)
    """
    sqrt2 = np.sqrt(2)
    real_out = np.zeros_like(cplx_coeffs.real)

    m_neg_mask = (m_array < 0)
    m_zero_mask = (m_array == 0)
    m_pos_mask = (m_array > 0)

    shape_diff = cplx_coeffs.ndim - 1
    m_brd = m_array.reshape((1,)*shape_diff + (len(m_array),))

    real_out += sqrt2 * cplx_coeffs.imag * (m_neg_mask == True)
    real_out += cplx_coeffs.real * (m_zero_mask == True)
    real_out += sqrt2 * cplx_coeffs.real * (m_pos_mask == True)
    return real_out


def generate_bspline_random_displacement_field(vol_shape, spacing=16, warp_scale=5.0):
    """
    Create a smooth random displacement via a coarse grid of random control points
    and cubic interpolation.
    vol_shape: (X, Y, Z)
    spacing: control point spacing in voxels
    warp_scale: +/- amplitude of random shifts
    returns disp_field: [X, Y, Z, 3]
    """
    X, Y, Z = vol_shape
    nCx = int(np.ceil(X/spacing)) + 1
    nCy = int(np.ceil(Y/spacing)) + 1
    nCz = int(np.ceil(Z/spacing)) + 1

    # random control points
    control_field = np.random.uniform(
        low=-warp_scale, high=warp_scale,
        size=(nCx, nCy, nCz, 3)
    ).astype(np.float32)

    disp_field = np.zeros((X, Y, Z, 3), dtype=np.float32)

    # Build coordinate grid in control-point space
    grid_z, grid_y, grid_x = np.meshgrid(
        np.arange(Z), np.arange(Y), np.arange(X), indexing='ij'
    )
    grid_xf = grid_x.astype(np.float32) / spacing
    grid_yf = grid_y.astype(np.float32) / spacing
    grid_zf = grid_z.astype(np.float32) / spacing

    coords_stack = np.vstack([grid_zf.ravel(), grid_yf.ravel(), grid_xf.ravel()])

    # Upsample each channel with cubic interpolation
    from scipy.ndimage import map_coordinates
    for c in range(3):
        # Reorient the coarse field to (nCz, nCy, nCx)
        vol_coarse = np.transpose(control_field[..., c], (2,1,0))  # shape [nCz,nCy,nCx]
        interp_vals = map_coordinates(vol_coarse, coords_stack, order=3, mode='nearest')
        disp_field[..., c] = interp_vals.reshape((Z, Y, X)).transpose((2,1,0))

    return disp_field


def build_local_bspline_displacement(
    vol_shape,
    patch_min_corner,
    patch_size,
    spacing=16,
    warp_scale=2.0,
    boundary_blend=2
):
    """
    Create a B-spline random displacement (like your global version) 
    *only inside* a specified patch, 
    with a smooth taper to zero at the patch boundary (Dirichlet boundary).

    Args:
      vol_shape: tuple (X, Y, Z).
      patch_min_corner: (x0, y0, z0).
      patch_size: (px, py, pz).
      spacing: B-spline control-point spacing for the local patch.
      warp_scale: amplitude of random warp in the patch.
      boundary_blend: # of voxels to blend from 1.0->0.0 near patch edges.

    Returns:
      disp_local: shape [X, Y, Z, 3], 
                  zero outside patch, 
                  random B-spline inside patch with smooth boundary.
    """

    X, Y, Z = vol_shape
    x0, y0, z0 = patch_min_corner
    px, py, pz = patch_size

    disp_patch = generate_bspline_random_displacement_field(
        (px, py, pz),
        spacing=spacing,
        warp_scale=warp_scale
    )  # shape [px, py, pz, 3]

    # 2) Build a smooth boundary mask to ensure the displacement is zero at patch edges
    #    (Dirichlet boundary condition)
    blend_mask = np.ones((px, py, pz), dtype=np.float32)

    ramp_x = make_1d_ramp(px, boundary_blend)
    ramp_y = make_1d_ramp(py, boundary_blend)
    ramp_z = make_1d_ramp(pz, boundary_blend)

    for ix in range(px):
        for iy in range(py):
            for iz in range(pz):
                blend_mask[ix, iy, iz] = ramp_x[ix] * ramp_y[iy] * ramp_z[iz]

    # Apply the blend mask to disp_patch
    for c in range(3):
        disp_patch[..., c] *= blend_mask

    # 3) Insert disp_patch into a volume-sized array, zero outside
    disp_local = np.zeros((X, Y, Z, 3), dtype=np.float32)
    x1 = x0 + px
    y1 = y0 + py
    z1 = z0 + pz

    disp_local[x0:x1, y0:y1, z0:z1, :] = disp_patch

    return disp_patch
    #return disp_local


def build_local_shear_displacement(
    vol_shape,
    patch_min_corner,
    patch_size,
    shear_factors=(0.1, 0.0, 0.0),
    boundary_blend=2):
    """
    Create a local *pure shear* displacement field, zero outside patch,
    smoothly transitioning to zero at patch boundary.
    """

    X, Y, Z = vol_shape
    x0, y0, z0 = patch_min_corner
    px, py, pz = patch_size
    sx, sy, sz = shear_factors

    disp_local = np.zeros((X, Y, Z, 3), dtype=np.float32)

    # local coords
    xx = np.arange(px, dtype=np.float32)
    yy = np.arange(py, dtype=np.float32)
    zz = np.arange(pz, dtype=np.float32)
    zzg, yyg, xxg = np.meshgrid(zz, yy, xx, indexing='ij')  # shape [pz, py, px]

    # Shear matrix
    shear_mat = np.array([
        [1,  sx,  sx],
        [sy, 1,   sy],
        [sz, sz,  1 ]
    ], dtype=np.float32)

    local_coords = np.stack([xxg, yyg, zzg], axis=-1)  # [pz, py, px, 3], "patch space"
    shp = local_coords.shape
    coords_flat = local_coords.reshape(-1, 3)

    # Apply shear: new = shear_mat @ old
    sheared_flat = coords_flat @ shear_mat.T
    disp_patch = (sheared_flat - coords_flat).reshape(shp)  # [pz, py, px, 3]

    # boundary blend
    blend_mask = np.ones((pz, py, px), dtype=np.float32)

    ramp_x = make_1d_ramp(px, boundary_blend)
    ramp_y = make_1d_ramp(py, boundary_blend)
    ramp_z = make_1d_ramp(pz, boundary_blend)

    for iz in range(pz):
        for iy in range(py):
            for ix in range(px):
                blend_mask[iz, iy, ix] = ramp_x[ix] * ramp_y[iy] * ramp_z[iz]

    for c in range(3):
        disp_patch[..., c] *= blend_mask

    # Insert into disp_local at patch location
    x1 = x0 + px
    y1 = y0 + py
    z1 = z0 + pz

    # watch axis order carefully if needed:
    #disp_local[x0:x1, y0:y1, z0:z1, :] = disp_patch
    disp_local[x0:x1, y0:y1, z0:z1, :] = disp_patch.transpose((2, 1, 0, 3))

    return disp_patch.transpose((2, 1, 0, 3))
    #return disp_local


def compute_displacement_jacobian_field(displacement_field):
    """
    displacement_field: [X, Y, Z, 3]
    returns jac_field: [X, Y, Z, 3, 3]
    """
    X, Y, Z, _ = displacement_field.shape
    jac = np.zeros((X, Y, Z, 3, 3), dtype=displacement_field.dtype)

    disp_x = displacement_field[..., 0]
    disp_y = displacement_field[..., 1]
    disp_z = displacement_field[..., 2]

    # central diff along X
    jac[1:-1, :, :, 0, 0] = (disp_x[2:, :, :] - disp_x[:-2, :, :]) * 0.5
    jac[1:-1, :, :, 0, 1] = (disp_y[2:, :, :] - disp_y[:-2, :, :]) * 0.5
    jac[1:-1, :, :, 0, 2] = (disp_z[2:, :, :] - disp_z[:-2, :, :]) * 0.5

    # central diff along Y
    jac[:, 1:-1, :, 1, 0] = (disp_x[:, 2:, :] - disp_x[:, :-2, :]) * 0.5
    jac[:, 1:-1, :, 1, 1] = (disp_y[:, 2:, :] - disp_y[:, :-2, :]) * 0.5
    jac[:, 1:-1, :, 1, 2] = (disp_z[:, 2:, :] - disp_z[:, :-2, :]) * 0.5

    # central diff along Z
    jac[:, :, 1:-1, 2, 0] = (disp_x[:, :, 2:] - disp_x[:, :, :-2]) * 0.5
    jac[:, :, 1:-1, 2, 1] = (disp_y[:, :, 2:] - disp_y[:, :, :-2]) * 0.5
    jac[:, :, 1:-1, 2, 2] = (disp_z[:, :, 2:] - disp_z[:, :, :-2]) * 0.5

    return jac

def local_rotation_from_jacobian(jac_3x3):
    """
    jac_3x3: shape (3,3)
    T = I + jac
    do polar decomposition => R
    """
    T = np.eye(3, dtype=jac_3x3.dtype) + jac_3x3
    R, S = scipy.linalg.polar(T)  # R is pure rotation
    return R


def check_jacobian_all_positive(disp_field: np.ndarray) -> bool:
    """
    Returns False if any voxel's determinant (I + jac) <= 0.
    """
    jac = compute_displacement_jacobian_field(disp_field)  # shape [X,Y,Z,3,3]
    # T(x) = I + jac(x)
    # We'll flatten to do determinant in vectorized form
    shapeXYZ = jac.shape[:3]
    jac_flat = jac.reshape(-1, 3, 3)
    # Build T(x) for each voxel
    T_flat = np.empty_like(jac_flat)
    for i in range(jac_flat.shape[0]):
        T_flat[i] = np.eye(3, dtype=jac_flat.dtype) + jac_flat[i]
    dets = np.linalg.det(T_flat)
    return np.all(dets > 0)


class EvenLRotator:
    """
    A small helper that:
     - knows how to embed 28 complex coeffs into a 49 array,
     - calls spherical.Wigner.rotate,
     - extracts 28 again.
    """
    def __init__(self, lmax=6):
        self.lmax = lmax
        # build index mappings
        self.map_even_to_full, self.map_full_to_even = build_even_l_index_mappings(lmax)
        self.n_full = len(all_lm_pairs_up_to_lmax(lmax))   # 49
        self.n_even = len(even_lm_pairs_up_to_lmax(lmax))  # 28
        self.wigner = spherical.Wigner(lmax)

    def rotate(self, cplx_even_28, R_3x3):
        """
        cplx_even_28: shape (28,) complex array (even-l only)
        R_3x3: 3x3 rotation
        returns cplx_even_28 rotated
        """
        import quaternionic
        quat = quaternionic.array.from_rotation_matrix(R_3x3)

        # 1) embed into 49
        cplx_full_49 = np.zeros(self.n_full, dtype=np.complex128)
        for i in range(self.n_even):
            fidx = self.map_even_to_full[i]
            cplx_full_49[fidx] = cplx_even_28[i]

        # 2) spherical.Modes
        modes_in = spherical.Modes(cplx_full_49, lmax=self.lmax, spin_weight=0)

        # 3) rotate
        modes_out = self.wigner.rotate(modes_in, quat)
        cplx_rotated_full_49 = modes_out.ndarray

        # 4) extract even-l
        cplx_rotated_even_28 = np.zeros(self.n_even, dtype=np.complex128)
        for i in range(self.n_even):
            fidx = self.map_even_to_full[i]
            cplx_rotated_even_28[i] = cplx_rotated_full_49[fidx]

        return cplx_rotated_even_28


def warp_and_reorient_sh_volume_evenl(
    real_sh_coeffs_4d,
    displacement_field_4d,
    lmax=6
):
    """
    real_sh_coeffs_4d: shape [X, Y, Z, 28] for lmax=6 even-l
    displacement_field_4d: shape [X, Y, Z, 3]
    returns warped_sh of same shape
    """
    X, Y, Z, nCoeffs = real_sh_coeffs_4d.shape
    jac_field = compute_displacement_jacobian_field(displacement_field_4d)

    # Prep an EvenLRotator
    rotator = EvenLRotator(lmax)
    # Also get the arrays for real<->complex conversions
    l_array_even, m_array_even = generate_even_l_arrays(lmax)

    # Output
    warped_sh = np.zeros_like(real_sh_coeffs_4d, dtype=np.float32)

    # Build a coordinate grid in output space
    grid_z, grid_y, grid_x = np.meshgrid(
        np.arange(Z), np.arange(Y), np.arange(X), indexing='ij'
    )
    flat_xp = grid_x.ravel()
    flat_yp = grid_y.ravel()
    flat_zp = grid_z.ravel()
    Nvox = X*Y*Z

    disp_x = displacement_field_4d[..., 0].ravel()
    disp_y = displacement_field_4d[..., 1].ravel()
    disp_z = displacement_field_4d[..., 2].ravel()

    # Build input coords for map_coordinates
    input_coords = np.zeros((3, Nvox), dtype=np.float32)
    for idx in range(Nvox):
        ixp = flat_xp[idx]
        iyp = flat_yp[idx]
        izp = flat_zp[idx]
        dx = disp_x[idx]
        dy = disp_y[idx]
        dz = disp_z[idx]
        x_in = ixp - dx
        y_in = iyp - dy
        z_in = izp - dz
        # map_coordinates wants (z,y,x)
        input_coords[0, idx] = z_in
        input_coords[1, idx] = y_in
        input_coords[2, idx] = x_in

    # Interpolate each SH coeff channel
    interpolated_SH = np.zeros((Nvox, nCoeffs), dtype=np.float32)
    for c in range(nCoeffs):
        vol_c = real_sh_coeffs_4d[..., c]  # shape [X,Y,Z]
        vol_c_t = np.transpose(vol_c, (2,1,0))  # shape [Z,Y,X]
        vals_c = map_coordinates(vol_c_t, input_coords, order=1, mode='nearest')
        interpolated_SH[:, c] = vals_c

    # Now rotate each voxel's SH
    for idx in range(Nvox):
        ixp = flat_xp[idx]
        iyp = flat_yp[idx]
        izp = flat_zp[idx]
        jac_3x3 = jac_field[ixp, iyp, izp, :, :]
        R_3x3 = local_rotation_from_jacobian(jac_3x3)

        # 1) real->complex (28)
        sh_in_28 = interpolated_SH[idx, :]
        cplx_in_28 = real_to_complex_sh_coeffs_evenl(sh_in_28, m_array_even)

        # 2) rotate
        cplx_out_28 = rotator.rotate(cplx_in_28, R_3x3)

        # 3) complex->real (28)
        sh_out_28 = complex_to_real_sh_coeffs_evenl(cplx_out_28, m_array_even)

        # put in output
        warped_sh[ixp, iyp, izp, :] = sh_out_28

    return warped_sh


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="B-spline random warp + local Wigner-D rotation for even-l SH (28 coeffs)."
    )
    parser.add_argument("--in_sh", required=True,
        help="Path to input SH NIfTI [X,Y,Z,28] (lmax=6, even-l).")
    parser.add_argument("--out_sh", required=True,
        help="Path to output warped+reoriented SH NIfTI [X,Y,Z,28].")
    parser.add_argument("--lmax", type=int, default=6, help="Max SH order (only even-l used).")
    parser.add_argument("--spacing", type=int, default=16,
        help="Spacing for B-spline control points (vox).")
    parser.add_argument("--warp_scale", type=float, default=5.0,
        help="Amplitude of random warp at control points (vox).")
    parser.add_argument("--check_global_jacobian", type=bool, required=True,
        help="Perform a check on negative jacobian determinants and discard the displacement field if ")
    parser.add_argument("--patch_only", type=int, required=True,
        help="Only calculate the defomation field on a sub_patch.")
    args = parser.parse_args()


    sh_img = nib.load(args.in_sh)
    sh_data = sh_img.get_fdata(dtype=np.float32)
    affine_in = sh_img.affine
    X, Y, Z, nCoeffs = sh_data.shape

    if nCoeffs != 28:
        raise ValueError(f"Expected 28 SH coeffs for even-l up to lmax=6, got {nCoeffs}.")
    #print("patch only arg is: ", args.patch_only)
    if not args.patch_only:
        #print("DOING GLOB DEFORMATION")
        max_attempts = 30
        attempt = 0
        while attempt < max_attempts:
            #print("attempt global: ", attempt)
            disp_field = generate_bspline_random_displacement_field(
                (X, Y, Z),
                spacing=args.spacing,
                warp_scale=args.warp_scale)
            attempt += 1
            if check_jacobian_all_positive(disp_field):
                break
        else:
            #print("max attempts of global disp field generation reached")
            disp_field = np.zeros((X,Y,Z,3), dtype=np.float32)

    if args.patch_only:
        #disp_field = np.zeros_like(disp_field)

        px = np.random.randint(8, 18)
        py = np.random.randint(8, 18)
        pz = np.random.randint(8, 18)
        x0 = np.random.randint(0, max(X - px, 1))
        y0 = np.random.randint(0, max(Y - py, 1))
        z0 = np.random.randint(0, max(Z - pz, 1))

        x1 = x0 + px
        y1 = y0 + py
        z1 = z0 + pz

        #meth = np.random.choice(["bspline", "shear", "both"])
        meth = "both"

        max_attempts = 100
        attempt = 0
        while attempt < max_attempts:
            #print("patch attempt", attempt)
            #disp_local = np.zeros_like(disp_field)
            disp_local = np.zeros((px,py,pz,3), dtype=np.float32)

            if meth in ["bspline", "both"]:
                # Local B-spline
                #print("building spline")
                local_disp_bspline = build_local_bspline_displacement(
                    vol_shape=(X, Y, Z),
                    patch_min_corner=(x0, y0, z0),
                    patch_size=(px, py, pz),
                    spacing=np.random.uniform(1, np.max((px,py,pz))/2),
                    warp_scale=np.random.uniform(1,2.5),
                    boundary_blend=2)
                disp_local += local_disp_bspline

            if meth in ["shear", "both"]:
                # Local shear
                #print("building shear")
                sx = np.random.uniform(-0.25, 0.25)
                sy = np.random.uniform(-0.25, 0.25)
                sz = np.random.uniform(-0.25, 0.25)
                local_disp_shear = build_local_shear_displacement(
                    vol_shape=(X, Y, Z),
                    patch_min_corner=(x0, y0, z0),
                    patch_size=(px, py, pz),
                    shear_factors=(sx, sy, sz),
                    boundary_blend=2)
                disp_local += local_disp_shear

            disp_field = disp_local
            attempt += 1

            if check_jacobian_all_positive(disp_local):
                break
        else:
            #print("max amount of tries for local field reached")
            disp_field = np.zeros_like(disp_local)
        # simply add loc disp to glob disp
        #disp_field += disp_local


    #valid_det = check_jacobian_all_positive(disp_field)
    #if not valid_det:
    #    print(" ")
    #    print("bad deterninant!!")
    #    print(" ")
    if args.patch_only:
        warped_sh = sh_data
        warped_sh_patch = warp_and_reorient_sh_volume_evenl(sh_data[x0:x1, y0:y1, z0:z1, :], disp_field, lmax=args.lmax)
        warped_sh[x0:x1, y0:y1, z0:z1, :] = warped_sh_patch
    else:
        warped_sh = warp_and_reorient_sh_volume_evenl(sh_data, disp_field, lmax=args.lmax)

    # discard def fields with negative Jacs to avoid implausible
    # v1 computations
    #if args.check_global_jacobian:
    #    valid_det = check_jacobian_all_positive(disp_field)
    #    if not valid_det:
            #print("Warning: negative or zero determinant found! Skipping...")
    #        warped_sh = sh_data

    #if args.patch_only:
    #noise_std_min = 0.0
    #noise_std_max = 0.01
    #noise_std = noise_std_min + (noise_std_max - noise_std_min) * * np.random.rand()
    #warped_sh_patch_noisy = warped_sh_patch + noise_std * np.random.randn(*warped_sh_patch.shape)

    out_img = nib.Nifti1Image(warped_sh, affine_in)
    nib.save(out_img, args.out_sh)

