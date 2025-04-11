#!/usr/bin/env python3

import numpy as np
import nibabel as nib
import spherical
import quaternionic
import scipy.linalg
from scipy.ndimage import map_coordinates


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
    import argparse

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
    parser.add_argument("--check_global_jacobian", type=bool, default=True,
        help="Perform a check on negative jacobian determinants and discard the displacement field if ")
    args = parser.parse_args()


    sh_img = nib.load(args.in_sh)
    sh_data = sh_img.get_fdata(dtype=np.float32)
    affine_in = sh_img.affine
    X, Y, Z, nCoeffs = sh_data.shape

    if nCoeffs != 28:
        raise ValueError(f"Expected 28 SH coeffs for even-l up to lmax=6, got {nCoeffs}.")

    disp_field = generate_bspline_random_displacement_field(
        (X, Y, Z),
        spacing=args.spacing,
        warp_scale=args.warp_scale
    )

    # discard def fields with negative Jacs to avoid implausible
    # v1 computations
    if args.check_global_jacobian:
        valid_det = check_jacobian_all_positive(disp_field)
        if not valid_det:
            #print("Warning: negative or zero determinant found! Skipping...")
            warped_sh = sh_data
        else:
            warped_sh = warp_and_reorient_sh_volume_evenl(sh_data, disp_field, lmax=args.lmax)


    out_img = nib.Nifti1Image(warped_sh, affine_in)
    nib.save(out_img, args.out_sh)
    #print(f"Saved warped SH to: {args.out_sh}")

