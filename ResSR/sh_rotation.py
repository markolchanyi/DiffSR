import numpy as np
import quaternionic
import spherical
import nibabel as nib
import time
from scipy import ndimage

from joblib import Parallel, delayed
import multiprocessing
import argparse

def num_sh_coeffs(lmax):
    """Calculate the number of SH coefficients for even l up to lmax."""
    return sum([2 * l + 1 for l in range(0, lmax + 1, 2)])

def generate_l_m_arrays(lmax):
    """Generate arrays of l and m values corresponding to each SH coefficient index."""
    l_list = []
    m_list = []
    for l in range(0, lmax + 1, 2):  # Even l only
        m_vals = list(range(-l, l + 1))
        l_list.extend([l] * len(m_vals))
        m_list.extend(m_vals)
    l_array = np.array(l_list)  # Shape: (n_coeffs,)
    m_array = np.array(m_list)  # Shape: (n_coeffs,)
    return l_array, m_array

def real_to_complex_sh_coeffs_volume(real_coeffs_volume, m_array):
    """Convert real SH coefficients to complex SH coefficients for the entire 4D volume."""
    # Reshape m_array for broadcasting
    m_array = m_array[np.newaxis, np.newaxis, np.newaxis, :]  # Shape: (1, 1, 1, n_coeffs)

    sqrt2 = np.sqrt(2)

    # Perform conversion using element-wise operations and broadcasting
    complex_coeffs_volume = np.zeros_like(real_coeffs_volume, dtype=complex)

    # Compute masks
    m_neg_mask = (m_array < 0)
    m_zero_mask = (m_array == 0)
    m_pos_mask = (m_array > 0)

    # Apply conversion formulas
    complex_coeffs_volume += (1j * real_coeffs_volume / sqrt2) * m_neg_mask
    complex_coeffs_volume += real_coeffs_volume * m_zero_mask
    complex_coeffs_volume += (real_coeffs_volume / sqrt2) * m_pos_mask

    return complex_coeffs_volume

def complex_to_real_sh_coeffs_volume(complex_coeffs_volume, m_array):
    """Convert complex SH coefficients back to real SH coefficients for the entire 4D volume."""
    # Reshape m_array for broadcasting
    m_array = m_array[np.newaxis, np.newaxis, np.newaxis, :]
    sqrt2 = np.sqrt(2)

    # Initialize real coefficients array
    real_coeffs_volume = np.zeros_like(complex_coeffs_volume.real)

    # Compute masks
    m_neg_mask = (m_array < 0)
    m_zero_mask = (m_array == 0)
    m_pos_mask = (m_array > 0)

    # Apply conversion formulas
    real_coeffs_volume += sqrt2 * complex_coeffs_volume.imag * m_neg_mask
    real_coeffs_volume += complex_coeffs_volume.real * m_zero_mask
    real_coeffs_volume += sqrt2 * complex_coeffs_volume.real * m_pos_mask

    return real_coeffs_volume

def generate_lm_pairs(lmax):
    """Generate a list of (l, m) pairs up to lmax."""
    lm_pairs = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            lm_pairs.append((l, m))
    return lm_pairs

def process_slice(ix, complex_coeffs_memmap_filename, alpha, beta, gamma):
    # Load the memory-mapped array
    complex_coeffs_volume = np.load(complex_coeffs_memmap_filename, mmap_mode='r')

    # Initialize necessary variables inside the function
    lmax = 6
    n_coeffs = num_sh_coeffs(lmax)

    # Generate l and m arrays
    l_array, m_array = generate_l_m_arrays(lmax)

    # Generate all (l, m) pairs up to lmax
    lm_pairs = generate_lm_pairs(lmax)
    n_modes = len(lm_pairs)

    # Generate MRtrix (l, m) pairs (only even l)
    mrtrix_lm_pairs = []
    for l in range(0, lmax + 1, 2):  # Only even l
        for m in range(-l, l + 1):
            mrtrix_lm_pairs.append((l, m))

    # Create index mapping from MRtrix indices to spherical indices
    spherical_indices = [lm_pairs.index((l, m)) for (l, m) in mrtrix_lm_pairs]

    # Create quaternion from Euler angles
    R = quaternionic.array.from_euler_angles(alpha, beta, gamma)

    # Create Wigner object
    wigner = spherical.Wigner(lmax)

    # Process the slice
    rotated_slice = np.zeros_like(complex_coeffs_volume[ix, :, :, :])

    for iy in range(complex_coeffs_volume.shape[1]):
        for iz in range(complex_coeffs_volume.shape[2]):
            # Initialize full complex coefficients array
            full_complex_coeffs = np.zeros(n_modes, dtype=complex)
            full_complex_coeffs[spherical_indices] = complex_coeffs_volume[ix, iy, iz, :]

            modes = spherical.Modes(full_complex_coeffs, lmax=lmax, spin_weight=0)
            rotated_modes = wigner.rotate(modes, R)
            rotated_full_complex_coeffs = rotated_modes.ndarray
            rotated_slice[iy, iz, :] = rotated_full_complex_coeffs[spherical_indices]

    return ix, rotated_slice

def compute_rotation_matrix(alpha, beta, gamma):
        # Rotation matrices around the x, y, and z axes
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ])
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma),  np.cos(gamma), 0],
            [0, 0, 1]
        ])
        # Combined rotation matrix
        R = Rz @ Ry @ Rx  # Note the order of multiplication
        return R

def rotate_channel(channel, volume, inverse_matrix, offset):
    rotated_channel = ndimage.affine_transform(
        volume[..., channel],
        inverse_matrix,
        offset=offset,
        order=3  # Cubic interpolation
    )
    return rotated_channel



##############################################################
##############################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Caller script to rotate SH coefficients using rotate_sh.py.")
    parser.add_argument('-i', '--input', required=True, help='Path to input SH coefficients NIfTI file.')
    parser.add_argument('-o', '--output', required=True, help='Path to save rotated SH coefficients NIfTI file.')
    parser.add_argument('--alpha', type=float, default=0.0, help='Rotation angle around x-axis in degrees.')
    parser.add_argument('--beta', type=float, default=0.0, help='Rotation angle around y-axis in degrees.')
    parser.add_argument('--gamma', type=float, default=0.0, help='Rotation angle around z-axis in degrees.')
    parser.add_argument('--lmax', type=int, default=6, help='Maximum SH order.')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs. -1 uses all available cores.')

    args = parser.parse_args()

    sh_img_path = args.input
    lmax = args.lmax
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    n_jobs = args.n_jobs
    output_file = args.output

    n_coeffs = num_sh_coeffs(lmax)
    sh_img = nib.load(str(sh_img_path))
    real_sh_coeffs_volume = sh_img.get_fdata()
    affine = sh_img.affine
    volume_shape = real_sh_coeffs_volume.shape

    # Random rotation Angles (Euler angles)
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    # Generate l and m arrays
    l_array, m_array = generate_l_m_arrays(lmax)

    # Convert real SH coefficients to complex SH coefficients
    complex_coeffs_volume = real_to_complex_sh_coeffs_volume(real_sh_coeffs_volume, m_array)

    # Save complex_coeffs_volume to a .npy file
    complex_coeffs_memmap_filename = 'complex_coeffs_volume_memmap.npy'
    np.save(complex_coeffs_memmap_filename, complex_coeffs_volume, allow_pickle=False)

    # Prepare indices
    #num_cores = multiprocessing.cpu_count()
    #print(f"Using {num_cores} cores for parallel processing.")

    slice_indices = range(volume_shape[0])

    start_bulk = time.time()
    # Process slices in parallel
    results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(process_slice)(ix, complex_coeffs_memmap_filename, alpha, beta, gamma) for ix in slice_indices
    )

    # Collect results
    rotated_complex_coeffs_volume = np.zeros_like(complex_coeffs_volume)
    for ix, rotated_slice in results:
        rotated_complex_coeffs_volume[ix, :, :, :] = rotated_slice

    # Convert rotated complex SH coefficients back to real SH coefficients
    rotated_real_sh_coeffs_volume = complex_to_real_sh_coeffs_volume(rotated_complex_coeffs_volume, m_array)

    ##### bulk rotation #####
    rotation_matrix = compute_rotation_matrix(alpha, beta, gamma)
    inverse_rotation_matrix = rotation_matrix.T

    center = np.array(rotated_real_sh_coeffs_volume.shape[:3]) / 2.0
    offset = center - np.dot(inverse_rotation_matrix, center)

    bulk_rotated_sh_coeffs_volume = np.zeros_like(complex_coeffs_volume)

    ##### Bulk Affine Transformation for SH Coefficients #####
    print("Starting bulk affine transformation for SH coefficients...")

    # Parallelize the bulk affine transform loop for SH rotation
    rotated_sh_channels = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(rotate_channel)(
            channel,
            rotated_real_sh_coeffs_volume,
            inverse_rotation_matrix,
            offset
        ) for channel in range(rotated_real_sh_coeffs_volume.shape[-1])
    )

    # Stack the rotated channels to form the rotated SH coefficients volume
    bulk_rotated_sh_coeffs_volume = np.stack(rotated_sh_channels, axis=-1)

    # End timing
    end_bulk = time.time()
    print(f"Bulk rotation of SH coefficients completed in {end_bulk - start_bulk:.2f} seconds.")


    # Save the rotated volume data
    bulk_rotated_sh_coeffs_img = nib.Nifti1Image(bulk_rotated_sh_coeffs_volume, affine=affine)
    nib.save(bulk_rotated_sh_coeffs_img, output_file)

    print("done!")
