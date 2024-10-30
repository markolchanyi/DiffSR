import torch
import numpy as np
import nibabel as nib
from scipy.special import sph_harm
from dipy.data import get_sphere

# Generate a larger set of search directions
sphere = get_sphere('repulsion724')  # 724 directions for better precision
cartesian_directions = torch.tensor(sphere.vertices, dtype=torch.float32).to('cuda')

# Predefined set of search directions (phi, theta) pairs (in radians)
'''
search_directions = torch.tensor([
    [0, 0],
    [-3.14159, 1.3254],
    [-2.58185, 1.50789],
    [2.23616, 1.46585],
    [0.035637, 0.411961],
    [2.65836, 0.913741],
    [0.780743, 1.23955],
    [-0.240253, 1.58088],
    [-0.955334, 1.08447],
    [1.12534, 1.78765],
    [1.12689, 1.30126],
    [0.88512, 1.55615],
    [2.08019, 1.16222],
    [0.191423, 1.06076],
    [1.29453, 0.707568],
    [2.794, 1.24245],
    [2.02138, 0.337172],
    [1.59186, 1.30164],
    [-2.83601, 0.910221],
    [0.569095, 0.96362],
    [3.05336, 1.00206],
    [2.4406, 1.19129],
    [0.437969, 1.30795],
    [0.247623, 0.728643],
    [-0.193887, 1.0467],
    [-1.34638, 1.14233],
    [1.35977, 1.54693],
    [1.82433, 0.660035],
    [-0.766769, 1.3685],
    [-2.02757, 1.02063],
    [-0.78071, 0.667313],
    [-1.47543, 1.45516],
    [-1.10765, 1.38916],
    [-1.65789, 0.871848],
    [1.89902, 1.44647],
    [3.08122, 0.336433],
    [-2.35317, 1.25244],
    [2.54757, 0.586206],
    [-2.14697, 0.338323],
    [3.10764, 0.670594],
    [1.75238, 0.991972],
    [-1.21593, 0.82585],
    [-0.259942, 0.71572],
    [-1.51829, 0.549286],
    [2.22968, 0.851973],
    [0.979108, 0.954864],
    [1.36274, 1.04186],
    [-0.0104792, 1.33716],
    [-0.891568, 0.33526],
    [-2.0635, 0.68273],
    [-2.41353, 0.917031],
    [2.57199, 1.50166],
    [0.965936, 0.33624],
    [0.763244, 0.657346],
    [-2.61583, 0.606725],
    [-0.429332, 1.30226],
    [-2.91118, 1.56901],
    [-2.79822, 1.24559],
    [-1.70453, 1.20406],
    [-0.582782, 0.975235]
], dtype=torch.float32)
'''

# Convert Cartesian to spherical coordinates
def cartesian_to_mrtrix_spherical(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to MRtrix-compatible spherical coordinates (phi, theta).
    :param x: Cartesian x
    :param y: Cartesian y
    :param z: Cartesian z
    :return: phi (azimuth), theta (polar angle)
    """
    theta = torch.acos(z)  # Polar angle (0 to pi)
    phi = torch.atan2(y, x)  # Azimuthal angle (-pi to pi)
    return phi, theta


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (phi, theta).
    :param x: Cartesian x
    :param y: Cartesian y
    :param z: Cartesian z
    :return: phi (azimuth), theta (polar angle)
    """
    theta = torch.acos(z)  # Polar angle
    phi = torch.atan2(y, x)  # Azimuthal angle
    return phi, theta

# Convert spherical to Cartesian coordinates
def spherical_to_cartesian(phi, theta):
    """
    Convert spherical coordinates (phi, theta) to Cartesian coordinates (x, y, z).
    """
    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

# Precompute SH basis for all search directions
# Precompute SH basis for all search directions
def precompute_sh_basis(phi, theta, lmax):
    """
    Precompute the SH basis functions for all search directions.
    This only needs to be done once for all voxels.
    """
    Y = []
    # Move phi and theta to CPU for use with NumPy (SciPy's sph_harm)
    phi_cpu = phi.cpu().numpy()
    theta_cpu = theta.cpu().numpy()

    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            # Use NumPy (sph_harm) to compute SH basis and then convert to PyTorch
            Y.append(torch.real(torch.from_numpy(sph_harm(m, l, phi_cpu, theta_cpu))).float())
    # Stack and return as a PyTorch tensor
    return torch.stack(Y, dim=-1)

# Use precomputed SH basis in evaluation
def evaluate_fod_with_precomputed(sh_coeffs, Y_precomputed):
    """
    Use precomputed SH basis functions to evaluate FOD for all directions.
    """
    # FOD value is the dot product of SH coefficients and the SH basis
    fod_values = torch.matmul(sh_coeffs, Y_precomputed.T)
    return fod_values

# Vectorized extraction of largest peaks for all voxels
def extract_largest_peak_vectorized(sh_volume, lmax, Y_precomputed):
    """
    Vectorized extraction of largest peaks for all voxels in the volume.
    :param sh_volume: SH coefficient volume (torch.Tensor, shape N x N x N x H)
    :param lmax: Maximum SH order (int)
    :param Y_precomputed: Precomputed SH basis for search directions
    :return: Largest peak directions (torch.Tensor, shape N x N x N x 3)
    """
    print("shape: ", sh_volume.shape)
    N, _, _, H = sh_volume.shape

    # Reshape SH volume to (N^3 x H) to treat all voxels as a batch
    sh_volume_flat = sh_volume.reshape(-1, H)  # Shape: (N^3, H)

    # Evaluate FOD for all voxels in parallel
    fod_values = torch.matmul(sh_volume_flat, Y_precomputed.T)  # Shape: (N^3, D)

    # Find the direction with the maximum FOD for each voxel
    max_indices = torch.argmax(fod_values, dim=1)  # Shape: (N^3,)

    # Move search_directions to the same device as max_indices
    search_directions_device = search_directions.to(max_indices.device)

    # Convert max indices to Cartesian coordinates using precomputed directions
    peak_directions_flat = spherical_to_cartesian(search_directions_device[max_indices, 0], search_directions_device[max_indices, 1])

    # Reshape back to (N x N x N x 3)
    peak_directions = peak_directions_flat.view(N, N, N, 3)

    # Convert max indices to Cartesian coordinates using precomputed directions
    #peak_directions_flat = cartesian_directions[max_indices]  # Already in Cartesian

    # Reshape back to (N x N x N x 3)
    #peak_directions = peak_directions_flat.reshape(N, N, N, 3)

    return peak_directions

# Example usage
# Example usage

def load_nifti_as_tensor(nifti_file):
    """
    Load a NIfTI file and convert it to a PyTorch tensor.
    Assumes the NIfTI file contains SH coefficients in the last dimension.
    """
    nifti_data = nib.load(nifti_file).get_fdata()  # Load data as a NumPy array
    sh_tensor = torch.tensor(nifti_data, dtype=torch.float32).unsqueeze(0).to('cuda')  # Convert to PyTorch tensor on GPU
    return sh_tensor


nifti_file = '/autofs/space/nicc_005/users/olchanyi/DiffSR/test_data/tmp/target.nii'  # Path to your NIfTI file
sh_volume = load_nifti_as_tensor(nifti_file)  # Load SH volume as PyTorch tensor
sh_volume = torch.squeeze(sh_volume)

lmax = 6  # Maximum SH order
#Y_precomputed = precompute_sh_basis(search_directions[:, 0], search_directions[:, 1], lmax).to(sh_volume.device)  # Precompute SH basis
phi, theta = cartesian_to_mrtrix_spherical(cartesian_directions[:, 0], cartesian_directions[:, 1], cartesian_directions[:, 2])
search_directions = torch.stack([phi, theta], dim=1)
#Y_precomputed = precompute_sh_basis(phi, theta, lmax).to(sh_volume.device)
Y_precomputed = precompute_sh_basis(search_directions[:, 0], search_directions[:, 1], lmax).to(sh_volume.device)  # Precompute SH basis
# Extract the largest peak from the FOD volume
largest_peaks = extract_largest_peak_vectorized(sh_volume, lmax, Y_precomputed)

# Optionally save the result as a NIfTI file
def save_tensor_as_nifti(tensor, reference_nifti_file, output_nifti_file):
    """
    Save a PyTorch tensor as a NIfTI file, using a reference NIfTI file for affine and header.
    """
    tensor = tensor.squeeze(0).cpu().numpy()  # Convert back to NumPy array
    nifti_img = nib.Nifti1Image(tensor, affine=nib.load(reference_nifti_file).affine)
    nib.save(nifti_img, output_nifti_file)

output_nifti_file = '/autofs/space/nicc_005/users/olchanyi/DiffSR/test_data/tmp/target_test_fod.nii'
save_tensor_as_nifti(largest_peaks, nifti_file, output_nifti_file)

print("Largest peak directions shape:", largest_peaks.shape)

