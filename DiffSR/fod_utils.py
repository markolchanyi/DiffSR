import torch
import numpy as np
import nibabel as nib
from scipy.special import sph_harm
from dipy.data import get_sphere


sphere = get_sphere('repulsion724') 
cartesian_directions = torch.tensor(sphere.vertices, dtype=torch.float32).to('cuda')

# Predefined set
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


def cartesian_to_mrtrix_spherical(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to MRtrix-compatible spherical coordinates (phi, theta).
    :param x: Cartesian x
    :param y: Cartesian y
    :param z: Cartesian z
    :return: phi (azimuth), theta (polar angle)
    """
    theta = torch.acos(z)
    phi = torch.atan2(y, x)
    return phi, theta


def cartesian_to_spherical(x, y, z):

    theta = torch.acos(z)  # Polar
    phi = torch.atan2(y, x)  # Azimuthal 
    return phi, theta


def spherical_to_cartesian(phi, theta):

    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

# Precompute SH basis for all search directions
# Precompute SH basis for all search directions
def precompute_sh_basis(phi, theta, lmax):

    Y = []
    phi_cpu = phi.cpu().numpy()
    theta_cpu = theta.cpu().numpy()

    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            Y.append(torch.real(torch.from_numpy(sph_harm(m, l, phi_cpu, theta_cpu))).float())
    return torch.stack(Y, dim=-1)


def evaluate_fod_with_precomputed(sh_coeffs, Y_precomputed):

    fod_values = torch.matmul(sh_coeffs, Y_precomputed.T)
    return fod_values


def extract_largest_peak_vectorized(sh_volume, lmax, Y_precomputed):

    print("shape: ", sh_volume.shape)
    N, _, _, H = sh_volume.shape

    sh_volume_flat = sh_volume.reshape(-1, H)

    fod_values = torch.matmul(sh_volume_flat, Y_precomputed.T)

    max_indices = torch.argmax(fod_values, dim=1)

    search_directions_device = search_directions.to(max_indices.device)
    peak_directions_flat = spherical_to_cartesian(search_directions_device[max_indices, 0], search_directions_device[max_indices, 1])


    peak_directions = peak_directions_flat.view(N, N, N, 3)

    return peak_directions


def load_nifti_as_tensor(nifti_file):

    nifti_data = nib.load(nifti_file).get_fdata()
    sh_tensor = torch.tensor(nifti_data, dtype=torch.float32).unsqueeze(0).to('cuda')
    return sh_tensor


nifti_file = '/autofs/space/nicc_005/users/olchanyi/DiffSR/test_data/tmp/target.nii'
sh_volume = load_nifti_as_tensor(nifti_file)
sh_volume = torch.squeeze(sh_volume)

lmax = 6
phi, theta = cartesian_to_mrtrix_spherical(cartesian_directions[:, 0], cartesian_directions[:, 1], cartesian_directions[:, 2])
search_directions = torch.stack([phi, theta], dim=1)
Y_precomputed = precompute_sh_basis(search_directions[:, 0], search_directions[:, 1], lmax).to(sh_volume.device)
largest_peaks = extract_largest_peak_vectorized(sh_volume, lmax, Y_precomputed)


def save_tensor_as_nifti(tensor, reference_nifti_file, output_nifti_file):
    """
    Save a PyTorch tensor as a NIfTI file, using a reference NIfTI file for affine and header.
    """
    tensor = tensor.squeeze(0).cpu().numpy()
    nifti_img = nib.Nifti1Image(tensor, affine=nib.load(reference_nifti_file).affine)
    nib.save(nifti_img, output_nifti_file)

output_nifti_file = '/autofs/space/nicc_005/users/olchanyi/DiffSR/test_data/tmp/target_test_fod.nii'
save_tensor_as_nifti(largest_peaks, nifti_file, output_nifti_file)

print("Largest peak directions shape:", largest_peaks.shape)

