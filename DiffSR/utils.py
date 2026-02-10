# This file contains a bunch of functions (and now more!!!) from Benjamin's lab2im package
# (it's just much lighter to import...)
import nibabel as nib
import numpy as np
import os
import torch
import math
from torch.nn import L1Loss, MSELoss
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage import gaussian_filter as gauss_filt
from scipy.special import lpmv

# Load nifti or mgz file
def load_volume(path_volume):

    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz')), 'Unknown data file: %s' % path_volume

    x = nib.load(path_volume)
    volume = x.get_fdata()
    aff = x.affine

    return volume, aff


# Save volume
def save_volume(volume, aff, path):

    header = nib.Nifti1Header()
    if aff is None:
        aff = np.eye(4)
    nifti = nib.Nifti1Image(volume, aff, header)
    nib.save(nifti, path)


def myzoom_torch(X, factor, device='cpu'):

    if len(X.shape) == 3:
        X = X[..., None]

    delta = (1.0-factor)/(2.0*factor)
    newsize = np.round(X.shape[:-1]*factor).astype(int)

    vx = torch.arange(delta[0], delta[0] + newsize[0] / factor[0], 1 / factor[0], device=device)
    vy = torch.arange(delta[1], delta[1] + newsize[1] / factor[1], 1 / factor[1], device=device)
    vz = torch.arange(delta[2], delta[2] + newsize[2] / factor[2], 1 / factor[2], device=device)
    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0] - 1)] = (X.shape[0] - 1)
    vy[vy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    vz[vz > (X.shape[2] - 1)] = (X.shape[2] - 1)
    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    wcy = vy - fy
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros([newsize[0], newsize[1], newsize[2], X.shape[3]], device=device)

    for channel in range(X.shape[3]):
        Xc = X[..., channel]
        tmp1 = torch.zeros([newsize[0], Xc.shape[1], Xc.shape[2]], device=device)
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] + wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros([newsize[0], newsize[1], Xc.shape[2]], device=device)
        for j in range(newsize[1]):
            tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] + wcy[j] * tmp1[:, cy[j], :]
        for k in range(newsize[2]):
            Y[:, :, k, channel] = wfz[k] * tmp2[:, :, fz[k]] + wcz[k] * tmp2[:, :, cz[k]]

    if Y.shape[3] == 1:
        Y = Y[:, :, :, 0]

    return Y


def myzoom_torch_better(X, factor, device='cpu'):

    if not torch.is_tensor(X):
        X = torch.as_tensor(X, device=device)
    else:
        X = X.to(device)

    if X.ndim == 3:
        X = X[None, None, ...]
    elif X.ndim == 4:
        if X.shape[-1] < min(X.shape[:-1]):
            X = X.permute(3, 0, 1, 2)
        X = X.unsqueeze(0)
    elif X.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected X.ndim={X.ndim}")

    if isinstance(factor, (int, float)):
        scale_factor = (factor, factor, factor)
    else:
        scale_factor = tuple(float(f) for f in factor)

    Y = F.interpolate(X, scale_factor=scale_factor, mode="trilinear", align_corners=False)

    if Y.shape[0] == 1:
        Y = Y.squeeze(0)
    Y = Y.permute(1, 2, 3, 0)
    return Y


def fast_3D_interp_torch(X, II, JJ, KK, mode, device='cpu'):
    # Get dimensions
    D1 = X.shape[0]
    D2 = X.shape[1]
    D3 = X.shape[2]

    # Bound the indices to be within [0, D-1]
    II = II.clamp(0, D1 - 1)
    JJ = JJ.clamp(0, D2 - 1)
    KK = KK.clamp(0, D3 - 1)

    # Flatten the input volume along spatial dimensions
    X_flat = X.view(-1)  # Flattened tensor of size (D1*D2*D3,)

    if mode == 'nearest':
        # Round indices to nearest integer indices
        IIf = II.round().long()
        JJf = JJ.round().long()
        KKf = KK.round().long()

        # Compute linear indices
        linear_indices = IIf * (D2 * D3) + JJf * D3 + KKf

        # Gather values from the flattened tensor
        interpolated_values = X_flat[linear_indices]

    elif mode == 'linear':
        # Floor and ceil for trilinear interpolation
        I0 = II.floor().long()
        I1 = (I0 + 1).clamp(0, D1 - 1)
        J0 = JJ.floor().long()
        J1 = (J0 + 1).clamp(0, D2 - 1)
        K0 = KK.floor().long()
        K1 = (K0 + 1).clamp(0, D3 - 1)

        # Fractional part used for interpolation weights
        dI = II - I0.float()
        dJ = JJ - J0.float()
        dK = KK - K0.float()

        # Compute weights
        w000 = (1 - dI) * (1 - dJ) * (1 - dK)
        w001 = (1 - dI) * (1 - dJ) * dK
        w010 = (1 - dI) * dJ * (1 - dK)
        w011 = (1 - dI) * dJ * dK
        w100 = dI * (1 - dJ) * (1 - dK)
        w101 = dI * (1 - dJ) * dK
        w110 = dI * dJ * (1 - dK)
        w111 = dI * dJ * dK

        # Compute linear indices for the vertices of the cube
        idx000 = I0 * (D2 * D3) + J0 * D3 + K0
        idx001 = I0 * (D2 * D3) + J0 * D3 + K1
        idx010 = I0 * (D2 * D3) + J1 * D3 + K0
        idx011 = I0 * (D2 * D3) + J1 * D3 + K1
        idx100 = I1 * (D2 * D3) + J0 * D3 + K0
        idx101 = I1 * (D2 * D3) + J0 * D3 + K1
        idx110 = I1 * (D2 * D3) + J1 * D3 + K0
        idx111 = I1 * (D2 * D3) + J1 * D3 + K1

        # Gather the values and compute interpolated result
        interpolated_values = (
            w000 * X_flat[idx000] + w001 * X_flat[idx001] +
            w010 * X_flat[idx010] + w011 * X_flat[idx011] +
            w100 * X_flat[idx100] + w101 * X_flat[idx101] +
            w110 * X_flat[idx110] + w111 * X_flat[idx111]
        )

    else:
        raise ValueError("Invalid mode. Use 'nearest' or 'linear'.")

    return interpolated_values


def myzoom_np(X, factor, device='cpu'):

    delta = (1.0-factor)/(2.0*factor)
    newsize = np.round(X.shape*factor).astype(int)

    vx = np.arange(delta[0], delta[0]+newsize[0]/factor[0], 1/factor[0])
    vy = np.arange(delta[1], delta[1]+newsize[1]/factor[1], 1/factor[1])
    vz = np.arange(delta[2], delta[2]+newsize[2]/factor[2], 1/factor[2])

    vx[vx<0] = 0
    vy[vy<0] = 0
    vz[vz<0] = 0
    vx[vx>(X.shape[0]-1)] = (X.shape[0]-1)
    vy[vy>(X.shape[1]-1)] = (X.shape[1]-1)
    vz[vz>(X.shape[2]-1)] = (X.shape[2]-1)

    fx = np.floor(vx).astype(int)
    cx = fx + 1
    cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
    wcx = vx - fx
    wfx = 1 - wcx

    fy = np.floor(vy).astype(int)
    cy = fy + 1
    cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    wcy = vy - fy
    wfy = 1 - wcy

    fz = np.floor(vz).astype(int)
    cz = fz + 1
    cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
    wcz = vz - fz
    wfz = 1 - wcz

    Y = np.zeros((newsize[0], newsize[1], newsize[2]))

    for i in range(newsize[0]):
        for j in range(newsize[1]):
            for k in range(newsize[2]):
                Y[i, j, k] = (
                    (wfx[i] * wfy[j] * wfz[k] * X[fx[i], fy[j], fz[k]]) +
                    (wfx[i] * wfy[j] * wcz[k] * X[fx[i], fy[j], cz[k]]) +
                    (wfx[i] * wcy[j] * wfz[k] * X[fx[i], cy[j], fz[k]]) +
                    (wfx[i] * wcy[j] * wcz[k] * X[fx[i], cy[j], cz[k]]) +
                    (wcx[i] * wfy[j] * wfz[k] * X[cx[i], fy[j], fz[k]]) +
                    (wcx[i] * wfy[j] * wcz[k] * X[cx[i], fy[j], cz[k]]) +
                    (wcx[i] * wcy[j] * wfz[k] * X[cx[i], cy[j], fz[k]]) +
                    (wcx[i] * wcy[j] * wcz[k] * X[cx[i], cy[j], cz[k]])
                )

    return Y


def myzoom_vol(volume, aff, factor, thresh=0., interp_order=1):
    """Zoom volume according to given factor and fill affine matrix accordingly.
    volume = 4D or 3D np.array
    factor = float or iterable of size 3
    affine matrix associated to input volume"""

    # reformat inputs
    if not isinstance(factor, (list, tuple, np.ndarray)):
        factor = np.array([factor] * 3)

    volume_shape, n_dims, n_channels, _, _ = get_volume_info(volume)
    if volume.shape[-1] == 1:
        volume = np.squeeze(volume)
    else:
        raise Exception('myzoom was only tested for single-channel images')

    # zoom
    nb_dims, n_channels = get_dims(volume_shape)
    if interp_order == 0:  # nearest interpolation
        new_shape = [math.ceil(volume_shape[i] * factor[i]) for i in range(nb_dims)]
        zoom = rgi(points=[np.arange(i) for i in volume_shape],
                   values=volume,
                   bounds_error=False,
                   fill_value=0)
        x = np.arange(0, volume_shape[0], 1 / factor[0])
        y = np.arange(0, volume_shape[1], 1 / factor[1])
        z = np.arange(0, volume_shape[2], 1 / factor[2])
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)
        volume_filt = zoom((xx, yy, zz))
        thresh = np.max(volume_filt) * thresh
        volume_filt[volume_filt < thresh] = 0
    elif interp_order == 1:  # linear interpolation
        new_shape = [math.ceil(volume_shape[i] / factor[i]) for i in range(nb_dims)]
        zoom = rgi(points=[np.arange(i) for i in volume_shape],
                   values=volume,
                   bounds_error=False,
                   fill_value=0)
        start = [0.5 * (factor[i] - 1) for i in range(3)]
        stop = [volume_shape[i] - 0.5 * (factor[i] - 1) for i in range(3)]
        step = [factor[i] for i in range(3)]
        xi = np.arange(start=start[0], stop=stop[0], step=step[0])
        yi = np.arange(start=start[1], stop=stop[1], step=step[1])
        zi = np.arange(start=start[2], stop=stop[2], step=step[2])
        xi[xi < 0] = 0
        yi[yi < 0] = 0
        zi[zi < 0] = 0
        xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
        yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
        zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

        xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
        volume2 = my_interpolating_function((xig, yig, zig))

        aff2 = aff.copy()
        for c in range(3):
            aff2[:-1, c] = aff2[:-1, c] / factor[c]
        aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))
        return volume2, aff2
    else:
        raise Exception('interp_order must be 0 or 1')

    volume2 = myzoom_np(volume, 1 / factor)
    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))
    return volume2, aff2


#-----------------------------------------------------#
def get_volume_info(volume, aff=None, return_volume=False):
    """Get info on the provided volume, and return re-formatted volume and aff if necessary.
    Namely the returned volume and afine matrix are either 4D and 4*4.
    :param volume: a numpy array
    :param aff: if volume was loaded with lab2im, this is the affine matrix at loading.
    :param return_volume: whether to return reformated volume and affine matrix.
    :return: volume shape, number of dimensions, etc"""

    # get dimension of the volume
    volume = np.squeeze(volume)
    n_dims = len(volume.shape)
    assert n_dims in [3, 4], 'volume should be 3 or 4-dimensional, had %s dimensions' % n_dims

    # reformat volume to have a last dimension for channels
    if n_dims == 3:
        volume = volume[..., np.newaxis]
    volume_shape = list(volume.shape[:3])
    n_channels = volume.shape[-1]

    if return_volume:
        # reformat affine matrix
        if aff is None:
            aff = np.eye(4)
        elif aff.shape == (3, 3):
            aff2 = np.eye(4)
            aff2[:3, :3] = aff
            aff = aff2
        elif not aff.shape == (4, 4):
            raise Exception('affine matrix should be 3x3 or 4x4, had: %s' % aff.shape)
        return volume, aff, volume_shape, n_dims, n_channels
    else:
        return volume_shape, n_dims, n_channels, volume, aff


# ------------------------------------------------- transformation utils -------------------------------------------------


def get_dims(shape, max_channels=10):
    """Get number of spatial dimensions and number of channels.
    see get_volume_info for more details.
    The argument max_channels permits to differentiate between channels and additional dimensions (if len(shape)>4).
    e.g.:
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).

    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this funtion returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformated list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        var = list(var)
    elif not isinstance(var, list):
        raise TypeError('var should be an int, float, list, tuple, or numpy array')

    # check if list has right length
    if length is not None:
        if len(var) == 1:
            var = var * length
        elif len(var) != length:
            raise Exception('var should have %s elements, but list has %s elements' % (length, len(var)))

    # convert type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise Exception('dtype should be "int", "float", "bool", or "str".')

    return var


def load_array_if_path(var, load_as_numpy=False):

    if isinstance(var, str) and load_as_numpy:
        if var.endswith('.npy'):
            assert os.path.isfile(var), 'no such path: %s' % var
            var = np.load(var)
        else:
            raise ValueError('file not supported by load_array_if_path: %s' % var)
    return var


def my_interp3(volume, xx, yy, zz, inter_method='linear'):
    # Define the interpolation method
    if inter_method not in ['linear', 'nearest']:
        raise ValueError("inter_method should be either 'linear' or 'nearest'")
    interpolating_function = rgi(
        (np.arange(volume.shape[0]), np.arange(volume.shape[1]), np.arange(volume.shape[2])),
        volume,
        method=inter_method,
        bounds_error=False,
        fill_value=0
    )

    # Perform interpolation
    interpolated_values = interpolating_function((xx, yy, zz))
    return interpolated_values


def random_transform3_label_map(volume1, sc_labels=None, flipping=False, scaling_bounds=0.2, rotation_bounds=180,
                                shearing_bounds=0.012, translation_bounds=False, nonlin_std=3., nonlin_shape_factor=.0625,
                                data_res=None, n_dims=None, voxel_size=None, aff=None, nonlin_cpu=False):
    """Randomly transform volume, and return the transformed volume and corresponding affine matrix (as a linearly
    transformed version of the input affine matrix).
    The transformation is as follows:
    - rescale to the power of the inputs of the network (used to handle super-resolution)
    - rigid (flipping, scaling, rotation, shearing, translation)
    - non-linear deformation
    :param volume1: input volume to be deformed
    :param sc_labels: whether to transform volume as label map (nearest neighbour interpolation), or image (trilinear)
    :param flipping: (optional) can be True, or a list of booleans of length n_dims. If True, default behaviour is to
    flip the first axis.
    :param scaling_bounds: (optional) can be a number, a sequence, or a sequence of sequences. If it is a number,
    the scaling factor is drawn uniformly in [1-scaling_bounds; 1+scaling_bounds]. If it is a sequence, the scaling
    factor is drawn uniformly between the two values. If it is a sequence of sequences, the scaling factor for each
    dimension is sampled independently.
    :param rotation_bounds: (optional) can be a number, a sequence or a sequence of sequences. If it is a number,
    the rotation angle (in degrees) is drawn uniformly in [-rotation_bounds; rotation_bounds]. If it is a sequence,
    the rotation angle is drawn uniformly between the two values. If it is a sequence of sequences, the rotation
    angle for each dimension is sampled independently.
    :param shearing_bounds: (optional) if not None, should be a sequence, or a sequence of sequences. If it's a
    sequence, the shearing factor is drawn uniformly between the two values. If it is a sequence of sequences,
    the shearing factor for each dimension is sampled independently.
    :param translation_bounds: (optional) can be 'True' or a sequence. If 'True', the translation is drawn uniformly
    in [-10; 10]. If it is a sequence, the translation is drawn uniformly between the two values.
    :param nonlin_std: (optional) standard deviation of the Gaussian distributions from which the deformations are
    sampled. Set to None to completely turn the non-linear deformation off.
    :param nonlin_shape_factor: (optional) if nonlin_std is not None, the smoothing/kernel size will be
    nonlin_std/nonlin_shape_factor.
    :param data_res: (optional) resolution at which the network is trained. Used to rescale the inputs of the network
    to voxels, at which we sample the transformation.
    :param n_dims: (optional) number of dimensions of the input volume.
    :param voxel_size: (optional) voxel size of the input volume, in case aff is None.
    :param aff: (optional) affine matrix of the input volume.
    :param nonlin_cpu: (optional) whether to perform the non-linear transformation on the CPU.
    :return: transformed volume and corresponding linearly transformed affine matrix
    """

    # reformat input data
    [scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds] =\
        reformat_to_list([scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds])
    volume1, aff, volume_shape1, n_dims, _ = get_volume_info(volume1, aff, return_volume=True)

    # get info
    shape1 = np.array(volume_shape1)
    if n_dims is None:
        n_dims = len(volume_shape1)
    else:
        assert n_dims == len(volume_shape1), 'n_dims should be equal to len(volume_shape), had {} and {}'.format(
            n_dims, len(volume_shape1))
    if voxel_size is None:
        if aff is None:
            voxel_size = 1
        else:
            voxel_size = np.sqrt(np.sum(aff[:-1, :-1] ** 2, 0))
    if data_res is not None:
        data_res = reformat_to_list(data_res, length=n_dims)
        scaling_factor = [voxel_size[i] / data_res[i] for i in range(n_dims)]
    else:
        scaling_factor = [1.0] * n_dims

    # get transformation
    transfo_params = sample_affine_transform(sc_labels=sc_labels, scaling_bounds=scaling_bounds,
                                             rotation_bounds=rotation_bounds, shearing_bounds=shearing_bounds,
                                             translation_bounds=translation_bounds)

    # where to sample GDF
    if data_res is not None:
        x = [np.arange(shape1[i]) * scaling_factor[i] for i in range(n_dims)]
    else:
        x = [np.arange(shape1[i]) for i in range(n_dims)]

    # apply affine transform to get corresponding points in input data
    y = np.meshgrid(*x, indexing='ij')
    y = [y[d].astype(float) for d in range(n_dims)]
    coords = np.concatenate([y[d].reshape([-1, 1]) for d in range(n_dims)], axis=1)
    drot = np.eye(n_dims + 1)
    drot[:n_dims, :n_dims] = transfo_params[:n_dims, :n_dims]
    drot[:n_dims, -1] = transfo_params[:n_dims, -1]
    R = drot[:n_dims, :n_dims]
    offset = drot[:n_dims, -1]

    # apply transformation
    coords = np.dot(coords, R.T) + offset
    coords = [coords[:, i].reshape(list(shape1)) for i in range(n_dims)]

    # non-linear deformation
    if (nonlin_std is not None) & (nonlin_std > 0):
        if nonlin_cpu:
            warp_ = [gauss_filt(np.random.normal(size=volume_shape1, scale=nonlin_std),
                                sigma=nonlin_std/nonlin_shape_factor) for _ in range(n_dims)]
            coords = [coords[i] + warp_[i] for i in range(n_dims)]
        else:
            warp_ = [gauss_filt(np.random.normal(size=volume_shape1, scale=nonlin_std),
                                sigma=nonlin_std / nonlin_shape_factor) for _ in range(n_dims)]
            coords = [coords[i] + warp_[i] for i in range(n_dims)]

    # interpolate data
    # we have a bug in case of nearest and label_maps, scale to range(0,1) first
    if sc_labels:
        min1 = volume1.min()
        max1 = volume1.max()
        volume1 = (volume1 - min1) / (max1 - min1)
        march_method = 'nearest'
    else:
        march_method = 'linear'

    # do interpolation
    volume1 = my_interp3(volume1, coords[0], coords[1], coords[2], inter_method=march_method)
    if sc_labels:
        volume1 = np.round(volume1 * (max1 - min1) + min1)

    # transform affine matrix
    if voxel_size is not None:
        if np.any(np.array(voxel_size) != 1):
            aff2 = aff.copy()
            aff2[:-1, :-1] = np.dot(transfo_params, aff2[:-1, :-1])
            aff2[:-1, -1] = np.dot(transfo_params, aff2[:-1, -1])
        else:
            aff2 = aff
    else:
        aff2 = None

    return volume1, aff2


def sample_affine_transform(sc_labels=None, scaling_bounds=0.2, rotation_bounds=180, shearing_bounds=0.012,
                            translation_bounds=False):
    """Sample an affine transformation (rotation, scaling, shearing, translation).
    Namely the transformation is obtained by combining rotation, scaling, shearing and translation.
    """

    scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds = \
        reformat_to_list([scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds])

    rotation_bounds = np.array(rotation_bounds) * math.pi / 180

    # Create random rotation matrix (R)
    rx, ry, rz = np.random.uniform(-rotation_bounds, rotation_bounds)
    cosx, sinx = np.cos(rx), np.sin(rx)
    cosy, siny = np.cos(ry), np.sin(ry)
    cosz, sinz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Create random scaling matrix (S)
    scales = np.random.uniform(1 - scaling_bounds, 1 + scaling_bounds, size=3)
    S = np.diag(scales)

    # Create random shearing matrix (Sh)
    shx, shy, shz = np.random.uniform(-shearing_bounds, shearing_bounds, size=3)
    Sh = np.array([[1, shx, shy],
                   [0, 1, shz],
                   [0, 0, 1]])

    # Combine transformations: first scaling, then rotation, then shearing
    affine_matrix = np.dot(Sh, np.dot(R, S))

    # Add translation (T)
    if translation_bounds is not False:
        translations = np.random.uniform(-translation_bounds, translation_bounds, size=3)
    else:
        translations = np.zeros(3)

    affine_transform = np.eye(4)
    affine_transform[:3, :3] = affine_matrix
    affine_transform[:3, 3] = translations

    return affine_transform


def make_rotation_matrix(angles):
    """
    Generate a 3D rotation matrix from Euler angles.
    angles: (alpha, beta, gamma) in degrees
    """
    alpha, beta, gamma = np.deg2rad(angles)

    # Rotation around the x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

    # Rotation around the y-axis
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    # Rotation around the z-axis
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx  # Note: order of multiplication matters

    return R


def my_interpolating_function(points, values=None, method='linear'):
    if values is None:
        raise ValueError("Values must be provided for interpolation")
    return rgi(points, values, method=method, bounds_error=False, fill_value=0)


def rotation_matrix(axis, angle):
    """
    Create a rotation matrix for rotating around a given axis by a given angle
    axis: 3-element array-like
    angle: in radians
    """
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)  # Normalize the axis vector
    x, y, z = axis

    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    R = np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ])

    return R


def random_3d_affine_transform_matrix(rotation_range=10, scaling_range=0.1, translation_range=10):
    """
    Generate a random 3D affine transformation matrix that can be applied to a volume.

    Parameters:
    - rotation_range: max rotation angle in degrees around each axis
    - scaling_range: max scaling factor deviation around 1 (e.g., 0.1 means [0.9, 1.1])
    - translation_range: max translation in voxels along each dimension

    Returns:
    - affine_matrix: 4x4 numpy array representing the affine transformation
    """
    # Random rotations
    angles = np.radians(np.random.uniform(-rotation_range, rotation_range, size=3))
    Rx = rotation_matrix([1, 0, 0], angles[0])
    Ry = rotation_matrix([0, 1, 0], angles[1])
    Rz = rotation_matrix([0, 0, 1], angles[2])

    R = Rz @ Ry @ Rx  # Combined rotation

    # Random scaling
    scale_factors = np.random.uniform(1 - scaling_range, 1 + scaling_range, size=3)
    S = np.diag(scale_factors)

    # Random translation
    translation = np.random.uniform(-translation_range, translation_range, size=3)

    # Construct affine matrix
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = R @ S  # Apply rotation and scaling
    affine_matrix[:3, 3] = translation  # Apply translation

    return affine_matrix


def sh_norm(sh_tensor,
            lowb_index: int = 0,
            l0_index: int = 1,
            thr: float = 1e-3,
            eps: float = 1e-6):
    """
    Normalize low-b, l0, and l2 SH channels to have comparable range.

    - low-b scaled to ~[0, 1] using 1–99 percentiles
    - l0 scaled to ~[0, 1] using 1–99 percentiles
    - l>=2(+) (if using but not in this config  scaled so that the 99 percentile of SH is  approx 1
    # still clamped
    """


    if isinstance(sh_tensor, torch.Tensor):
        is_torch = True
        device = sh_tensor.device
        dtype  = sh_tensor.dtype
    elif isinstance(sh_tensor, np.ndarray):
        is_torch = False
        device = None
        dtype  = None
    else:
        raise TypeError("sh_tensor must be a torch.Tensor or numpy.ndarray")

    x = sh_tensor


    lowb = x[..., lowb_index]
    l0   = x[..., l0_index]

    high = x[..., (l0_index + 1):]   # (..., n_high)

    def robust_p1_p99(arr_t):
        # arr_t: torch tensor.
        arr_np = arr_t.detach().cpu().numpy()
        p1  = np.percentile(arr_np,  1.0)
        p99 = np.percentile(arr_np, 99.0)
        return p1, p99

    # LOW-B
    if is_torch:
        mask_lowb = (lowb > thr)
        if mask_lowb.any():
            lowb_filtered = lowb[mask_lowb]
            p1_lowb, p99_lowb = robust_p1_p99(lowb_filtered)
            p1_lowb_t  = torch.tensor(p1_lowb, device=device, dtype=dtype)
            p99_lowb_t = torch.tensor(p99_lowb, device=device, dtype=dtype)

            # Clip to percentile range and scale
            lowb_clipped = torch.clamp(lowb, min=p1_lowb_t, max=p99_lowb_t)
            lowb_norm = (lowb_clipped - p1_lowb_t) / (p99_lowb_t - p1_lowb_t + eps)
        else:
            #no valid voxels, just set to zero
            lowb_norm = torch.zeros_like(lowb)
    else:
        mask_lowb = (lowb > thr)
        if mask_lowb.any():
            lowb_filtered = lowb[mask_lowb]
            p1_lowb, p99_lowb = np.percentile(lowb_filtered, [1.0, 99.0])
            lowb_clipped = np.clip(lowb, p1_lowb, p99_lowb)
            lowb_norm = (lowb_clipped - p1_lowb) / (p99_lowb - p1_lowb + eps)
        else:
            lowb_norm = np.zeros_like(lowb)

    # l0
    if is_torch:
        mask_l0 = (l0 > thr)
        if mask_l0.any():
            l0_filtered = l0[mask_l0]
            p1_l0, p99_l0 = robust_p1_p99(l0_filtered)
            p1_l0_t  = torch.tensor(p1_l0, device=device, dtype=dtype)
            p99_l0_t = torch.tensor(p99_l0, device=device, dtype=dtype)

            l0_clipped = torch.clamp(l0, min=p1_l0_t, max=p99_l0_t)
            l0_norm = (l0_clipped - p1_l0_t) / (p99_l0_t - p1_l0_t + eps)
        else:
            l0_norm = torch.zeros_like(l0)
    else:
        mask_l0 = (l0 > thr)
        if mask_l0.any():
            l0_filtered = l0[mask_l0]
            p1_l0, p99_l0 = np.percentile(l0_filtered, [1.0, 99.0])
            l0_clipped = np.clip(l0, p1_l0, p99_l0)
            l0_norm = (l0_clipped - p1_l0) / (p99_l0 - p1_l0 + eps)
        else:
            l0_norm = np.zeros_like(l0)

    # l2
    if high.shape[-1] > 0:
        if is_torch:
            mask_high = (high.abs() > thr)
            if mask_high.any():
                high_filtered = high[mask_high]
                # use abs values for percentile
                high_abs_np = high_filtered.abs().detach().cpu().numpy()
                p99_high = np.percentile(high_abs_np, 99.0)
                scale_high_t = torch.tensor(max(p99_high, eps), device=device, dtype=dtype)
                high_norm = high / scale_high_t
                high_norm = torch.clamp(high_norm, min=-1.0, max=1.0)
            else:
                high_norm = torch.zeros_like(high)
        else:
            mask_high = (np.abs(high) > thr)
            if mask_high.any():
                high_filtered = high[mask_high]
                p99_high = np.percentile(np.abs(high_filtered), 99.0)
                scale_high = max(p99_high, eps)
                high_norm = high / scale_high
                #high_norm = np.clip(high_norm, -1.0, 1.0)
            else:
                high_norm = np.zeros_like(high)
    else:
        high_norm = high  # no high-order coeffs


    if is_torch:
        out = sh_tensor.clone()
    else:
        out = np.array(sh_tensor, copy=True)

    out[..., lowb_index] = lowb_norm
    out[..., l0_index]   = l0_norm
    if high.shape[-1] > 0:
        out[..., (l0_index + 1):] = high_norm

    return out


def percentile_scaling(sh_tensor, l0_index=1, threshold=0.01, new_min=-1.0, new_max=1.0):

    # Extract l=0 channel
    l0 = sh_tensor[..., l0_index]
    l2 = sh_tensor[..., 1:6] # normalize to l2 coeffs as to avoid l0 dominance

    mask_l0 = l0 > threshold
    l0_filtered = l0[mask_l0]

    # Compute lower and upper percentile values
    lower_percentile_l0=1.0
    upper_percentile_l0=99.0
    lower_percentile_l2=1.0
    upper_percentile_l2=99.0

    l0_low = np.percentile(l0_filtered, lower_percentile_l0)
    l0_high = np.percentile(l0_filtered, upper_percentile_l0)


    l0_norm = (l0 - l0_low) / (l0_high - l0_low)
    l0_norm = np.clip(l0_norm, 0, 1)
    l0_scaled = new_min + l0_norm * (new_max - new_min)
    sh_tensor_scaled = np.copy(sh_tensor)
    sh_tensor_scaled[..., l0_index] = l0_scaled

    return sh_tensor_scaled


def rand_lowrank_mix(S, rank=2, scale=0.02):
    """
    Apply a random low-rank mixing of SH channels to model direction-dependent bias during training
    Low param count and differentiable in PyTorch.
    """
    *spatial_dims, C = S.shape
    device = S.device
    S_flat = S.reshape(-1, C)

    A = torch.randn(C, rank, device=device) * scale
    B = torch.randn(rank, C, device=device) * scale

    M = A @ B
    S_mixed = S_flat @ (torch.eye(C, device=device) + M) #simple
    return S_mixed.view(*spatial_dims, C)


def make_gaussian_kernel(sigma):
    sl = np.ceil(sigma*2.5).astype(int)
    v = np.arange(-sl, sl+1)
    gauss = np.exp((-(v/sigma)**2/2))
    kernel = gauss/np.sum(gauss)
    return kernel


def compute_patch_indices(random_patch_center, patch_size, orig_shape, orig_center):
    # compute distance from center
    patch_offsets = tuple(random_patch_center[i] - orig_center[i] for i in range(3))

    # apply offsets
    patch_indices = []
    for i in range(3):
        patch_start = int(patch_offsets[i] - patch_size[i] // 2 + orig_center[i])
        patch_end = patch_start + patch_size[i]

        # Ensure we stay within the bounds of the original volume
        patch_start = max(0, patch_start)
        patch_end = min(orig_shape[i], patch_end)

        patch_indices.append(slice(patch_start, patch_end))

    return tuple(patch_indices)


def random_crop(hr, crop_size):

    spatial_dims = hr.shape[:-1]
    while True:
        start = [torch.randint(0, spatial_dims[i] - crop_size[i], (1,)).item() for i in range(3)]
        end = [start[i] + crop_size[i] for i in range(3)]

        crop = hr[start[0]:end[0], start[1]:end[1], start[2]:end[2], :]
        non_zero_fraction = (crop[..., 0] != 0).float().mean().item()

        if non_zero_fraction > 0.7:
            return crop


def gradient_loss(pred, target):
    """
    grad loss for training (clearly explicit)
    """

    grad_x_pred = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    grad_y_pred = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    grad_z_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]

    grad_x_target = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    grad_y_target = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    grad_z_target = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

    grad_diff_x = (grad_x_pred - grad_x_target) ** 2
    grad_diff_y = (grad_y_pred - grad_y_target) ** 2
    grad_diff_z = (grad_z_pred - grad_z_target) ** 2

    return (grad_diff_x.mean() + grad_diff_y.mean() + grad_diff_z.mean()) / 3



def mixed_loss(
    pred,
    target,
    low_mult: float = 1000.0,
    sh_mult: float = 500.0,
    ang_mult: float = 100.0,
    ang_dirs=None):
    """
    Combined loss, then return everything separately
    """
    # split channels
    low_diff = pred[:, 0:2, ...] - target[:, 0:2, ...]
    sh_diff  = pred[:, 2:,  ...] - target[:, 2:,  ...]

    # L2 loss on b0 + l0
    l2_low = (low_diff ** 2).mean()

    # L1 loss on angualr stuff
    l1_sh = sh_diff.abs().mean()

    total = low_mult * l2_low + sh_mult * l1_sh

    ang_loss = None
    if ang_dirs is not None:
        ang_loss = sh_angular_mse_loss(
            pred,
            target,
            directions=ang_dirs,
            lmax=6,
            gamma=10.0,
            b0_index=0)
        total = total + ang_mult * ang_loss

    # for logging
    low_scaled = low_mult * l2_low
    sh_scaled  = sh_mult  * l1_sh
    ang_scaled = ang_mult * ang_loss if ang_loss is not None else 0.0

    return total, low_scaled, sh_scaled, ang_scaled



def make_random_rotation_matrix():
    """
    Generate a random rotation matrix with (sortof) euler angles.
    """
    angles = np.random.uniform(-np.pi, np.pi, size=3) # radians!
    alpha, beta, gamma = angles

    # x axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

    # y axis
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    # z axis
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    # R = R(z) * R(y) * R(x)
    R = Rz @ Ry @ Rx
    return R


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None):
    """This function aligns a volume to a reference affine matrix.
    If aff_ref is None, the volume is simply aligned to the axes.
    :param volume: numpy array of the volume, can be 1- to 4d. We then assume that the first dimensions are spatial,
    and the last dimensions are channels.
    :param aff: original affine matrix of the volume.
    :param aff_ref: (optional) affine matrix we want to align volume and aff to. If aff_ref is None, the function simply
    aligns the volume to the axes.
    :param return_aff: (optional) whether to return the new affine matrix. Default is False.
    :param n_dims: (optional) number of dimensions of the volume. If None, this is computed from the volume.
    :return: reoriented volume, and affine matrix if return_aff is True."""

    # reformat input volume
    vol, aff, vol_shape, n_dims, n_channels = get_volume_info(volume, aff, return_volume=True)

    # align volume
    if aff_ref is None:
        if n_dims == 2:
            aff_ref = np.eye(3)
        else:
            aff_ref = np.eye(4)
        aff_ref[:-1, :-1] = np.diag(np.sqrt(np.sum(aff[:-1, :-1] * aff[:-1, :-1], 0)))

    # get RAS matrix and new volume via affine alignment
    RAS = aff_ref[:-1, :-1]
    RAS = RAS / np.sqrt(np.sum(RAS ** 2, 0))
    vox2ras = np.linalg.inv(aff)
    vox2ras = RAS.dot(vox2ras[:-1, :-1])
    new_vol = myzoom_np(vol, np.sqrt(np.sum(vox2ras * vox2ras, 0)))

    # get corresponding affine matrix
    if return_aff:
        true_aff = np.eye(4)
        true_aff[:-1, :-1] = RAS
        aff2 = true_aff.dot(aff)
        aff2[:-1, -1] = aff2[:-1, :-1].dot(-0.5 * (vol_shape - 1))
        aff2[:-1, -1] += np.array(new_vol.shape[:3]) / 2
        aff2 = np.dot(np.linalg.inv(aff_ref), aff2)
        return new_vol, aff2
    else:
        return new_vol



def pad_tensor_to_shape(vol, X, Y, Z):

    if vol.ndim < 3:
        raise ValueError(f"Expected at least 3D tensor, got {vol.shape}")

    *prefix, D, H, W = vol.shape

    def split_pad(s):
        diff = s[0] - s[1]
        pad1 = diff // 2
        pad2 = diff - pad1
        return (pad1, pad2)

    px = split_pad((X, D))
    py = split_pad((Y, H))
    pz = split_pad((Z, W))

    # torch.nn.functional.pad order for 4-D/5-D is (W, H, D, …):
    pad_tuple = (pz[0], pz[1],
                 py[0], py[1],
                 px[0], px[1])
    vol_padded = F.pad(vol, pad_tuple, mode="constant", value=0)

    return vol_padded, (px, py, pz)


# unpad from func above to original size
def unpad_tensor(vol_padded, pad_sizes):
    if np.isscalar(pad_sizes):
        pad_sizes = ((pad_sizes, pad_sizes),
                     (pad_sizes, pad_sizes),
                     (pad_sizes, pad_sizes))

    (px, py, pz) = pad_sizes

    D = vol_padded.shape[-3] - (px[0] + px[1])
    H = vol_padded.shape[-2] - (py[0] + py[1])
    W = vol_padded.shape[-1] - (pz[0] + pz[1])

    return vol_padded[..., px[0]:px[0]+D, py[0]:py[0]+H, pz[0]:pz[0]+W]


# SH coeff number when returned when including only even l
def num_evenl_coeffs_up_to(lmax):
    n_coeffs = 0
    for l in range(0, lmax + 1, 2):
        n_coeffs += 2 * l + 1
    return n_coeffs


# helper build like the standard mrtrix basis (read and only even order (i.e., no l1))
def _mrtrix_real_sh_basis(directions, lmax=6, device='cpu'):

    directions = directions.to(device)
    N = directions.size(0)

    # xyz to spherical angles
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]
    r = torch.sqrt(x * x + y * y + z * z).clamp(min=1e-9)

    theta = torch.acos((z / r).clamp(-1.0, 1.0))
    phi = torch.atan2(y, x)



    # if m < 0:  realY_{l,m} = sqrt(2) * Im(Y_l^{abs(m)})
    # if m = 0:  realY_{l,0} = Y_l^0
    # if m > 0:  realY_{l,m} = sqrt(2) * Re(Y_l^m)
    basis_cols = []
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):

            #associated Legendre pol
            mm = abs(m)
            Plm = torch.as_tensor(lpmv(mm, l, torch.cos(theta).cpu().numpy()), device=device, dtype=torch.float32)

            # norm factor...Condon-Shortley phase (-1)^m included
            sign = (-1) ** mm
            norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - mm) / math.factorial(l + mm))
            P_lm = sign * norm * Plm  # shape (N,)

            if m < 0:
                # imaginary part
                val = math.sqrt(2.0) * P_lm * torch.sin(mm * phi)
            elif m == 0:
                val = P_lm
            else:  # m > 0
                val = math.sqrt(2.0) * P_lm * torch.cos(m * phi)

            basis_cols.append(val.unsqueeze(1))

    basis = torch.cat(basis_cols, dim=1)
    basis = basis.to(torch.float32)
    return basis


# don't really use anymore (for angular sumsampling) but this is an apternative to
# the icosphere
def fibonacci_sphere(samples=1000, device='cpu'):

    indices = torch.arange(samples, device=device, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * indices / samples)
    theta = math.pi * (1 + 5 ** 0.5) * indices

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    dirs = torch.stack([x, y, z], dim=-1)
    return dirs

# evaluate SH volume at each direction in Ylm_matrix with matmul
def evaluate_mrtrix_sh(sh_volume, Ylm_matrix):

    B, nCoeffs, X, Y, Z = sh_volume.shape
    N = Ylm_matrix.size(0)

    sh_flat = sh_volume.flatten(start_dim=2, end_dim=4)

    # Expand Ylm to match batch
    Ylm_expanded = Ylm_matrix.unsqueeze(0).expand(B, -1, -1)

    signal_flat = torch.bmm(Ylm_expanded, sh_flat)

    signal_vol = signal_flat.reshape(B, N, X, Y, Z)

    return signal_vol


# get approximate principal direction with a pseudo soft-argmax
def soft_argmax_direction(signal_vol, directions, gamma=10.0, eps=1e-9):

    B, N, X, Y, Z = signal_vol.shape
    device = signal_vol.device

    sig_flat = signal_vol.flatten(start_dim=2)

    # avoid overflow
    max_vals, _ = sig_flat.max(dim=1, keepdim=True)
    sig_flat = sig_flat - max_vals

    exp_vals = torch.exp(gamma * sig_flat).clamp(max=1e20)
    weights = exp_vals / (exp_vals.sum(dim=1, keepdim=True) + eps)  # (B, N, M)

    # weighted sum of directions (N,3)
    weights_t = weights.transpose(1, 2)
    directions = directions.to(device)
    dirs_expanded = directions.unsqueeze(0).expand(B, -1, -1)

    sum_flat = torch.bmm(weights_t, dirs_expanded)
    norm_ = sum_flat.norm(dim=2, keepdim=True).clamp(min=eps)
    princ_2d = sum_flat / norm_

    princ_4d = princ_2d.reshape(B, X, Y, Z, 3).permute(0, 4, 1, 2, 3)
    return princ_4d

# get principal direction with soft argmax (main)
def principal_direction_from_sh(sh_volume, directions, lmax=6, gamma=10.0):

    device = sh_volume.device

    Ylm_matrix = _mrtrix_real_sh_basis(directions, lmax=lmax, device=device)
    signal_vol = evaluate_mrtrix_sh(sh_volume, Ylm_matrix)
    pdir = soft_argmax_direction(signal_vol, directions, gamma=gamma)
    return pdir

# total variation loss
def vec_tv(pev):

    dx = pev[:, :, 1:, :, :] - pev[:, :, :-1, :, :]
    dy = pev[:, :, :, 1:, :] - pev[:, :, :, :-1, :]
    dz = pev[:, :, :, :, 1:] - pev[:, :, :, :, :-1]

    tv = dx.norm(dim=1).mean() + dy.norm(dim=1).mean() + dz.norm(dim=1).mean()
    return tv

# get principal direction with soft argmax (main)
def sh_angular_loss(
    shA,
    shB,
    directions,
    lmax = 6,
    gamma = 10.0,
    eps = 1e-9):


    # principal dir for shA
    pA = principal_direction_from_sh(shA, directions, lmax=lmax, gamma=gamma)
    # principal dir for shB
    pB = principal_direction_from_sh(shB, directions, lmax=lmax, gamma=gamma)

    # assuming A is pred
    tvloss = vec_tv(pA) #no5 sure why this is still here


    dot = (pA * pB).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)

    loss = angle.mean()

    #print("angular loss is: ", loss)
    #print("TV loss is: ", tvloss)

    tot_ang_loss = loss + tvloss*100

    return tot_ang_loss


# angular MSE loss between (approximate) principal directions from SH
# loss = mean(angle^2) over all voxels and batch
def sh_angular_mse_loss(
    sh_pred,
    sh_gt,
    directions,
    lmax=6,
    gamma=10.0,
    b0_index=0,
    eps=1e-6,
):


    # drop b0
    sh_pred_sh = sh_pred[:, b0_index+1:, ...]
    sh_gt_sh   = sh_gt[:,   b0_index+1:, ...]


    p_pred = principal_direction_from_sh(sh_pred_sh, directions, lmax=lmax, gamma=gamma)
    p_gt   = principal_direction_from_sh(sh_gt_sh,   directions, lmax=lmax, gamma=gamma)

    dot = (p_pred * p_gt).sum(dim=1)

    # V and -V should be equivalent
    dot = dot.abs().clamp(min=0.0, max=1.0 - eps)

    # radians
    angle = torch.acos(dot)  # [0, pi/2] after abs

    loss = (angle ** 2).mean()
    return loss


_LAPLACIAN_KERNEL = torch.tensor(
    [[[0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]],
     [[0, 1, 0],
      [1, -6, 1],
      [0, 1, 0]],
     [[0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]]],
    dtype=torch.float32,
).unsqueeze(0).unsqueeze(0)

# 3D Laplacian filter 
def laplacian3d(x):

    B, C, *_ = x.shape
    k = _LAPLACIAN_KERNEL.to(x)
    k = k.repeat(C, 1, 1, 1, 1)
    return F.conv3d(x, k, padding=1, groups=C)


def laplacian_loss(pred, target, w=0.05):
    """
    L1 loss on Laplacian-filtered volumes.
    """
    return w * F.l1_loss(laplacian3d(pred), laplacian3d(target))


"""
    Approximate forward operator for data consistency loss:

    blur (per-axis sigma = blur_sigma_factor * ratio) + downsample by 'ratios', then upsample back to original size.
    all done to be differentiable

    Returns: degraded DiffSR'd volume
"""

def forward_blur_down_up(sh_hr, ratios, blur_sigma_factor):

    if sh_hr.ndim != 5:
        raise ValueError(f"expected (B,C,D,H,W), got {sh_hr.shape}")

    B, C, D, H, W = sh_hr.shape
    device = sh_hr.device
    dtype = sh_hr.dtype

    if isinstance(ratios, (list, tuple)):
        ratios = torch.tensor(ratios, dtype=dtype, device=device)

    if ratios.ndim == 1:
        ratios = ratios[None, :].expand(B, -1)
    elif ratios.ndim != 2 or ratios.shape[1] != 3:
        raise ValueError(f"ratios must be (B,3) or (3,), got {ratios.shape}")

    out_list = []

    for b in range(B):
        x = sh_hr[b : b + 1]
        r = ratios[b]

        #blur along each axis
        sigmas = blur_sigma_factor * r  # (3,)
        for axis in range(3):
            sigma = float(sigmas[axis].item())
            if sigma <= 0.0:
                continue
            ker_1d = torch.tensor(make_gaussian_kernel(sigma), dtype=dtype, device=device)
            pad = ker_1d.numel() // 2
            if axis == 0:   # D
                ker = ker_1d.view(1, 1, -1, 1, 1)
                padding = (pad, 0, 0)
            elif axis == 1: # H
                ker = ker_1d.view(1, 1, 1, -1, 1)
                padding = (0, pad, 0)
            else:           # W
                ker = ker_1d.view(1, 1, 1, 1, -1)
                padding = (0, 0, pad)
            ker = ker.repeat(C, 1, 1, 1, 1)
            x = F.conv3d(x, ker, padding=padding, groups=C)

        # downsample (not bejamin's way)
        d_lr = int(round(D / float(r[0].item())))
        h_lr = int(round(H / float(r[1].item())))
        w_lr = int(round(W / float(r[2].item())))
        x_lr = F.interpolate(x, size=(d_lr, h_lr, w_lr), mode="trilinear", align_corners=False)

        # rasample back
        x_hr = F.interpolate(x_lr, size=(D, H, W), mode="trilinear", align_corners=False)
        out_list.append(x_hr)

    return torch.cat(out_list, dim=0)



# zero pad tensor so that that spatial dims are multiples of stride (for inference)
def pad_to_stride_torch(vol, stride):

    _, X, Y, Z = vol.shape

    def split_pad(n):
        r = (-n) % stride
        return r // 2, r - r // 2       # (left, right)

    px = split_pad(X)
    py = split_pad(Y)
    pz = split_pad(Z)

    pad_tuple = (pz[0], pz[1],
                 py[0], py[1],
                 px[0], px[1])
    vol_padded = F.pad(vol, pad_tuple, mode="constant", value=0)

    return vol_padded, (px, py, pz)


def unpad_tensor(vol_padded, pad_sizes):
    if np.isscalar(pad_sizes):
        return vol_padded[:, pad_sizes:-pad_sizes, pad_sizes:-pad_sizes, pad_sizes: -pad_sizes]
    else:
        (px_l, px_r), (py_l, py_r), (pz_l, pz_r) = pad_sizes

        x_end = -px_r if px_r > 0 else None
        y_end = -py_r if py_r > 0 else None
        z_end = -pz_r if pz_r > 0 else None

        return vol_padded[:, px_l:x_end, py_l:y_end, pz_l:z_end]
