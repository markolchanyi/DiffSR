# This file contains a bunch of functions from Benjamin's lab2im package
# (it's just much lighter to import...)
import nibabel as nib
import numpy as np
import os
import torch
import math
from torch.nn import L1Loss, MSELoss
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

# Save nifti or mgz file
def save_volume(volume, aff, path):

    header = nib.Nifti1Header()

    if aff is None:
        aff = np.eye(4)

    nifti = nib.Nifti1Image(volume, aff, header)

    nib.save(nifti, path)



def myzoom_torch(X, factor, device='cpu'):

    if len(X.shape)==3:
        X = X[..., None]

    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(delta[0], delta[0] + newsize[0] / factor[0], 1 / factor[0], device=device)
    vy = torch.arange(delta[1], delta[1] + newsize[1] / factor[1], 1 / factor[1], device=device)
    vz = torch.arange(delta[2], delta[2] + newsize[2] / factor[2], 1 / factor[2], device=device)

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0]-1)] = (X.shape[0]-1)
    vy[vy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    vz[vz > (X.shape[2] - 1)] = (X.shape[2] - 1)

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0]-1)] = (X.shape[0]-1)
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1]-1)] = (X.shape[1]-1)
    wcy = vy - fy
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2]-1)] = (X.shape[2]-1)
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros([newsize[0], newsize[1], newsize[2], X.shape[3]], device=device)

    for channel in range(X.shape[3]):
        Xc = X[:,:,:,channel]

        tmp1 = torch.zeros([newsize[0], Xc.shape[1], Xc.shape[2]], device=device)
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] +  wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros([newsize[0], newsize[1], Xc.shape[2]], device=device)
        for j in range(newsize[1]):
            tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] +  wcy[j] * tmp1[:, cy[j], :]
        for k in range(newsize[2]):
            Y[:, :, k, channel] = wfz[k] * tmp2[:, :, fz[k]] +  wcz[k] * tmp2[:, :, cz[k]]

    if Y.shape[3] == 1:
        Y = Y[:,:,:, 0]

    return Y

# Nearst negithbor or trilinear 3D interpolation with pytorch
def fast_3D_interp_torch(X, II, JJ, KK, mode, device='cpu'):
    if mode=='nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if len(X.shape)==3:
            X = X[..., None]
        Y = torch.zeros([*II.shape, X.shape[3]], device=device)
        for channel in range(X.shape[3]):
            aux = X[:, :, :, channel]
            Y[:,:,:,channel] = aux[IIr, JJr, KKr]
        if Y.shape[3] == 1:
            Y = Y[:, :, :, 0]

    elif mode=='linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz

        if len(X.shape)==3:
            X = X[..., None]

        Y = torch.zeros([*II.shape, X.shape[3]], device=device)
        for channel in range(X.shape[3]):
            Xc = X[:, :, :, channel]

            c000 = Xc[fx, fy, fz]
            c100 = Xc[cx, fy, fz]
            c010 = Xc[fx, cy, fz]
            c110 = Xc[cx, cy, fz]
            c001 = Xc[fx, fy, cz]
            c101 = Xc[cx, fy, cz]
            c011 = Xc[fx, cy, cz]
            c111 = Xc[cx, cy, cz]

            c00 = c000 * wfx + c100 * wcx
            c01 = c001 * wfx + c101 * wcx
            c10 = c010 * wfx + c110 * wcx
            c11 = c011 * wfx + c111 * wcx

            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy

            c = c0 * wfz + c1 * wcz

            Yc = torch.zeros(II.shape, device=device)
            Yc[ok] = c.float()
            Y[:,:,:,channel] = Yc

        if Y.shape[3]==1:
            Y = Y[:,:,:,0]

    else:
        raise Exception('mode must be linear or nearest')

    return Y


# Make a discrete gaussian kernel
def make_gaussian_kernel(sigma):
    sl = np.ceil(sigma * 2.5).astype(int)
    v = np.arange(-sl, sl+1)
    gauss = np.exp((-(v / sigma)**2 / 2))
    kernel = gauss / np.sum(gauss)

    return kernel


#
#

#
#

#
#
#
#
def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
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
        var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
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
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var

def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var

def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            volume = np.swapaxes(volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            volume = np.flip(volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (volume.shape[i] - 1)

    if return_aff:
        return volume, aff_flo
    else:
        return volume

def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    return img_ras_axes




def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
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

def resample_like(vol_ref, aff_ref, vol_flo, aff_flo, method='linear'):
    """This function reslices a floating image to the space of a reference image
    :param vol_res: a numpy array with the reference volume
    :param aff_ref: affine matrix of the reference volume
    :param vol_flo: a numpy array with the floating volume
    :param aff_flo: affine matrix of the floating volume
    :param method: linear or nearest
    :return: resliced volume
    """

    T = np.matmul(np.linalg.inv(aff_flo), aff_ref)

    xf = np.arange(0, vol_flo.shape[0])
    yf = np.arange(0, vol_flo.shape[1])
    zf = np.arange(0, vol_flo.shape[2])

    my_interpolating_function = rgi((xf, yf, zf), vol_flo, method=method, bounds_error=False, fill_value=0.0)

    xr = np.arange(0, vol_ref.shape[0])
    yr = np.arange(0, vol_ref.shape[1])
    zr = np.arange(0, vol_ref.shape[2])

    xrg, yrg, zrg = np.meshgrid(xr, yr, zr, indexing='ij', sparse=False)
    n = xrg.size
    xrg = xrg.reshape([n])
    yrg = yrg.reshape([n])
    zrg = zrg.reshape([n])
    bottom = np.ones_like(xrg)
    coords = np.stack([xrg, yrg, zrg, bottom])
    coords_new = np.matmul(T, coords)[:-1, :]
    result = my_interpolating_function((coords_new[0, :], coords_new[1, :], coords_new[2, :]))

    if vol_ref.size == result.size:
        return result.reshape(vol_ref.shape)
    else:
        return result.reshape([*vol_ref.shape, vol_flo.shape[-1]])





# Computer linear (flattened) indices for linear interpolation of a volume of size nx x ny x nz at locations xx, yy, zz
# (as well as a boolean vector 'ok' telling which indices are inbounds)
# Note that it doesn't support sparse xx, yy, zz
def nn_interpolator_indices(xx, yy, zz, nx, ny, nz):
    xx2r = np.round(xx).astype(int)
    yy2r = np.round(yy).astype(int)
    zz2r = np.round(zz).astype(int)
    ok = (xx2r >= 0) & (yy2r >= 0) & (zz2r >= 0) & (xx2r <= nx - 1) & (yy2r <= ny - 1) & (zz2r <= nz - 1)
    idx = xx2r[ok] + nx * yy2r[ok] + nx * ny * zz2r[ok]
    return idx, ok

# Similar to nn_interpolator_indices but does not check out of bounds.
# Note that it *does* support sparse xx, yy, zz
def nn_interpolator_indices_nocheck(xx, yy, zz, nx, ny, nz):
    xx2r = np.round(xx).astype(int)
    yy2r = np.round(yy).astype(int)
    zz2r = np.round(zz).astype(int)
    idx = xx2r + nx * yy2r + nx * ny * zz2r
    return idx

# Subsamples a volume by a given ration in each dimension.
# It carefully accounts for origin shifts
def subsample(X, ratio, size, method='linear', return_locations=False):
    xi = np.arange(0.5 * (ratio[0] - 1.0), size[0] - 1 + 1e-6, ratio[0])
    yi = np.arange(0.5 * (ratio[1] - 1.0), size[1] - 1 + 1e-6, ratio[1])
    zi = np.arange(0.5 * (ratio[2] - 1.0), size[2] - 1 + 1e-6, ratio[2])
    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    interpolator = rgi((range(size[0]), range(size[1]), range(size[2])), X, method=method)
    Y = interpolator((xig, yig, zig))
    if return_locations:
        return Y, xig, yig, zig
    else:
        return Y


def upsample(X, ratio, size, method='linear', return_locations=False):
    start = (1.0 - ratio[0]) / (2.0 * ratio[0])
    inc = 1.0 / ratio[0]
    end = start + inc * size[0] - 1e-6
    xi = np.arange(start, end, inc)
    xi[xi < 0] = 0
    xi[xi > X.shape[0] - 1] = X.shape[0] - 1

    start = (1.0 - ratio[1]) / (2.0 * ratio[1])
    inc = 1.0 / ratio[1]
    end = start + inc * size[1] - 1e-6
    yi = np.arange(start, end, inc)
    yi[yi < 0] = 0
    yi[yi > X.shape[1] - 1] = X.shape[1] - 1

    start = (1.0 - ratio[2]) / (2.0 * ratio[2])
    inc = 1.0 / ratio[2]
    end = start + inc * size[2] - 1e-6
    zi = np.arange(start, end, inc)
    zi[zi < 0] = 0
    zi[zi > X.shape[2] - 1] = X.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    interpolator = rgi((range(X.shape[0]), range(X.shape[1]), range(X.shape[2])), X, method=method)
    Y = interpolator((xig, yig, zig))

    if return_locations:
        return Y, xig, yig, zig
    else:
        return Y

# Augmentation of FA with gaussian noise and gamma transform
def augment_fa(X, gamma_std, max_noise_std_fa):
    gamma_fa = np.exp(gamma_std * np.random.randn(1)[0])
    noise_std = max_noise_std_fa * np.random.rand(1)[0]
    Y = X + noise_std * np.random.randn(*X.shape)
    Y[Y < 0] = 0
    Y[Y > 1] = 1
    Y = Y ** gamma_fa
    return Y

# Augmentation of T1 intensities with random contrast, brightness, gamma, and gaussian noise
def augment_t1(X, gamma_std, contrast_std, brightness_std, max_noise_std):
    # TODO: maybe add bias field? If we're working with FreeSurfer processed images maybe it's not too important
    gamma_t1 = np.exp(gamma_std * np.random.randn(1)[0])  # TODO: maybe make it spatially variable?
    contrast = np.min((1.4, np.max((0.6, 1.0 + contrast_std * np.random.randn(1)[0]))))
    brightness = np.min((0.4, np.max((-0.4, brightness_std * np.random.randn(1)[0]))))
    noise_std = max_noise_std * np.random.rand(1)[0]
    Y = ((X - 0.5) * contrast + (0.5 + brightness)) + noise_std * np.random.randn(*X.shape)
    Y[Y < 0] = 0
    Y[Y > 1] = 1
    Y = Y ** gamma_t1
    return Y


def rescale_voxel_size(volume, aff, new_vox_size):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gauss_filt(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = rgi((x, y, z), volume_filt)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)

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


def gradient_loss(pred, target):
    # Calculate gradient differences along spatial dimensions only (not the channel dimension)
    grad_x_pred = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    grad_y_pred = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    grad_z_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]

    grad_x_target = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    grad_y_target = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    grad_z_target = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

    # Ensure the dimensions match before calculating the loss
    grad_diff_x = (grad_x_pred - grad_x_target) ** 2
    grad_diff_y = (grad_y_pred - grad_y_target) ** 2
    grad_diff_z = (grad_z_pred - grad_z_target) ** 2

    # Combine gradient differences
    return (grad_diff_x.mean() + grad_diff_y.mean() + grad_diff_z.mean()) / 3



def mixed_loss(pred, target, l1_loss_fn, l2_loss_fn, alpha=0.5, beta=0.1):
    l1_loss = l1_loss_fn(pred, target)
    l2_loss = l2_loss_fn(pred, target)
    #grad_loss = gradient_loss(pred, target)
    #print("L1: ", l1_loss)
    #print("L2: ", l2_loss)
    #print("Grad: ", grad_loss)
    return alpha * l1_loss + (1 - alpha) * l2_loss


def random_crop(hr, crop_size):
    # hr is expected to be of shape (N, N, N, C)
    # only return a crop where majority of l=0 intensities are non-zero
    # i.e., not the edge of the orig volume with no info
    spatial_dims = hr.shape[:-1]
    while True:
        start = [torch.randint(0, spatial_dims[i] - crop_size[i], (1,)).item() for i in range(3)]
        end = [start[i] + crop_size[i] for i in range(3)]

        crop = hr[start[0]:end[0], start[1]:end[1], start[2]:end[2], :]
        non_zero_fraction = (crop[..., 0] != 0).float().mean().item()

        if non_zero_fraction > 0.5:
            return crop

def make_rotation_matrix(angles):
    """
    Generates a 3D rotation matrix from three rotation angles (x, y, z).
    Args:
    - angles: tensor of shape (3,) representing rotation angles in radians.

    Returns:
    - Rotation matrix (3x3).
    """
    cos_x, cos_y, cos_z = torch.cos(angles)
    sin_x, sin_y, sin_z = torch.sin(angles)

    Rx = torch.tensor([[1, 0, 0],
                       [0, cos_x, -sin_x],
                       [0, sin_x, cos_x]])

    Ry = torch.tensor([[cos_y, 0, sin_y],
                       [0, 1, 0],
                       [-sin_y, 0, cos_y]])

    Rz = torch.tensor([[cos_z, -sin_z, 0],
                       [sin_z, cos_z, 0],
                       [0, 0, 1]])

    # Combine rotations (Rz * Ry * Rx)
    R = torch.matmul(Rz, torch.matmul(Ry, Rx))
    return R

def wigner_d_matrix(l, beta):
    """
    Compute the Wigner d-matrix for spherical harmonics of degree l.

    Args:
    - l: Degree of the SH (integer).
    - beta: Rotation angle (around the Y-axis) in radians.

    Returns:
    - d_matrix: Wigner d-matrix of shape (2l+1, 2l+1).
    """
    d_matrix = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.float32)

    for m in range(-l, l + 1):
        for mp in range(-l, l + 1):
            d_matrix[m + l, mp + l] = wigner_d_matrix_element(l, m, mp, beta)

    return d_matrix

def wigner_d_matrix_element(l, m, mp, beta):
    """
    Compute the Wigner d-matrix element for the given l, m, mp, and beta angle.

    Args:
    - l: Degree of the spherical harmonic.
    - m: Order of the spherical harmonic.
    - mp: Order after rotation.
    - beta: Euler angle (rotation around the Y-axis).

    Returns:
    - d_l_m_mp: The Wigner small d-matrix element.
    """
    # Ensure m and mp are within the valid range for the degree l
    if abs(m) > l or abs(mp) > l:
        return 0

    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)

    # Compute the element using a simplified form based on associated Legendre polynomials
    pre_factor = math.sqrt(math.factorial(l + m) * math.factorial(l - m) *
                           math.factorial(l + mp) * math.factorial(l - mp))
    sum_term = 0

    # Summation over k, part of the Wigner d-matrix formula
    for k in range(max(0, m - mp), min(l - mp, l + m) + 1):
        numerator = (-1)**k * binomial_coefficient(l + m, k) * binomial_coefficient(l - m, l - mp - k)
        denominator = math.factorial(l - mp - k)
        term = numerator / denominator * (cos_beta / 2)**(2 * l - 2 * k - m + mp) * (sin_beta / 2)**(m - mp)
        sum_term += term

    return pre_factor * sum_term

def rotate_sh_vector(sh_coeffs, R, lmax=6):
    """
    Rotate SH coefficients according to a 3D rotation matrix.

    Args:
    - sh_coeffs: Tensor of SH coefficients (C,) where C is the number of coefficients (e.g., 28 for lmax=6).
    - R: Rotation matrix (3x3).
    - lmax: Maximum SH degree (default is 6).

    Returns:
    - rotated_sh_coeffs: Rotated SH coefficients of the same shape as input.
    """
    # Convert rotation matrix to Euler angles
    alpha, beta, gamma = rotation_matrix_to_euler_angles(R)

    device = sh_coeffs.device
    dtype = torch.float32

    R = R.to(device, dtype=dtype)

    # Initialize rotated SH coefficients
    rotated_sh_coeffs = torch.zeros_like(sh_coeffs, dtype=dtype, device=device)

    # Index to keep track of SH coefficients
    idx = 0

    for l in range(0, lmax + 1, 2):  # Spherical harmonics are even orders only
        D_l = wigner_d_matrix(l, beta).to(device, dtype=dtype)  # Get Wigner D-matrix for degree l
        coeffs_l = sh_coeffs[idx:idx + (2 * l + 1)].to(device, dtype=dtype)  # Extract SH coefficients for degree l
        rotated_coeffs_l = torch.matmul(D_l, coeffs_l)  # Rotate the SH coefficients using Wigner D-matrix
        rotated_sh_coeffs[idx:idx + (2 * l + 1)] = rotated_coeffs_l  # Store the rotated coefficients
        idx += (2 * l + 1)  # Move to the next set of SH coefficients

    return rotated_sh_coeffs

def binomial_coefficient(n, k):
    """
    Compute the binomial coefficient C(n, k) = n! / (k! * (n - k)!).
    This is equivalent to math.comb(n, k) in Python 3.8+, but works in earlier versions.

    Args:
    - n: The total number of items.
    - k: The number of chosen items.

    Returns:
    - The binomial coefficient.
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Take advantage of symmetry
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c

def rotation_matrix_to_euler_angles(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (alpha, beta, gamma).

    Args:
    - R: Rotation matrix (3x3).

    Returns:
    - alpha, beta, gamma: Euler angles (in radians).
    """
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        alpha = torch.atan2(R[2, 1], R[2, 2])
        beta = torch.atan2(-R[2, 0], sy)
        gamma = torch.atan2(R[1, 0], R[0, 0])
    else:
        alpha = torch.atan2(-R[1, 2], R[1, 1])
        beta = torch.atan2(-R[2, 0], sy)
        gamma = 0

    return alpha, beta, gamma

def random_rotate_sh(hr, lmax=6, probability=0.8):
    """
    Randomly rotate spherical harmonic coefficients with a given probability.

    Args:
    - hr: Tensor of shape (N, N, N, C), where C is the number of SH coefficients (e.g., 28).
    - lmax: Maximum SH degree (default is 6).
    - probability: Probability of applying the rotation.

    Returns:
    - hr_rot: Tensor with rotated SH coefficients (same shape as hr).
    """
    if torch.rand(1).item() < probability:
        # Generate random rotation angles
        angles = torch.rand(3) * 2 * torch.pi  # Random angles in radians

        # Create a 3D rotation matrix
        R = make_rotation_matrix(angles)

        # Rotate SH coefficients voxel by voxel
        N = hr.shape[0]
        C = hr.shape[-1]  # Number of SH coefficients
        hr_rot = torch.zeros_like(hr, dtype=torch.complex64)

        # Rotate each voxel's SH coefficients
        for x in range(N):
            for y in range(N):
                print("rotating for y: ", y)
                for z in range(N):
                    sh_coeffs = hr[x, y, z, :]
                    rotated_coeffs = rotate_sh_vector(sh_coeffs, R, lmax=lmax)
                    hr_rot[x, y, z, :] = rotated_coeffs

        return hr_rot

    return hr  # Return the original hr if no rotation is applied


def make_random_rotation_matrix():
    """
    Generate a random 3D rotation matrix using random Euler angles.

    Returns:
    - R: A random 3x3 rotation matrix.
    """
    # Generate random Euler angles (alpha, beta, gamma) in radians
    angles = torch.rand(3) * 2 * torch.pi  # Random angles in the range [0, 2*pi]
    return make_rotation_matrix(angles)  # Generate the rotation matrix


def batch_rotate_sh(hr, lmax=6, probability=1.0):
    """
    Apply random rotation to SH coefficients for the entire 3D volume according to a random 3D rotation matrix.

    Args:
    - hr: Tensor of SH coefficients with shape (N, N, N, C), where C is the number of SH coefficients (e.g., 28 for lmax=6).
    - lmax: Maximum SH degree (default is 6).
    - probability: Probability of applying the rotation (between 0 and 1).

    Returns:
    - hr_rot: The randomly rotated SH coefficients for the entire volume (same shape as input).
    """
    # Check if we should apply the rotation based on the probability
    if torch.rand(1).item() < probability:
        # Generate a random rotation matrix
        R = make_random_rotation_matrix()

        # Ensure the rotation matrix and SH coefficients are on the same device and are float32
        device = hr.device
        dtype = torch.float32
        R = R.to(device, dtype=dtype)

        # Convert rotation matrix to Euler angles
        alpha, beta, gamma = rotation_matrix_to_euler_angles(R)

        # Initialize the rotated SH coefficients volume (same shape as hr)
        hr_rot = torch.zeros_like(hr, device=device, dtype=dtype)

        # Iterate over SH degrees (even orders only)
        idx = 0
        for l in range(0, lmax + 1, 2):
            # Get Wigner d-matrix for the current degree l
            D_l = wigner_d_matrix(l, beta).to(device, dtype=dtype)

            # Extract the SH coefficients for degree l for the entire volume (shape: (N, N, N, 2l+1))
            coeffs_l = hr[:, :, :, idx:idx + (2 * l + 1)].to(device, dtype=dtype)

            # Reshape the SH coefficients to (N*N*N, 2l+1) for batch processing
            N = hr.shape[0]
            coeffs_l = coeffs_l.reshape(-1, 2 * l + 1)  # Shape: (N*N*N, 2l+1)

            # Apply Wigner D-matrix to all SH coefficients at once using batch matrix multiplication
            rotated_coeffs_l = torch.matmul(coeffs_l, D_l.T)  # Shape: (N*N*N, 2l+1)

            # Reshape back to the original volume shape (N, N, N, 2l+1)
            rotated_coeffs_l = rotated_coeffs_l.reshape(N, N, N, 2 * l + 1)

            # Store the rotated SH coefficients back in the hr_rot volume
            hr_rot[:, :, :, idx:idx + (2 * l + 1)] = rotated_coeffs_l
            idx += (2 * l + 1)

        return hr_rot

    # If no rotation is applied, return the original SH tensor
    return hr


def median_iqr_scaling(sh_tensor, l0_index=0, k=2.0, new_min=0.0, new_max=1.0, eps=1e-8):
    # Extract the l=0 channel
    l0 = sh_tensor[:, l0_index]

    # Compute median and IQR
    median = torch.mean(l0)
    q = torch.tensor([0.75, 0.25], dtype=l0.dtype, device=l0.device)
    q75, q25 = torch.quantile(l0, q)
    iqr = q75 - q25

    # Debugging statements
    print(f"Median: {median.item():.4f}, IQR: {iqr.item():.4f}")

    # Define scaling bounds based on median and IQR
    lower_bound = median - k * iqr
    upper_bound = median + k * iqr

    print(f"Lower Bound: {lower_bound.item():.4f}, Upper Bound: {upper_bound.item():.4f}")

    # Apply the scaling formula
    if iqr < eps:
        # If IQR is too small, set normalized l0 to 0.5 to avoid division by zero
        l0_normalized = torch.full_like(l0, 0.5)
        print("IQR is too small. Setting normalized l0 to 0.5.")
    else:
        # Scale the l0 channel
        l0_normalized = (l0 - lower_bound) / (upper_bound - lower_bound)
        # Shift to center around 0.5
        l0_normalized = l0_normalized * (new_max - new_min) + new_min
        l0_normalized = torch.clamp(l0_normalized, min=new_min, max=new_max)

    sh_tensor_normalized = sh_tensor.clone()
    sh_tensor_normalized[:, l0_index] = l0_normalized

    return sh_tensor_normalized
