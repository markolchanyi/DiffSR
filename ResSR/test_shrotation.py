import numpy as np, nibabel as nib, pyshtools as pysh
import scipy.ndimage as ndi, math, time
from joblib import Parallel, delayed            # pip install joblib
from typing import Tuple

# ======= globals that helpers need =======
SQRT2     = math.sqrt(2.0)
INV_SQRT2 = 1.0 / SQRT2
# =========================================

# ------------------------------------------------------------------
def bulk_affine_rotate(vol4d, euler_deg, order=3, pad_frac=0.15, n_jobs=30):
    """
    Rotate a 4-D volume (…X×Y×Z×C) in voxel space.
    Channels are NOT mixed – transform acts only on spatial axes.
    """
    a, b, g = map(math.radians, euler_deg)
    Nx, Ny, Nz, C = vol4d.shape
    pad = int(pad_frac * max(Nx, Ny, Nz))
    volP = np.pad(vol4d, ((pad, pad), (pad, pad), (pad, pad), (0, 0)),
                  mode="constant", constant_values=0)

    # voxel-space rotation matrix Rvox = Rz·Ry·Rx
    Rx = [[1,0,0],[0, math.cos(a),-math.sin(a)],[0, math.sin(a), math.cos(a)]]
    Ry = [[ math.cos(b),0, math.sin(b)],[0,1,0],[-math.sin(b),0, math.cos(b)]]
    Rz = [[ math.cos(g),-math.sin(g),0],[ math.sin(g), math.cos(g),0],[0,0,1]]
    Rvox = np.matmul(Rz, np.matmul(Ry, Rx))
    Avox = Rvox.T                               # inverse for ndimage

    NxP, NyP, NzP, _ = volP.shape
    centreP = np.array([(NxP-1)/2, (NyP-1)/2, (NzP-1)/2])
    offsetP = centreP - Avox @ centreP

    # threaded per-channel warp (3-D each -> releases GIL)
    def warp_one(chan):
        return ndi.affine_transform(
            chan, Avox, offset=offsetP,
            order=order, mode='constant', cval=0.0)

    t0 = time.perf_counter()
    warped = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(warp_one)(volP[..., k]) for k in range(C))
    t = time.perf_counter() - t0
    print(f"  bulk warp ({C} ch) : {t:0.2f} s")

    vol_rot = np.stack(warped, axis=-1)
    return vol_rot[pad:pad+Nx, pad:pad+Ny, pad:pad+Nz, :]


# ------------------------------------------------------------------
def sh_rotate_coeff(sh4d, euler_deg, lmax=6):
    """
    Rotate MR-trix real SH coefficients (last axis) inside each voxel.
    Assumes even-ℓ only (28 coeffs for lmax=6).
    """
    a, b, g = map(math.radians, euler_deg)
    K      = lmax // 2
    ncoef  = 2*K*K + 3*K + 1            # 28
    if sh4d.shape[-1] != ncoef:
        raise ValueError("Last axis size != expected SH coeff count.")

    # build 28×28 matrix once
    mr2 = [(l, m) for l in range(0, lmax+1, 2) for m in range(-l, l+1)]
    dj  = pysh.rotate.djpi2(lmax)
    R   = np.zeros((ncoef, ncoef))
    for i, (l, m) in enumerate(mr2):
        e = np.zeros((2, lmax+1, lmax+1))
        if   m == 0: e[0, l, 0] = 1
        elif m > 0:  e[0, l, m] = INV_SQRT2
        else:        e[1, l, -m] = INV_SQRT2 * (-1)**(m+1)
        r = pysh.rotate.SHRotateRealCoef(e, [a, b, g], dj)
        col = np.empty(ncoef)
        for j, (l2, m2) in enumerate(mr2):
            if   m2 == 0: col[j] = r[0, l2, 0]
            elif m2 > 0:  col[j] = SQRT2 * r[0, l2, m2]
            else:         col[j] = SQRT2 * (-1)**(m2+1) * r[1, l2, -m2]
        R[:, i] = col

    flat = sh4d.reshape(-1, ncoef)
    t0 = time.perf_counter()
    flat_rot = flat @ R.T
    t = time.perf_counter() - t0
    print(f"  SH coeff DGEMM   : {t:0.2f} s")

    return flat_rot.reshape(sh4d.shape)


# ################# driver ################################################
if __name__ == "__main__":

    # ---------- user parameters ----------
    INFILE        = "../scripts/tmp_epoch_output/target.nii.gz"
    OUTFILE       = "../scripts/tmp_epoch_output/target_rot.nii.gz"
    LMAX          = 6                     # even  ← add this
    EULER_DEG     = (0., 20., 0.)         # a, b, g  (deg, Z–Y′–Z″)
    INTERP_ORDER  = 3
    PAD_FRACTION  = 0.1
    N_JOBS        = -1
    # -------------------------------------
    # ------- load & split -----------------
    vol = nib.load(INFILE).get_fdata(dtype=np.float32)  # (X,Y,Z,29)
    b0  = vol[..., 0:1]           # keep 4-D shape for bulk function
    sh  = vol[..., 1:]            # 28-coeff slab

    # ------- bulk warp (all 29) ----------
    print("Bulk spatial rotation:")
    vol_rot = bulk_affine_rotate(vol, EULER_DEG,
                                 order=INTERP_ORDER,
                                 pad_frac=PAD_FRACTION,
                                 n_jobs=N_JOBS)
    vol_rot = vol
    # ------- SH coeff rotation -----------
    print("Coefficient rotation:")
    sh_rot = sh_rotate_coeff(vol_rot[..., 1:], EULER_DEG, lmax=LMAX)

    # replace rotated SH channels
    vol_rot[..., 1:] = sh_rot

    # ------- save ------------------------
    nib.save(nib.Nifti1Image(vol_rot, np.eye(4)), OUTFILE)
    print(f"Saved → {OUTFILE}")

    #nib.save(nib.Nifti1Image(vol, np.eye(4)), "target_noheader.nii.gz")

