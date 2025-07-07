import numpy as np, nibabel as nib, pyshtools as pysh
import scipy.ndimage as ndi, math, time
from joblib import Parallel, delayed            # pip install joblib
from typing import Tuple
import numpy as np, scipy.ndimage as ndi, scipy.linalg, math, random
from scipy.spatial.transform import Rotation
import pyshtools as pysh

# ======= globals that helpers need =======
SQRT2     = math.sqrt(2.0)
INV_SQRT2 = 1.0 / SQRT2
# =========================================


def mrtrix2cilm(vec28, lmax=6):
    out = np.zeros((2, lmax+1, lmax+1))
    k = 0
    for l in range(0, lmax+1, 2):
        for m in range(-l, l+1):
            v = vec28[k]; k += 1
            if   m == 0: out[0,l,0] = v
            elif m > 0:  out[0,l,m] = v * INV_SQRT2
            else:        out[1,l,-m] = INV_SQRT2 * (-1)**(m+1) * v
    return out

def cilm2mrtrix(cilm, lmax=6):
    out = np.empty(28, np.float32); k = 0
    for l in range(0, lmax+1, 2):
        for m in range(-l, l+1):
            if   m == 0: out[k] = cilm[0,l,0]
            elif m > 0:  out[k] = SQRT2 * cilm[0,l,m]
            else:        out[k] = SQRT2 * (-1)**(m+1) * cilm[1,l,-m]
            k += 1
    return out

# ---------- Jacobian & positivity check ------------------------
# ---------- Jacobian & positivity check ------------------------
def _jacobian_field(disp):
    """
    Central-difference ∇u.
    Returns array shape (*XYZ, 3, 3) — component axes last.
    """
    gx = np.gradient(disp[..., 0], edge_order=2)
    gy = np.gradient(disp[..., 1], edge_order=2)
    gz = np.gradient(disp[..., 2], edge_order=2)

    # unpack
    du_dx, du_dy, du_dz = gx
    dv_dx, dv_dy, dv_dz = gy
    dw_dx, dw_dy, dw_dz = gz

    # stack 9 components then reshape → (...,3,3)
    J = np.stack([du_dx, du_dy, du_dz,
                  dv_dx, dv_dy, dv_dz,
                  dw_dx, dw_dy, dw_dz], axis=-1)

    return J.reshape(*disp.shape[:3], 3, 3)

def _jacobian_ok(disp, eps=-1e-4, rim=2):
    """Accept if det(I+∇u) > eps everywhere except a rim voxels."""
    J = _jacobian_field(disp[rim:-rim, rim:-rim, rim:-rim])
    dets = np.linalg.det(J + np.eye(3, dtype=disp.dtype))
    print("det is ", np.min(dets))
    return np.all(dets > eps)

# ---------- random patch displacement --------------------------
def _cos_taper(length, blend):
    ramp = np.ones(length, np.float32)
    if blend > 0:
        k = np.arange(blend, dtype=np.float32) / blend
        ramp[:blend]        = 0.5 * (1 - np.cos(math.pi * k))
        ramp[-blend:] = ramp[:blend][::-1]
    return ramp

def make_patch_displacement(vol_shape,
                            patch_size=(12,12,12),
                            spacing=4,
                            warp_scale=2.0,
                            blend=3,
                            mode="both",
                            max_tries=1000):
    """
    Returns:
      disp : (X,Y,Z,3) float32   (zero outside ROI)
      roi  : (x0,x1,y0,y1,z0,z1)
    """
    X,Y,Z = vol_shape
    px,py,pz = patch_size

    for _ in range(max_tries):
        x0 = random.randint(0, max(X-px,1));  x1 = x0+px
        y0 = random.randint(0, max(Y-py,1));  y1 = y0+py
        z0 = random.randint(0, max(Z-pz,1));  z1 = z0+pz

        disp_patch = np.zeros((pz,py,px,3), np.float32)

        if mode in ("bspline", "both"):
            nx,ny,nz = (int(np.ceil(px/spacing))+1,
                        int(np.ceil(py/spacing))+1,
                        int(np.ceil(pz/spacing))+1)
            ctrl = np.random.uniform(-warp_scale, warp_scale,
                                     size=(nx,ny,nz,3)).astype(np.float32)
            zz,yy,xx = np.meshgrid(np.arange(pz), np.arange(py), np.arange(px),
                                   indexing='ij')
            coords = np.vstack([(zz/spacing).ravel(),
                                (yy/spacing).ravel(),
                                (xx/spacing).ravel()])
            for c in range(3):
                vol_c = np.transpose(ctrl[...,c], (2,1,0))
                disp_patch[...,c] += ndi.map_coordinates(
                        vol_c, coords, order=3, mode='nearest'
                    ).reshape(pz,py,px)

        if mode in ("shear", "both"):
            sx, sy, sz = [random.uniform(-0.25,0.25) for _ in range(3)]
            S = np.array([[1, sx, sx],
                          [sy, 1, sy],
                          [sz, sz, 1 ]], np.float32)
            xx,yy,zz = np.meshgrid(np.arange(px), np.arange(py), np.arange(pz),
                                   indexing='ij')
            coords   = np.stack([xx,yy,zz],-1).reshape(-1,3)
            sheared  = coords @ S.T
            disp_patch += (sheared - coords).reshape(px,py,pz,3).transpose(2,1,0,3)

        # taper
        rx = _cos_taper(px, blend); ry=_cos_taper(py,blend); rz=_cos_taper(pz,blend)
        mask = rz[:,None,None]*ry[None,:,None]*rx[None,None,:]
        disp_patch *= mask[...,None]

        if _jacobian_ok(disp_patch):
            disp = np.zeros((X,Y,Z,3), np.float32)
            disp[x0:x1, y0:y1, z0:z1] = disp_patch.transpose(2,1,0,3)
            print("jak succeeded!!")
            return disp, (x0,x1,y0,y1,z0,z1)
        else:
            print("jacobian failed")

    # fallback: give zero field
    return np.zeros((X,Y,Z,3), np.float32), (0,0,0,0,0,0)

# ---------- main warp + local SH rotation ----------------------
def apply_patch_warp_and_reorient(vol,
                                  disp,
                                  *,
                                  lmax=6,
                                  noise_std_range=(0.0,0.02)):
    """
    vol  : (X,Y,Z,29) float32
    disp : (X,Y,Z,3)  local displacement (zero outside ROI)
    Returns a NEW 4-D array with warp + re-orientation + in-patch noise.
    """
    X,Y,Z,_ = vol.shape

    # 1 ▸ spatial warp (linear)
    gz,gy,gx = np.meshgrid(np.arange(Z),np.arange(Y),np.arange(X),
                           indexing='ij')
    coords = np.array([gz - disp[...,2],
                       gy - disp[...,1],
                       gx - disp[...,0]])      # (3,Z,Y,X)

    warped = np.empty_like(vol)
    for c in range(29):
        warped[...,c] = ndi.map_coordinates(
            np.transpose(vol[...,c], (2,1,0)),
            coords, order=1, mode='nearest'
        ).transpose(2,1,0)

    # 2 ▸ voxel-wise SH rotation inside ROI
    mask = np.any(disp!=0, axis=-1)
    J    = _jacobian_field(disp)
    dj   = pysh.rotate.djpi2(lmax)

    for (i,j,k) in zip(*np.where(mask)):
        R = scipy.linalg.polar(np.eye(3)+J[i,j,k])[0]       # rotation
        ang = Rotation.from_matrix(R).as_euler('ZYZ', degrees=False)
        cilm   = mrtrix2cilm(warped[i,j,k,1:], lmax)
        cilm_r = pysh.rotate.SHRotateRealCoef(cilm, ang, dj)
        warped[i,j,k,1:] = cilm2mrtrix(cilm_r, lmax)

    # 3 ▸ add Gaussian noise only inside ROI
    if noise_std_range is not None:
        low, high = noise_std_range
        sigma = random.uniform(low, high)
        warped[mask] += sigma * np.random.randn(*warped[mask].shape).astype(np.float32)

    return warped



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
    OUTFILE       = "../scripts/tmp_epoch_output/target_rot_deform.nii.gz"
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

    # ------- SH coeff rotation -----------
    print("Coefficient rotation:")
    sh_rot = sh_rotate_coeff(vol_rot[..., 1:], EULER_DEG, lmax=LMAX)

    # replace rotated SH channels
    vol_rot[..., 1:] = sh_rot

    t0 = time.perf_counter()
    disp, _ = make_patch_displacement(vol_rot.shape[:3],
                                  patch_size=(20,20,20),
                                  spacing=8, warp_scale=2,
                                  mode="both")

    vol_rot = apply_patch_warp_and_reorient(vol_rot, disp, lmax=6)
    t = time.perf_counter() - t0
    print(f"  SH coeff DEFORM   : {t:0.2f} s")

    # ------- save ------------------------
    nib.save(nib.Nifti1Image(vol_rot, np.eye(4)), OUTFILE)
    print(f"Saved → {OUTFILE}")

    #nib.save(nib.Nifti1Image(vol, np.eye(4)), "target_noheader.nii.gz")

