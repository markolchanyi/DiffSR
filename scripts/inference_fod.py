import torch, contextlib
import os,sys
import math


sys.path.append('/autofs/space/nicc_003/users/olchanyi/DiffSR_testing')

from ResSR.models import S2UNetGlobalL2
from ResSR.utils import load_volume, save_volume, align_volume_to_ref, myzoom_torch, myzoom_torch_better, percentile_scaling, sh_norm, pad_to_stride_torch, unpad_tensor
import numpy as np
import argparse

# if too big for gpu just do patch-wise (update loop below pls)
def run_in_patches(vol, model, device, patch_size=64, overlap=16):
    C, X, Y, Z = vol.shape
    out = torch.zeros_like(vol)
    weight = torch.zeros(1, X, Y, Z, device=device)

    step = patch_size - overlap
    for x0 in range(0, X, step):
        x1 = min(x0 + patch_size, X)
        for y0 in range(0, Y, step):
            y1 = min(y0 + patch_size, Y)
            for z0 in range(0, Z, step):
                z1 = min(z0 + patch_size, Z)

                patch = vol[:, x0:x1, y0:y1, z0:z1].unsqueeze(0)
                with torch.no_grad():
                    pred_patch = model(patch)[0]

                out[:, x0:x1, y0:y1, z0:z1] += pred_patch
                weight[:, x0:x1, y0:y1, z0:z1] += 1.0

    out = out / weight
    return out


# sliding window
def sliding_window_inference(
    vol,
    model,
    device,
    patch_size=(64, 64, 64),
    overlap=(16, 16, 16),
):

    assert vol.dim() == 4, f"Expected (C,X,Y,Z), got {vol.shape}"
    C, X, Y, Z = vol.shape
    dx, dy, dz = patch_size
    ox, oy, oz = overlap

    sx = dx - ox
    sy = dy - oy
    sz = dz - oz
    if sx <= 0 or sy <= 0 or sz <= 0:
        raise ValueError(f"overlap must be smaller than patch_size; got patch={patch_size}, overlap={overlap}")

    # cosine blend
    def one_dim(n):
        t = torch.linspace(0, math.pi, steps=n, device=device)
        w = 0.5 * (1.0 - torch.cos(t))
        return w

    wx = one_dim(dx)
    wy = one_dim(dy)
    wz = one_dim(dz)
    w3d = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    w3d = w3d / w3d.max() #normalize
    w_patch = w3d.view(1, 1, dx, dy, dz)

    out = torch.zeros((C, X, Y, Z), device=device, dtype=vol.dtype)
    weight = torch.zeros((1, X, Y, Z), device=device, dtype=vol.dtype)

    for x0 in range(0, max(X - dx + 1, 1), sx):
        for y0 in range(0, max(Y - dy + 1, 1), sy):
            for z0 in range(0, max(Z - dz + 1, 1), sz):
                x1 = x0 + dx
                y1 = y0 + dy
                z1 = z0 + dz

                if x1 > X:
                    x0 = max(X - dx, 0)
                    x1 = x0 + dx
                if y1 > Y:
                    y0 = max(Y - dy, 0)
                    y1 = y0 + dy
                if z1 > Z:
                    z0 = max(Z - dz, 0)
                    z1 = z0 + dz

                patch = vol[:, x0:x1, y0:y1, z0:z1].unsqueeze(0)
                patch = torch.nan_to_num(patch)

                with torch.no_grad():
                    pred_patch = model(patch)

                pred_patch = pred_patch * w_patch  # apply window

                out[:, x0:x1, y0:y1, z0:z1] += pred_patch.squeeze(0)
                weight[:, x0:x1, y0:y1, z0:z1] += w_patch.squeeze(0)

    weight = torch.clamp(weight, min=1e-6)
    out = out / weight 

    return out



def main():

    parser = argparse.ArgumentParser(description="Upscaling diffusion weighted images of any resolution to 1.25mm isotropic ", epilog='\n')
    parser.add_argument("--i", help="Image to super-resolve. Tyipcally a 3D or 4D nifti.")
    parser.add_argument("--o", help="Output image. It will also be a 3D / 4D nifti")
    parser.add_argument("--model", help="Model file")
    parser.add_argument("--device", default='cpu', help="device (cpu or cuda)")
    parser.add_argument("--upscaled", default=None, help="linearly upscaled output (for debuggin)")
    parser.add_argument("--frames", type=int, default=100000, help="(optional) Nnumber of frames to process (useful for debugging).")

    parser.add_argument("--bzeroscale", type=float, default=1, help="b0 scale")
    parser.add_argument("--bzerodrift", type=float, default=0, help="b0 drift")
    parser.add_argument("--lzeroscale", type=float, default=1, help="b0 scale")
    parser.add_argument("--lzerodrift", type=float, default=0, help="b0 scale")
    parser.add_argument("--lhighscale", type=float, default=1, help="b0 scale")
    parser.add_argument("--lhighdrift", type=float, default=0, help="b0 scale")

    #drifts are there because in our ULF volumes, sometimes normalization (even with IQR) sucks. just keep at 1's and 0's


    args = parser.parse_args()

    # arguments
    device = args.device
    #device = "cuda:0"
    #device = "cpu"
    model_file = args.model
    input_file = args.i
    output_file = args.o
    upscaled_file = args.upscaled
    n_frames = args.frames

    # Constants
    ref_res = 1.3 # HCP-ish
    #n_channels = 28

    print('Preparing model and loading weights')

    model = S2UNetGlobalL2().to(device)


    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print('Loading input volume and normalizing to [0,1]')
    image, aff = load_volume(input_file)
    image = image.astype(float)

    print("input volume shape is: ", image.shape)

    #image2, aff2 = align_volume_to_ref(image, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
    image2, aff2 = image, aff

    image_torch = torch.tensor(image2.copy(), device=device).float()
    print("performing percentile scaling...")
    image_torch = sh_norm(image_torch, l0_index=1)
    image_torch[torch.isnan(image_torch)] = 0.0
    image_torch[torch.isinf(image_torch)] = 0.0
    #image_torch = torch.clamp(image_torch, min=-1, max=1)
    print("done")

    print('Upscaling to target resolution')
    voxsize = np.sqrt(np.sum(aff2 ** 2, axis=0))[:-1]
    print("found voxel size: ", voxsize)

    factors = (voxsize / ref_res)
    upscaled = myzoom_torch_better(image_torch, factors, device=device)
    #upscaled = image_torch

    aff_upscaled = aff2.copy()
    for j in range(3):
        aff_upscaled[:-1, j] = aff_upscaled[:-1, j] / factors[j]
    aff_upscaled[:-1, -1] = aff_upscaled[:-1, -1] - np.matmul(aff_upscaled[:-1, :-1], 0.5 * (factors - 1))


    print('Pushing data through the CNN')

    #upscaled_unpermuted = upscaled.clone()
    upscaled = upscaled.permute(3, 0, 1, 2)

    ## just in case
    upscaled = torch.nan_to_num(upscaled)
    neg_mask_b0 = upscaled[0, ...] < 0
    neg_mask_l0 = upscaled[1, ...] < 0
    upscaled[0,...][neg_mask_b0] = 0
    upscaled[1,...][neg_mask_l0] = 0

    upscaled[0,...] = (upscaled[0,...] + args.bzerodrift) * args.bzeroscale
    upscaled[1,...] = (upscaled[1,...] + args.lzerodrift) * args.lzeroscale
    upscaled[2:,...] = (upscaled[2:,...] + args.lhighdrift) * args.lhighscale

    neg_mask_b0 = upscaled[0, ...] < 0
    neg_mask_l0 = upscaled[1, ...] < 0
    upscaled[0,...][neg_mask_b0] = 0
    upscaled[1,...][neg_mask_l0] = 0


    #upscaled = torch.clamp(upscaled, min=-1, max=1)

    upscaled, pads = pad_to_stride_torch(upscaled, stride=16)

    upscaled = torch.nan_to_num(upscaled)

    with torch.no_grad():
        #if not device == "cpu":
            #with torch.autocast(dtype=torch.float16, device_type=args.device):
        #    pred = model(upscaled[None, :])
        #else:
        #    print("running without autocast")
            #pred = model(upscaled[None, :])

        #pred = run_in_patches(upscaled, model, device=device)
        pred = model(upscaled[None, :])[0]
    #pred = torch.squeeze(pred)

    pred = unpad_tensor(pred, pads)
    upscaled = unpad_tensor(upscaled, pads)

    #print(" ")
    #print("-------- MEAN INTENSITY IS: ", torch.mean(pred), "  ------------")
    #print(" ")

    pred = pred.permute(1,2,3,0)
    upscaled = upscaled.permute(1,2,3,0)
    print('\nSaving to disk')
    #print("Mean is: ", np.mean(pred.detach().cpu().numpy()))
    save_volume(pred.detach().cpu().numpy(), aff_upscaled, output_file)

    root, ext = os.path.splitext(output_file)
    root, ext = os.path.splitext(root)
    save_volume(upscaled.detach().cpu().numpy(), aff_upscaled, root + "_upscaled.nii.gz")

    print('All done!')

##################################################
if __name__ == '__main__':
    main()

