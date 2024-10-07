import torch
from ResSR.models import SRmodel
from ResSR.utils import load_volume, save_volume, align_volume_to_ref, myzoom_torch
import numpy as np
import argparse

def main():

    parser = argparse.ArgumentParser(description="Upscaling diffusion weighted images of any resolution to 1.25mm isotropic ", epilog='\n')
    parser.add_argument("--i", help="Image to super-resolve. Tyipcally a 3D or 4D nifti.")
    parser.add_argument("--o", help="Output image. It will also be a 3D / 4D nifti")
    parser.add_argument("--model", help="Model file")
    parser.add_argument("--device", default='cpu', help="device (cpu or cuda)")
    parser.add_argument("--upscaled", default=None, help="linearly upscaled output (for debuggin)")
    parser.add_argument("--frames", type=int, default=100000, help="(optional) Nnumber of frames to process (useful for debugging).")
    args = parser.parse_args()

    # arguments
    device = args.device
    model_file = args.model
    input_file = args.i 
    output_file = args.o
    upscaled_file = args.upscaled
    n_frames = args.frames

    # Constants
    num_filters = 64
    num_residual_blocks = 16
    kernel_size = 3
    use_global_residual = True
    ref_res = 0.0375

    print('Preparing model and loading weights')
    model = SRmodel(num_filters, num_residual_blocks, kernel_size, use_global_residual).to(device)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Loading input volume and normalizing to [0,1]')
    image, aff = load_volume(input_file)
    image = image.astype(float)
    image2, aff2 = align_volume_to_ref(image, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
    if len(image2.shape) == 3:
        image2 = image2[..., np.newaxis]
    nc = image2.shape[3]
    if nc > n_frames:
        nc = n_frames

    image_torch = torch.tensor(image2.copy(), device=device)
    maxis = torch.zeros(nc)
    for c in range(nc):
        maxis[c] = torch.max(image_torch[:, :, :, c])
        image_torch[:, :, :, c] = image_torch[:, :, :, c] / maxis[c]

    print('Upscaling to target resolution')
    voxsize = np.sqrt(np.sum(aff2 ** 2, axis=0))[:-1]
    factors = voxsize / ref_res
    upscaled = myzoom_torch(image_torch, factors, device=device)
    if len(upscaled.shape) == 3:
        upscaled = upscaled[..., np.newaxis]
    aff_upscaled = aff2.copy()
    for j in range(3):
        aff_upscaled[:-1, j] = aff_upscaled[:-1, j] / factors[j]
    aff_upscaled[:-1, -1] = aff_upscaled[:-1, -1] - np.matmul(aff_upscaled[:-1, :-1], 0.5 * (factors - 1))


    print('Pushing data through the CNN')
    with torch.no_grad():
        sr = torch.zeros_like(upscaled)
        for c in range(nc):
            print('   Frame ' + str(1 + c) + ' of ' + str(nc), end="\r")
            pred = model(upscaled[:,:,:,c][None, None, ...])
            sr[:, :, :, c] = maxis[c] * torch.squeeze(pred)
            upscaled[:, :, :, c] = upscaled[:,:,:,c] * maxis[c]  # TODO: possibly disable

    print('\nSaving to disk')
    save_volume(sr[:,:,:,:nc].detach().cpu().numpy(), aff_upscaled, output_file)
    save_volume(upscaled[:,:,:,:nc].detach().cpu().numpy(), aff_upscaled, upscaled_file)

    print('All done')
    print('freeview ' + input_file + ' ' + upscaled_file + ' ' + output_file)


# execute script
if __name__ == '__main__':
    main()

