import torch
import sys

sys.path.append('/autofs/space/nicc_003/users/olchanyi/DiffSR')
from ResSR.models_fused import SRmodel
from ResSR.utils import load_volume, save_volume, align_volume_to_ref, myzoom_torch, percentile_scaling, sh_norm
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
    num_filters = 256
    num_residual_blocks = 24
    kernel_size = 3
    use_global_residual = False
    ref_res = 1.25
    n_channels = 28

    print('Preparing model and loading weights')
    #model = SRmodel(num_filters, num_residual_blocks, kernel_size, use_global_residual).to(device)
    model = SRmodel(num_filters=num_filters,
                num_residual_blocks=num_residual_blocks,
                kernel_size=kernel_size,
                use_global_residual=use_global_residual,
                num_filters_l0=64,
                num_residual_blocks_l0=12).to(device)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Loading input volume and normalizing to [0,1]')
    image, aff = load_volume(input_file)
    image = image.astype(float)

    image2, aff2 = align_volume_to_ref(image, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)


    image_torch = torch.tensor(image2.copy(), device=device).float()
    print("performing percentile scaling...")
    image_torch = sh_norm(image_torch, l0_index=0)
    image_torch[torch.isnan(image_torch)] = 0.0
    image_torch[torch.isinf(image_torch)] = 0.0
    image_torch = torch.clamp(image_torch, min=-1, max=1)
    print("done")

    print('Upscaling to target resolution')
    voxsize = np.sqrt(np.sum(aff2 ** 2, axis=0))[:-1]
    print("found voxel size: ", voxsize)

    factors = (voxsize / ref_res)
    upscaled = myzoom_torch(image_torch, factors, device=device)

    aff_upscaled = aff2.copy()
    for j in range(3):
        aff_upscaled[:-1, j] = aff_upscaled[:-1, j] / factors[j]
    aff_upscaled[:-1, -1] = aff_upscaled[:-1, -1] - np.matmul(aff_upscaled[:-1, :-1], 0.5 * (factors - 1))


    print('Pushing data through the CNN')

    upscaled_unpermuted = upscaled.clone()
    upscaled = upscaled.permute(3, 0, 1, 2)

    print("shape: ", upscaled.shape)
    upscaled[0:1,...] = upscaled[0:1,...] * 5

    with torch.no_grad():
        pred = model(upscaled[None, :])

    pred = torch.squeeze(pred)
    pred = pred.permute(1,2,3,0)
    print('\nSaving to disk')
    #print("Mean is: ", np.mean(pred.detach().cpu().numpy()))
    save_volume(pred.detach().cpu().numpy(), aff_upscaled, output_file)
    #save_volume(upscaled_unpermuted.detach().cpu().numpy(), aff_upscaled, upscaled_file)

    print('All done!')


# execute script
if __name__ == '__main__':
    main()

