Hi Alice,

The main entry point is scripts/test_training.py

1. Take all your high-resolution training files and put them in a directory. It's *crucial* that they all have the same orientation. It's easiest to use a diagonal matrix. You can use utils.align_volume_to_ref for this. For example:

IM, aff = utils.load_volume('whatever.nii.gz')
IM_reoriented, aff_reoriented = utils.align_volume_to_ref(IM, aff, aff_ref=np.eye(4), return_aff=True)

The directory where you put the training files is training_data_dir in the script. 

2. You'll see there's a device for data generation and one for training. If you run out of GPU memory, you can generate with the CPU (even though it's a bit slower).

3. Your models will be written to the directory specified by output_directory

4. If you want to finetune an existing model, you can use the variable initial_model (right now it's set to None, which means "train from scratch")

5. The losses get saved to the checkpoints. If you want to plot them, use plot_loss.py.

6. The single most important parameter is the standard deviation of the simulated noise. Too low and your CNN will be sensitive to noise. Too high and it will just blur the image. I would try several values between 0.01 and 0.10. The default 0.05 is a good start.

7. The other crucial thing: it seems like all your datasets are twice as small (in every dimension) compared with the ground truth. That's why I have set lowres_min = lowres_max = 2.0.  Essentially, the generator downsamples by a factor equal to a random number between lowres_min and lowres_max. So setting both to 2 makes sure your CNN will be optimized to upscale by a factor of 2. 

Once a model is trained, you can use test_inference.py. Two quick comments about it.

1. The resolution of the training data is hardcoded as ref_res = 0.0375. You may want to change that.

2. Right now, the code is written for 4D images. I think that 3D will work just fine (as a special case of 4D with 1 frame), but you may have to fix that if it crashes. It shouldn't be too hard.

Enjoy!

/E


