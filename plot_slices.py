import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


refpath = "/usr/users/nmri/tools/fsl/6.0.3/data/standard" # FSL template
refbrain = refpath+'/MNI152_T1_1mm_brain_mask.nii.gz' # brain MNI 

subpath = '/usr/users/tummala/testBigData'
no_of_slices = 5
show_plot = True

for subject in os.listdir(subpath):
    print(f'generating image for {subject}')
    required_folder = subpath+'/'+subject+'/mni'
    for file in os.listdir(required_folder):
        if file.endswith('reoriented.mni.nii') and 'hrFLAIR' in file:
            image = nib.load(required_folder+'/'+file) # actual image
            image_data = image.get_fdata()
            x, y, z = image_data.shape
            
            ref_image = nib.load(refbrain) # reference image (MNI brain)
            ref_data = ref_image.get_fdata()

            if no_of_slices == 3:
                weights = [0.25, 0.5, 0.75]
            elif no_of_slices == 5:
                weights = [0.20, 0.35, 0.5, 0.65, 0.80]
            else:
                print('choose no.of slices either 3 or 5')

            fig, axs = plt.subplots(3, no_of_slices, facecolor = (0, 0, 0), gridspec_kw = {'wspace':0, 'hspace':0})

            for i in range(3):
                for j in range(no_of_slices):
                    if i == 0: # sagittal
                        axs[i,j].imshow(np.rot90(image_data[round(weights[j]*x), :, :]), cmap = 'gray')
                        axs[i,j].imshow(np.rot90(ref_data[round(weights[j]*x), :, :]), cmap = 'binary_r', alpha = 0.6)
                        axs[i,j].axis('off')
                    elif i == 1: # coronal
                        axs[i,j].imshow(np.rot90(image_data[:, round(weights[j]*y), :]), cmap = 'gray')
                        axs[i,j].imshow(np.rot90(ref_data[:, round(weights[j]*y), :]), cmap = 'binary_r', alpha = 0.6)
                        axs[i,j].axis('off')
                    elif i == 2: # axial
                        axs[i,j].imshow(np.rot90(image_data[:, :, round(weights[j]*z)]), cmap = 'gray')
                        axs[i,j].imshow(np.rot90(ref_data[:, :, round(weights[j]*z)]), cmap = 'binary_r', alpha = 0.6)
                        axs[i,j].axis('off')

            fig.set_tight_layout('tight')

            if show_plot:
                plt.show()
            else:
                save_fig = file[0:-4]+'.png'
                plt.savefig(required_folder+'/'+save_fig, facecolor = (0,0,0), edgecolor = (0,0,0))
                plt.close(fig)
