import os
import cv2 # open cv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


refpath = "/usr/users/nmri/tools/fsl/6.0.3/data/standard" # FSL template
refbrain_path = refpath+'/MNI152_T1_1mm_brain_mask.nii.gz' # brain MNI 

subpath = '/usr/users/tummala/testBigData'

# key world arguments for differnt plot options
no_of_slices = 3 # choose either 3 or 5
show_plot = True
plot_binary_mask = False
plot_outline = True
mask_alpha = 0.5 # choose value between o and 1, choose lower value for more transparancy
outline_alpha = 0.6 # choose value between o and 1, choose lower value for more transparancy
outline_thickness = 1 # choose an integer
use_all_contours = True # if True it uses all contours, otherwise largest connedted component will be plotted
outline_color = (200, 100, 20) # outline color

def find_contours(mask):
    '''find contours (outline) of given binary mask'''
    mask = np.uint8(np.rot90(mask)) # converting to unit8 for open cv
    kernel = np.ones((5,5),np.uint8) # kernel for closing operation
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # morphological closing operation
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0] # finding contours (interms of a list containing coordinates) 
    return mask, contours

def find_largest_connected_component(contours):
    ''' find the index of largest connected component out of all contours'''
    contour_area = []
    for idx in range(len(contours)):
        contour_area.append(cv2.contourArea(contours[idx]))    
    return contour_area.index(max(contour_area))
    

for subject in os.listdir(subpath):
    print(f'generating image for {subject}')
    required_folder = subpath+'/'+subject+'/mni'
    for file in os.listdir(required_folder):
        if file.endswith('reoriented.mni.nii') and 'hrFLAIR' in file:
            image = nib.load(required_folder+'/'+file) # actual image
            image_data = image.get_fdata()
            x, y, z = image_data.shape
            
            ref_image = nib.load(refbrain_path) # reference image (MNI brain)
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
                        axs[i,j].imshow(np.rot90(image_data[round(weights[j]*x), :, :]), cmap = 'gray') # original image slice
                        if plot_binary_mask:
                            axs[i,j].imshow(np.rot90(ref_data[round(weights[j]*x), :, :]), cmap = 'binary_r', alpha = mask_alpha) # mask slice with transparancy
                        if plot_outline:
                            mask, contours = find_contours(ref_data[round(weights[j]*x), :, :]) # finding countours for slice of the mask
                            if use_all_contours:
                                outline_contour_idx = -1
                            else:
                                outline_contour_idx = find_largest_connected_component(contours) # finding largest connected component of the slice
                            axs[i,j].imshow(cv2.drawContours(mask, contours, outline_contour_idx, outline_color, outline_thickness), cmap = 'binary_r', alpha = outline_alpha) # contour for the correspoding slice
                        axs[i,j].axis('off')
                    elif i == 1: # coronal
                        axs[i,j].imshow(np.rot90(image_data[:, round(weights[j]*y), :]), cmap = 'gray')
                        if plot_binary_mask:
                            axs[i,j].imshow(np.rot90(ref_data[:, round(weights[j]*y), :]), cmap = 'binary_r', alpha = mask_alpha)
                        if plot_outline:
                            mask, contours = find_contours(ref_data[:, round(weights[j]*y), :])
                            if use_all_contours:
                                outline_contour_idx = -1
                            else:
                                outline_contour_idx = find_largest_connected_component(contours)
                            axs[i,j].imshow(cv2.drawContours(mask, contours, outline_contour_idx, outline_color, outline_thickness), cmap = 'binary_r', alpha = outline_alpha)
                        axs[i,j].axis('off')
                    elif i == 2: # axial
                        axs[i,j].imshow(np.rot90(image_data[:, :, round(weights[j]*z)]), cmap = 'gray')
                        if plot_binary_mask:
                            axs[i,j].imshow(np.rot90(ref_data[:, :, round(weights[j]*z)]), cmap = 'binary_r', alpha = mask_alpha)
                        if plot_outline:
                            mask, contours = find_contours(ref_data[:, :, round(weights[j]*z)])
                            if use_all_contours:
                                outline_contour_idx = -1
                            else:
                                outline_contour_idx = find_largest_connected_component(contours)
                            axs[i,j].imshow(cv2.drawContours(mask, contours, outline_contour_idx, outline_color, outline_thickness), cmap = 'binary_r', alpha = outline_alpha)
                        axs[i,j].axis('off')

            fig.set_tight_layout('tight')

            if show_plot:
                plt.show()
            else:
                save_fig = file[0:-4]+'.png'
                plt.savefig(required_folder+'/'+save_fig, facecolor = (0,0,0), edgecolor = (0,0,0))
                plt.close(fig)
