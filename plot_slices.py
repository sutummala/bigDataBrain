# created by Sudhakar on June 2020
# plots given 3D image into slices in each direction, with an outline of the brain 

import os
import cv2 # open cv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def find_contours(mask):
    '''find contours (outline) of given binary mask'''
    mask = np.uint8(np.rot90(mask)) # converting to unit8 for open cv
    kernel = np.ones((5,5),np.uint8) # kernel for morphological closing operation
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # morphological closing operation
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0] # finding contours (interms of a list containing coordinates) 
    return mask, contours

def find_largest_connected_component(contours):
    ''' find the index of largest connected component out of all contours'''
    contour_area = []
    for idx in range(len(contours)):
        contour_area.append(cv2.contourArea(contours[idx]))    
    return contour_area.index(max(contour_area))

# # key world arguments for differnt plot options
# no_of_slices = 3 # choose either 3 or 5
# show_plot = True
# plot_binary_mask = False
# plot_outline = True
# mask_alpha = 0.5 # choose value between o and 1, choose lower value for more transparancy
# outline_alpha = 0.6 # choose value between o and 1, choose lower value for more transparancy
# outline_thickness = 1 # choose an integer
# use_all_contours = True # if True it uses all contours, otherwise largest connedted component will be plotted
# outline_color = (200, 100, 20) # outline color   

def plot_image_in_slices(required_folder, fig_path, file, no_of_slices, show_plot, plot_binary_mask, plot_outline, mask_alpha, outline_alpha, outline_thickness, use_all_contours, outline_color):
    '''
    Parameters
    ----------
    required_folder : str
        folder containing the actual data.
    fig_path : str
        path to save the generated png figure
    file : str
        image file name.
    no_of_slices : int
        number of slices to plot in each direction, choose either 3 or 5.
    show_plot : boolean
        if True, plot will be displayed.
    plot_binary_mask : boolean
        if True, binary mask will be used for plotting.
    plot_outline : boolean
        if True, an outline will be plotted.
    mask_alpha : float
        choose value between o and 1, choose lower value for more transparancy
    outline_alpha : float
        choose value between o and 1, choose lower value for more transparancy
    outline_thickness : int
        thickness of the outline, default is 1, higher the number, more thickness
    use_all_contours : boolean
        if True it uses all contours, otherwise largest connedted component will be plotted
    outline_color : tuple
        color of the outline given as a tuple.

    Returns
    -------
    a png image of file with number of slices plotted as requested

    '''
    refpath = "/usr/users/nmri/tools/fsl/6.0.3/data/standard" # FSL template
    refbrain_path = refpath+'/MNI152_T1_1mm_brain_mask.nii.gz' # brain mask MNI 

    image = nib.load(required_folder+'/'+file) # actual image
    image_data = image.get_fdata()
    x, y, z = image_data.shape
    
    if file.find('alignedToT1') == -1:            
        ref_image = nib.load(refbrain_path) # reference image (MNI brain mask)
    else:
        for t1_brain_mask in os.lisdir(required_folder):
            if t1_brain_mask.find('nu_corr.brain.mask') != -1 and 'hrT1' in t1_brain_mask:
                refbrain_path = required_folder+'/'+t1_brain_mask
                ref_image = nib.load(refbrain_path) # reference image is corresponding T1 brain mask
                
    ref_data = ref_image.get_fdata()
    
    if no_of_slices == 3:
        weights = [0.25, 0.5, 0.75]
    elif no_of_slices == 5:
        weights = [0.20, 0.35, 0.5, 0.65, 0.80]
    else:
        print('choose no.of slices either 3 or 5')
    
    fig, axs = plt.subplots(3, no_of_slices, facecolor = (0, 0, 0), gridspec_kw = {'wspace':0, 'hspace':0})
    
    for i in range(3): # three directions
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
        plt.show(block = False)
        plt.pause(5)
        plt.close()
    else:
        save_fig = file[:-4]+'.png'
        plt.savefig(fig_path+'/'+save_fig, facecolor = (0,0,0), edgecolor = (0,0,0))
        print(f'figure saved {save_fig}\n')
        plt.close(fig)           
