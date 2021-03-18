# created by Sudhakar on June 2020
# verify the registration quality for each subject based on json files at reg_check folder

import os
import json
import nipype_all_functions as naf
import plot_slices as ps

data_path = '/media/tummala/TUMMALA/Work/Data/IXI-Re' 

save_or_show_image = True
all_counter = 0
success_counter = 0
subjects_counter = 0
manual_check = True

image_flag = 'align'
image_tag = 'hrT1'

subjects = sorted(os.listdir(data_path))

for subject in subjects:
    #print(f'checking for {subject}\n')
    
    required_image = False
    
    fig_path = os.path.join(data_path, subject, 'figs')
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
                            
    if image_flag == 'align':
        required_path = os.path.join(data_path, subject, 'align')
    elif image_flag == 'mni':
        required_path = os.path.join(data_path, subject, 'mni')
    elif image_flag == 'anat':
        required_path = os.path.join(data_path, subject, 'anat')
    
    for required_file in os.listdir(required_path):
        if image_tag in required_file:
            required_image = True
            if required_file.endswith('reoriented.align.nii') or required_file.endswith('reoriented.mni.nii') or required_file.endswith('alignedToT1.nii'):
                image_to_plot = required_file            
                
    # save the required image along with its brain outline in png format or display
    if save_or_show_image and manual_check and required_image:
        ps.plot_image_in_slices(required_path, fig_path, image_to_plot, no_of_slices = 3, show_plot = True, plot_binary_mask = True, plot_outline = True, mask_alpha = 0.1,
                    outline_alpha = 0.6, outline_thickness = 2, use_all_contours = True, outline_color = (200, 100, 20), pause_time = 5)
                           
        
                    
