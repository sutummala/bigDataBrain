# created by Sudhakar on June 2020
# verify the registration quality for each subject based on json files at reg_check folder

import os
import json
import nipype_all_functions as naf
import plot_slices as ps

data_path = '/usr/users/tummala/HCP-YA-Re'

merge_json = False
save_or_show_image = True
all_counter = 0
success_counter = 0
epsilon = 0.009
subjects_counter = 0

subjects = sorted(os.listdir(data_path))

for subject in subjects:
    #print(f'checking for {subject}\n')
    
    reg_path = os.path.join(data_path, subject, 'reg_check')
    fig_path = os.path.join(data_path, subject, 'figs')
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    if os.path.exists(reg_path):
        subjects_counter += 1
        for json_file in os.listdir(reg_path):
            all_counter += 1
            manual_check = False
            #print(f'checking for {json_file}\n')
            if os.path.getsize(reg_path+'/'+json_file) > 0: # making sure that the file is not empty
                with open(reg_path+'/'+json_file) as in_json:
                    data = json.load(in_json)
                    
                if data['file_name'].find('align.') != -1:
                    required_path = os.path.join(data_path, subject, 'align')
                elif data['file_name'].find('mni') != -1:
                    required_path = os.path.join(data_path, subject, 'mni')
                elif data['file_name'].find('alignedToT1') != -1:
                    required_path = os.path.join(data_path, subject, 'anat')
                
                # checking the registration flag
                if data['reg_flag']:
                    success_counter += 1
                    #print('registration is fine for:', data['file_name'], '\n')
                elif data['cost_actual'] > data['cost_threshold']-epsilon:
                    success_counter += 1
                    data['reg_flag'] = True # changing the flag to True
                elif data['cost_actual'] <= data['cost_threshold']-epsilon:
                    print('manual checking required for:', data['file_name'], '\n')
                    image_name = required_path+'/'+data['file_name']
                    manual_check = True
                    #os.system(f'fsleyes {image_name}')
                elif data['cost_actual'] <= data['cost_threshold_critical']:
                    print('image may not be aligned correctly may not be suitable for further processing:', data['file_name'], '\n')
                
                # write back the updated reg_flag to the corresponding json file
                with open(reg_path+'/'+json_file, 'w') as out_json:
                    json.dump(data, out_json, indent = 4)
                    
                # save the required image along with its brain outline in png format
                if save_or_show_image and manual_check:
                    ps.plot_image_in_slices(required_path, fig_path, data['file_name'], no_of_slices = 3, show_plot = True, plot_binary_mask = True, plot_outline = True, mask_alpha = 0.1,
                                        outline_alpha = 0.6, outline_thickness = 2, use_all_contours = True, outline_color = (200, 100, 20))
                       
    # if merging json is requested    
    if merge_json:
        naf.do_json_combine(reg_path, subject, remove_individual_json = False)

print(f'checked for {subjects_counter} subjects, total tentatively failed registrations are: {all_counter - success_counter} out of {all_counter}\n')

successful_registrations = (success_counter/all_counter) * 100
print(f'percentage of success is {successful_registrations}\n')    

        
                    