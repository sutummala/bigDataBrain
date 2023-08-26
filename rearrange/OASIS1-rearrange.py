# Code created by Sudhakar on Aug 2023
# code to re-arrange files 

import os
import shutil
import nibabel as nb

data_path = 'C:/Users/drsuc/Downloads/oasis1'
data_path_new = 'C:/Users/drsuc/Downloads/oasis1Re'

if not os.path.exists(data_path_new):
    os.makedirs(data_path_new)

subjects = os.listdir(data_path)

for subject in subjects:
    print(f're-arranging for {subject}')
    
    file_path = os.path.join(data_path, subject, 'RAW')
    file_path_new = os.path.join(data_path_new, subject, 'anat')
    
    if not os.path.exists(file_path_new):
        os.makedirs(file_path_new)
    
    files = os.listdir(file_path)
    
    # Converting .img files to nifti.
    # for file in files:
    #     if file.endswith('1_anon.img'):
    #         file_path_1 = os.path.join(file_path, file)
    #         img = nb.load(file_path_1)
    #         nb.save(img, file_path_1.replace('.img', '.nii'))
        
    for file in files:
        if file.endswith('.nii'):
            print(f'found {file}')
            shutil.copy(os.path.join(file_path, file), file_path_new)
            file_new = file.replace(file[14:], 'hrT1.nii')
            os.rename(os.path.join(file_path_new, file), os.path.join(file_path_new, file_new))
            
print('re-arrangement done sucessfully')
