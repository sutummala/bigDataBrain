# Code created by Sudhakar on Sep 2020
# code to re-arrange files 

import os
import shutil

data_path = 'D:/Tummala/Research/ABIDE'
data_path_new = 'D:/Tummala/Research/ABIDEre'

if not os.path.exists(data_path_new):
    os.makedirs(data_path_new)

subjects = os.listdir(data_path)

for subject in subjects:
    print(f're-arranging for {subject}')
    
    file_path = os.path.join(data_path, subject, 'MP-RAGE', '2000-01-01_00_00_00.0')
    file_path_new = os.path.join(data_path_new, subject)
    
    if not os.path.exists(file_path_new):
        os.makedirs(file_path_new)
    
    folders = os.listdir(file_path)
    
    for folder in folders:
        print(f'found T1-image at {folder}')
        
        final_file_path = os.path.join(file_path, folder)
        files = os.listdir(final_file_path)
        
        for file in files:
            print(f'found {file}')
            
            shutil.copy(os.path.join(final_file_path, file), file_path_new)
            file_new = file.replace(file[12:], 'hrT1.nii')
            os.rename(os.path.join(file_path_new, file), os.path.join(file_path_new, file_new))
            
print('re-arrangement done sucessfully')