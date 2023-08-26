# Code created by Sudhakar on Aug 2023
# code to re-arrange files 

import os
import shutil

data_path = 'C:/Users/drsuc/Downloads/foodStudy'
data_path_new = 'C:/Users/drsuc/Downloads/foodStudyRe'

if not os.path.exists(data_path_new):
    os.makedirs(data_path_new)

subjects = os.listdir(data_path)

for subject in subjects:
    print(f're-arranging for {subject}')
    
    file_path = os.path.join(data_path, subject, 'ses-1', 'anat')
    file_path_new = os.path.join(data_path_new, subject, 'anat')
    
    if not os.path.exists(file_path_new):
        os.makedirs(file_path_new)
    
    files = os.listdir(file_path)
    
        
    for file in files:
        print(f'found {file}')
            
        shutil.copy(os.path.join(file_path, file), file_path_new)
        file_new = file.replace('T1w', 'hrT1')
        os.rename(os.path.join(file_path_new, file), os.path.join(file_path_new, file_new))
            
print('re-arrangement done sucessfully')
