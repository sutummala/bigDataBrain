# Code created by Sudhakar on Aug 2023
# code to re-arrange files from OASIS4 dataset

import os
import shutil
import json

data_path = 'C:/Users/drsuc/Downloads/oasis4'
data_path_new = 'C:/Users/drsuc/Downloads/oasis4Re'

if not os.path.exists(data_path_new):
    os.makedirs(data_path_new)

subjects = os.listdir(data_path)

for subject in subjects:
    print(f're-arranging for {subject}')
    
    folders_path = os.path.join(data_path, subject)
    file_path_new = os.path.join(data_path_new, subject, 'anat')
    
    if not os.path.exists(file_path_new):
        os.makedirs(file_path_new)
    
    folders = os.listdir(folders_path)
    T1 = 0; T2 = 0
    for folder in folders:
        if folder.startswith('anat'):
            sub_folder_path = os.path.join(folders_path, folder)
            for sub_folder in os.listdir(sub_folder_path):
                if sub_folder.startswith('B'):
                    for json_file in os.listdir(os.path.join(sub_folder_path, sub_folder)):
                        json_data = open(os.path.join(sub_folder_path, sub_folder, json_file))
                        data = json.load(json_data)
                        series_des = data['SeriesDescription']
                        if series_des.find('MPRAGE') != -1:
                            print(f'found MPRAGE at {folder} for {subject}')
                            T1 = T1 + 1
                            if T1 <= 1:
                                image_path = os.path.join(sub_folder_path, 'NIFTI')
                                for image in os.listdir(image_path):
                                    print(image)
                                    shutil.copy(os.path.join(image_path, image), file_path_new)
                                    file_new = image.replace('T1w', 'hrT1')
                                    os.rename(os.path.join(file_path_new, image), os.path.join(file_path_new, file_new))
                        elif series_des.find('TRA_T2') != -1:
                            print(f'found T2 at {folder} for {subject}')
                            T2 = T2 + 1
                            if T2 <= 1:
                                image_path = os.path.join(sub_folder_path, 'NIFTI')
                                for image in os.listdir(image_path):
                                    print(image)
                                    shutil.copy(os.path.join(image_path, image), file_path_new)
                                    file_new = image.replace('T2w', 'hrT2')
                                    os.rename(os.path.join(file_path_new, image), os.path.join(file_path_new, file_new))
        
            
print('re-arrangement done sucessfully')
