# created by Sudhakar on July 2020
# Quality checking before actual pre-processing starts 
# check for general image quality such as signal to noise ratio, contrast to noise ratio etc.

def check_image_quality(in_image):
    print(f'checking image quality for {in_image}\n')
    
    im_quality_msi = acf.msi(in_image)
    print()







import os
import json
import all_cost_functions as acf

data_dir = '/usr/users/tummala/HCP-YA'

subjects = os.listdir(data_dir)

for subject in subjects:
    print(f'checking image quality for {subject}\n')
    raw_image_path = os.path.join(data_dir, subject, 'anat')
    
    images = os.path.listdir(raw_image_path)
    
    for image in images:
        if image.endswith('hrT1.nii.gz'):
            check_image_quality(image)
        elif image.endswith('hrT2.nii.gz'):
            check_image_quality(image)
        elif image.endswith('hrFLAIR.nii.gz'):
            check_image_quality(image)
        else:
            print(f'no raw image to check for {subject}\n')
                
    
    