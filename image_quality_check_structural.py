# created by Sudhakar on July 2020
# Quality checking before actual pre-processing starts 
# check for general image quality based on several image metrics.

def check_image_quality(*paths):
    
    image_reference = nib.load(os.path.join(paths[0], paths[1]))
    in_image = image_reference.get_fdata()
    
    im_quality_msi = acf.msi(in_image, 'sagittal')
    print(f'MSI for {paths[1]} is {im_quality_msi}\n')
    
    im_quality_snr, im_quality_svnr = acf.snr_and_svnr(*paths)
    print(f'SNR for {paths[1]} is {im_quality_snr}\nSVNR is {im_quality_svnr}\n')
    
    # im_quality_cnr, im_quality_cvnr = acf.cnr_and_cvnr(*paths)
    # print(f'CNR for {paths[1]} is {im_quality_cnr}\nCVNR is {im_quality_cvnr}\n')
    
    # im_quality_cov = acf.cov(in_image)
    # im_quality_fwhm = acf.fwhm(in_image)


import os
import json
import nibabel as nib
import numpy as np
import all_cost_functions as acf

data_dir = '/usr/users/tummala/HCP-YA'

subjects = os.listdir(data_dir)

for subject in subjects:
    print(f'checking image quality for {subject}---------------------------------------------------------------\n')
    raw_image_path = os.path.join(data_dir, subject, 'anat')
    
    images = os.listdir(raw_image_path)
    
    for image in images:
        if any([image.endswith('hrT1.nii.gz'), image.endswith('hrT2.nii.gz'), image.endswith('hrFLAIR.nii.gz')]):
            check_image_quality(raw_image_path, image)
        
                
    
    