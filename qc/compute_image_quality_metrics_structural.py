# created by Sudhakar on July/August 2020
# quality checking before actual pre-processing starts 
# compute different image quality metrics for structural data.

import os
import sys
import json
import nibabel as nib
import all_cost_functions as acf


def compute_image_quality_metrics(*paths):
    
    print(f'computing image quality metrics for {paths[2]}\n')
    
    image_reference = nib.load(os.path.join(paths[0], paths[2]))
    in_image = image_reference.get_fdata()
    
    main, ext = acf.get_file_name_and_extension(os.path.join(paths[0], paths[2]))
    json_file = main+'.image_quality_metrics.json'
    
    re_compute = False # Re-compute all metrics if this is True
    
    if os.path.exists(json_file) and os.path.getsize(json_file) > 0 and not re_compute:
        print(f'image quality metrics are already computed for {paths[2]} and saved at {json_file}\n')
    else:
        im_quality_msi = acf.msi(in_image, 'sagittal')
        print(f'MSI is {im_quality_msi}\n')
        
        im_quality_snr, im_quality_svnr, _ = acf.snr_and_svnr(paths[0], paths[2])
        print(f'SNR is {im_quality_snr}\nSVNR is {im_quality_svnr}\n')
        
        im_quality_cnr, im_quality_cvnr, im_quality_tctv = acf.cnr_and_cvnr_and_tctv(paths[0], paths[2])
        print(f'CNR is {im_quality_cnr}\nCVNR is {im_quality_cvnr}\nTCTV is {im_quality_tctv}\n')
        
        im_quality_fwhm = acf.fwhm(paths[0], paths[2])
        print(f'FWHM for {paths[1]} is {im_quality_fwhm}\n')
        
        im_quality_ent = acf.ent(paths[0], paths[2])
        print(f'Entropy is {im_quality_ent}\n')
        
        save_image_quality_metrics = {'subject_ID': paths[1], 'image_file': paths[2], 'MSI': im_quality_msi, 'SNR': im_quality_snr, 'SVNR': im_quality_svnr, 'CNR': im_quality_cnr, 'CVNR': im_quality_cvnr, 'TCTV': im_quality_tctv, 'FWHM': im_quality_fwhm, 'ENT': im_quality_ent}
        
        with open(json_file, 'w') as file:
            json.dump(save_image_quality_metrics, file, indent = 4)
            
        print(f'image quality metrics were computed for {paths[2]} and saved at {json_file}\n')

def main(data_dir, subject):
    '''
    Parameters
    ----------
    data_dir : str
        path to the data directory.
    subject : str
        subject ID.

    Returns
    -------
    a json with image_quality_metrics.

    '''
    raw_image_path = os.path.join(data_dir, subject, 'anat')
    
    if os.path.exists(raw_image_path):
        images = os.listdir(raw_image_path)
        for image in images:
            if any([image.endswith('hrT1.nii.gz'), image.endswith('hrT1.M.nii.gz')]): # hrT1
                compute_image_quality_metrics(raw_image_path, subject, image)
            elif any([image.endswith('hrT2.nii.gz'), image.endswith('hrT2.M.nii.gz')]): # hrT2
                compute_image_quality_metrics(raw_image_path, subject, image)
            elif any([image.endswith('hrFLAIR.nii.gz'), image.endswith('hrFLAIR.M.nii.gz')]): # hrFLAIR
                compute_image_quality_metrics(raw_image_path, subject, image)
            elif any([image.endswith('hrPD.nii.gz'), image.endswith('hrPD.M.nii.gz')]): # hrPD
                compute_image_quality_metrics(raw_image_path, subject, image)
                    
cluster = False
    
if cluster:
    data_dir = sys.argv[1]
    subject = sys.argv[2]
    main(data_dir, subject)
else:
    data_dir = '/usr/users/tummala/bigdata1'
    subjects = sorted(os.listdir(data_dir))

    for subject in subjects:
        main(data_dir, subject)
        
       
    
                
    
    
