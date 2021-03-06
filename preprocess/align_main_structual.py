# Code created by Sudhakar on March, modified on June 2020
# Processing for structural data (T1, T2 and FLAIR) aquired at 3T

import os
import concurrent.futures
from functools import partial
from multiprocessing import cpu_count
import nipype_preprocessing_main as npm
import nipype_spm_segmentation as nss
import nipype_freesurfer_processing as nfs

data_dir = "/usr/users/tummala/HCP-YA-Re" # Path to the subjects data directory
subjects = sorted(os.listdir(data_dir)) # Finds subjects in the data directory
print('Found', len(subjects), 'Subjects\n')

#workers = int(os.environ['SLURM_CPUS_PER_TASK']) # Maximum number of COREs requested (in a single node) in SLURM 
workers = 1 # this could be cpu_count() [maximum number of cores available in the machine]
print(f'workers are: {workers}\n')

def process_subject(data_dir, subject):
    '''
    Parameters
    ----------
    data_dir : str
        path to the data directory.
    subject : str
        subject ID.

    Returns
    -------
    all processed files.

    '''
    # pre-processing (cropping, bias-correction followed by rigid, affine transformation to MNI space)
    npm.preprocessing_main(data_dir, subject)
    
    # gray matter, white matter and CSF segmentation using SPM
    nss.do_spm_segmentation(data_dir, subject, image_type = 'anat', multi_channel = True)
        
    # Freesurfer processing (cortical segmentation, hippocampal subfields)
    #nfs.fs_Processing(data_dir, subject, 'anat') # doing it on anat (raw) images
    #nfs.fs_Processing(data_dir, subject, 'align') # doing it on aligned images
        
with concurrent.futures.ProcessPoolExecutor(max_workers = workers) as executor:
    executor.map(partial(process_subject, data_dir), subjects)
    # executor.map(process_subject, subjects, data_dir) # This could also be used to avoid 'partial'
      
