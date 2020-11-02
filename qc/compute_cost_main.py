# Code created by Sudhakar on Oct 2020
# Computing registration cost between aligned images (rigid/affine) with template

import os
import concurrent.futures
from functools import partial
from multiprocessing import cpu_count
import compute_reg_cost_grid as crcg
import gen_test_images_compute_reg_cost_grid as gticrcg


data_dir = '/media/tummala/New Volume/Tummala/Research/ABIDE-validate' # Path to the subjects data directory
subjects = sorted(os.listdir(data_dir)) # Finds subjects in the data directory
print('Found', len(subjects), 'Subjects\n')

#workers = int(os.environ['SLURM_CPUS_PER_TASK']) # Maximum number of COREs requested (in a single node) in SLURM 
workers = cpu_count() # this could be cpu_count() [maximum number of cores available in the machine]
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
    # Compute the cost for correctly aligned images
    crcg.main(data_dir, subject)
    
    # Generate test images for learning ML classifiers
    #gticrcg.main(data_dir, subject)
        
with concurrent.futures.ProcessPoolExecutor(max_workers = workers) as executor:
    executor.map(partial(process_subject, data_dir), subjects)
    # executor.map(process_subject, subjects, data_dir) # This could also be used to avoid 'partial'
      
