# Code created by Sudhakar on Oct 2020
# Computing registration cost between aligned images (rigid/affine) with template

import os
import concurrent.futures
from functools import partial
from multiprocessing import cpu_count
import compute_reg_cost_grid as crcg
import gen_test_images_compute_reg_cost_grid as gticrcg


#data_dir = '/media/tummala/TUMMALA/Work/Data/IXI-Re' # Path to the subjects data directory
#data_dir = '/media/tummala/SeagateBackupPlusDrive/Project/IXI-Re'
#data_dir = '/home/tummala/data/HCP-100re'
data_dir = '/media/tummala/TUMMALA/Work/Data/HCP-YAre'
subjects = sorted(os.listdir(data_dir)) # Finds subjects in the data directory
print('Found', len(subjects), 'Subjects\n')

#workers = int(os.environ['SLURM_CPUS_PER_TASK']) # Maximum number of COREs requested (in a single node) in SLURM 
workers = cpu_count() # this could be cpu_count() [maximum number of cores available in the machine]
print(f'workers are: {workers}\n')
voi_size = 9
stride = voi_size
no_of_test_images = 5


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
    crcg.main(data_dir, subject, voi_size, stride) # last variable is stride and last but one is voi_size
    
    # Generate test images and compute the cost for learning different supervised ML classifiers
    gticrcg.main(data_dir, subject, voi_size, stride, no_of_test_images) # last variable is number of test images

if __name__ == '__main__':        
    with concurrent.futures.ProcessPoolExecutor(max_workers = workers) as executor:
        executor.map(partial(process_subject, data_dir), subjects)
        #executor.map(process_subject, subjects, data_dir) # This could also be used to avoid 'partial'
      
