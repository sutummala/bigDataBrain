# submitting jobs to the cluster

import os
import sys

data_dir = '/usr/users/tummala/HCP-YA'

for subject in os.listdir(data_dir):
    ''' submits jobs to the cluster, one for each subject'''

    cmd = sys.executable
    cmd += " "+"/usr/users/tummala/python/gen_test_images_compute_reg_cost_grid.py"
    cmd += f' {data_dir} {subject}'
    
    os.system(f'nmri_qsub -runtime 2880 -title BIG-DATA-test-{subject} {cmd}')
        

      
