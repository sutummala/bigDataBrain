# submitting jobs to the cluster

import os
import sys

data_dir = '/usr/users/tummala/HCP-YA'

for subject in os.listdir(data_dir):
    ''' submits jobs to the cluster, one for each subject'''

    cmd = sys.executable
    cmd += " "+"/usr/users/tummala/python/image_quality_check_structural.py"
    cmd += f' {data_dir} {subject}'
    
    os.system(f'nmri_qsub -cpus 4 -runtime 300 -title {subject} {cmd}')
        

      
