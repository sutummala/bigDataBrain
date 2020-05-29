
import sys
import os
import numpy as np
import registration_cost_function as rcf

cluster = True

if cluster:
    data_dir = sys.argv[1] # Path to the subjects data directory
    subject = sys.argv[2] # Subject ID
else:
    data_dir = '/usr/users/tummala/bigdata' # Path to the subjects data directory
    subject = os.listdir(data_dir)[1] # Subject ID


refpath = "/usr/users/nmri/tools/fsl/6.0.3/data/standard" # FSL template
  
def compute_cost_vectors(data_dir, subject, reg_type, cost_func, tag):
    
    print(f'doing for {reg_type} for cost {cost_func}\n')
    if reg_type == 'align':
        required_folder = data_dir+'/'+subject+'/align'
        checking_tag = 'reoriented.align.nii'
    elif reg_type == 'affine':
        required_folder = data_dir+'/'+subject+'/mni'
        checking_tag = 'reoriented.mni.nii'
        
    cost_folder = data_dir+'/'+subject+'/cost'
    if not os.path.exists(cost_folder):
        os.makedirs(cost_folder)
    movingfiles = os.listdir(required_folder)
    for movingfile in movingfiles:
        if movingfile.endswith(checking_tag) and tag in movingfile:
            print(f'{subject}, checking file: {movingfile}')
            global_cost, local_cost = rcf.do_check_registration(refpath, required_folder+'/'+movingfile, cost_func, True, True, True)
            cost_file = movingfile[0:-4]+f'.{cost_func}.data'       
            np.savetxt(cost_folder+'/'+cost_file, [global_cost, local_cost], fmt = '%1.3f')
            
def compute_coreg_cost_vectors(data_dir, subject, cost_func, tag):
    
    #print(f'doing cost estimation for registering {tag} brain to hrT1 brain\n')
    required_folder = data_dir+'/'+subject+'/anat'
    checking_tag_ref = 'nu_corr.brain.nii'
    checking_tag_moving = 'alignedToT1.nii'
    
    ref_file = False
    moving_file = False
    
    cost_folder = data_dir+'/'+subject+'/cost'
    if not os.path.exists(cost_folder):
        os.makedirs(cost_folder)
    movingfiles = os.listdir(required_folder)
    for movingfile in movingfiles:
        if movingfile.endswith(checking_tag_ref) and 'hrT1' in movingfile:
            movingfile_ref = movingfile
            ref_file = True
        elif movingfile.endswith(checking_tag_moving) and tag in movingfile:
            movingfile_moving = movingfile
            moving_file = True
    if ref_file and moving_file:
        print(f'{subject}, checking files: {movingfile_ref, movingfile_moving}')
        global_cost, local_cost = rcf.do_check_coregistration(required_folder+'/'+movingfile_ref, required_folder+'/'+movingfile_moving, cost_func, True, True, True)
        cost_file = movingfile_moving[0:-4]+f'.{cost_func}.data'       
        np.savetxt(cost_folder+'/'+cost_file, [global_cost, local_cost], fmt = '%1.3f')

image_types = ['hrT1', 'hrT2', 'hrFLAIR']
costs = ['ncc', 'nmi']
reg_types = ['align', 'affine']

for image_type in image_types:
        for cost in costs:
            # computing cost for co-registration of T2/FLAIR brain to T1 brain
            compute_coreg_cost_vectors(data_dir, subject, cost, image_type)
            for reg_type in reg_types:
            # dealing with hrT1, hrT2 and hrFLAIR for bigdata and HCPYA, rigid and affine registration
                pass
                #compute_cost_vectors(data_dir, subject, reg_type, cost, image_type)
            
print('done computation\n')


    
    