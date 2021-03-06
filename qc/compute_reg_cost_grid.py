# code created by Sudhakar on May 2020
# compute registration cost

import sys
import os
import numpy as np
import registration_cost_function as rcf

# data_dir = sys.argv[1] # Path to the subjects data directory
# subject = sys.argv[2] # Subject ID

refpath = "/home/tummala/mri/tools/fsl/data/standard" # FSL template
  
def compute_cost_vectors(data_dir, subject, reg_type, cost_func, tag, voi_size, step_size):
    '''
    Parameters
    ----------
    data_dir : str
        path to the data directory.
    subject : str
        subject id.
    reg_type : str
        registration type, align for rigid and mni for affine.
    cost_func : str
        cost function, ncc: normalized correlation coefficient, nmi: normalized mutual information.
    tag : str
        image tag.
    voi_size : int
        size of the volume of interest, e.g. 3 or 5 or 7 etc.
    step_size : int
        size of the step size in sliding the VOI over the image, ideally it may be same as voi_size, but that is not the strict requirement.

    Returns
    -------
    returns global and local cost vectors 
    '''
    
    recompute =  False # flag if you want to recompute
    
    if reg_type == 'align':
        required_folder = data_dir+'/'+subject+'/align'
        checking_tag = 'reoriented.align.nii'
    elif reg_type == 'affine':
        required_folder = data_dir+'/'+subject+'/mni'
        checking_tag = 'reoriented.mni.nii'
        
    cost_folder = data_dir+'/'+subject+'/cost'+str(voi_size)+str(step_size)
    if not os.path.exists(cost_folder):
        os.makedirs(cost_folder)
    movingfiles = os.listdir(required_folder)
    for movingfile in movingfiles:
        if movingfile.endswith(checking_tag) and tag in movingfile:
            print(f'{subject}, checking file: {movingfile}')
            cost_file = movingfile[0:-4]+f'.{cost_func}.data'
            
            if not recompute and os.path.exists(cost_folder+'/'+cost_file) and os.path.getsize(cost_folder+'/'+cost_file) > 0:
                print(f'cost values were already computed at {cost_file}\n')
            else:
                global_cost, local_cost = rcf.do_check_registration(refpath, required_folder+'/'+movingfile, cost_func, voi_size, step_size, masking = True, measure_global = True, measure_local = True)
                np.savetxt(cost_folder+'/'+cost_file, [global_cost, local_cost])
            
def compute_coreg_cost_vectors(data_dir, subject, cost_func, tag, voi_size, step_size):
    '''
    Parameters
    ----------
    data_dir : str
        path to the data directory.
    subject : str
        subject id.
    cost_func : str
        cost function, ncc: normalized correlation coefficient, nmi: normalized mutual information.
    tag : str
        image tag.
    voi_size : int
        size of the volume of interest, e.g. 3 or 5 or 7 etc.
    step_size : int
        size of the step size in sliding the VOI over the image, ideally it may be same as voi_size, but that is not the strict requirement.

    Returns
    -------
    returns global and local cost vectors for coregistration of T2/FLAIR to corresponding T1
    '''
    recompute =  False # flag if you want to recompute
    
    #print(f'doing cost estimation for registering {tag} brain to hrT1 brain\n')
    required_folder = data_dir+'/'+subject+'/anat'
    checking_tag_ref = 'nu_corr.brain.nii'
    checking_tag_moving = 'alignedToT1.nii'
    
    ref_file = False
    moving_file = False
    
    cost_folder = data_dir+'/'+subject+'/cost'+str(voi_size)+str(step_size)
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
        print(f'{subject}, checking files: {movingfile_ref, movingfile_moving}\n')
        cost_file = movingfile_moving[0:-4]+f'.{cost_func}.data'
        
        if not recompute and os.path.exists(cost_folder+'/'+cost_file) and os.path.getsize(cost_folder+'/'+cost_file) > 0:
            print(f'cost values were already computed at {cost_file}\n')
        else:
            global_cost, local_cost = rcf.do_check_coregistration(required_folder+'/'+movingfile_ref, required_folder+'/'+movingfile_moving, cost_func, voi_size, step_size, masking = True, measure_global = True, measure_local = True)
            np.savetxt(cost_folder+'/'+cost_file, [global_cost, local_cost])

def main(data_dir, subject, voi_size, step_size):
    '''
    Parameters
    ----------
    data_dir : str
        string representing the data directory.
    subject : str
        subject ID.

    Returns
    -------
    SSD, NCC, NMI and CA values for quality control of rigid and affine registrations.
    '''
    image_types = ['hrT1', 'hrT2', 'hrFLAIR']
    costs = ['ncc', 'nmi', 'cor']
    reg_types = ['align', 'affine']
    
    for image_type in image_types:
        for cost in costs:
            # computing cost for co-registration of T2/FLAIR brain to T1 brain
            if image_type == 'hrFLAIR' or image_type == 'hrPD':
                compute_coreg_cost_vectors(data_dir, subject, cost, image_type, voi_size, step_size)
            for reg_type in reg_types:
            # dealing with hrT1, hrT2 and hrFLAIR for bigdata and HCPYA, rigid and affine registration
                compute_cost_vectors(data_dir, subject, reg_type, cost, image_type, voi_size, step_size)

# for image_type in image_types:
#         for cost in costs:
#             # computing cost for co-registration of T2/FLAIR brain to T1 brain
#             if image_type == 'hrT2' or image_type == 'hrFLAIR' or image_type == 'hrPD':
#                 compute_coreg_cost_vectors(data_dir, subject, cost, image_type, voi_size, step_size)
#             for reg_type in reg_types:
#             # dealing with hrT1, hrT2 and hrFLAIR for bigdata and HCPYA, rigid and affine registration
#                 compute_cost_vectors(data_dir, subject, reg_type, cost, image_type, voi_size, step_size)
            
    print('done computation\n')


    
    
