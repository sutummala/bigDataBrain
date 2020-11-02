# code created by Sudhakar on May 2020
# generate test images and compute registration cost

import os
import sys
import nipype_all_functions as naf
import numpy as np
import decompose_compose_trans_matrix as dctm
import registration_cost_function as rcf

# datapath = sys.argv[1] # Path to the subjects data directory
# subject = sys.argv[2] # Subject ID

refpath = "/home/tummala/mri/tools/fsl/data/standard" # FSL template
ref = refpath+'/MNI152_T1_1mm.nii.gz' # Whole brain MNI 

def generate_new_param(scales, trans, rots, t):
    ''' generate new scales, trans, rots'''
    
    if t <= .2*t:
        scales_new = scales + (np.random.rand(3) * 0.3) 
        trans_new = trans + (np.random.randn(3) * 10)
        rots_new = rots + (np.random.randn(3) * 0.2)
    elif t > .2*t and t <= .4*t:
        scales_new = scales 
        trans_new = trans + (np.random.randn(3) * 10)
        rots_new = rots + (np.random.randn(3) * 0.2)
    elif t > .4*t and t <= .6*t:
        scales_new = scales + (np.random.rand(3) * 0.2) 
        trans_new = trans 
        rots_new = rots + (np.random.randn(3) * 0.2)
    elif t > .6*t and t <= .8*t:
        scales_new = scales + (np.random.rand(3) * 0.3) 
        trans_new = trans + (np.random.randn(3) * 10)
        rots_new = rots 
    else:
        scales_new = scales + (np.random.rand(3) * 0.1) 
        trans_new = trans + (np.random.randn(3) * 5)
        rots_new = rots + (np.random.randn(3) * 0.1)
    
    return scales_new, trans_new, rots_new

def generate_coreg_test_images(datapath, subject, img_type, voi_size, step_size, no_of_test_images):
    ''' generating test images for co-registration of T2/FLAIR to T1'''
    
    mat_path = datapath+'/'+subject+'/mat'
    raw_path = datapath+'/'+subject+'/anat'
    
    for raw_file in os.listdir(raw_path):
        if raw_file.endswith('nu_corr.brain.nii') and 'hrT1' in raw_file:
            ref_file = raw_file
        elif raw_file.endswith('nu_corr.brain.nii') and img_type in raw_file:
            moving_file = raw_file
    
    if img_type == 'hrT2':
        mat_tag = 't2-t1.mat'
        test_mat_path = datapath+'/'+subject+'/test_mat_T2_T1'+str(voi_size)+str(step_size)
        test_imgs_path = datapath+'/'+subject+'/test_imgs_T2_T1'+str(voi_size)+str(step_size)
    elif img_type == 'hrFLAIR':
        mat_tag = 'flair-t1.mat'
        test_mat_path = datapath+'/'+subject+'/test_mat_FLAIR_T1'+str(voi_size)+str(step_size)
        test_imgs_path = datapath+'/'+subject+'/test_imgs_FLAIR_T1'+str(voi_size)+str(step_size)
           
    for test_matfile in sorted(os.listdir(mat_path)):
        if test_matfile.endswith(mat_tag):
            
            if not os.path.exists(test_imgs_path):
                os.makedirs(test_imgs_path)
            if not os.path.exists(test_mat_path):
                os.makedirs(test_mat_path)
                
            mat_orig = np.loadtxt(mat_path+'/'+test_matfile)
            scales, trans, rots = dctm.decompose(mat_orig, angles = True) # decomposing the transformation matrix into scales, translations and rotations
            for t in range(no_of_test_images):
                testfile = f'{test_matfile[0:-4]}.test{t+1}.mat'
                outfile = f'{moving_file[0:-4]}.test{t+1}.nii' 
                
                scales_new, trans_new, rots_new = generate_new_param(scales, trans, rots, t)
                                
                mat_modified = dctm.compose(scales_new, trans_new, rots_new, origin = None)
                #print(mat_orig - mat_modified)
                
                if os.path.exists(test_mat_path+'/'+testfile):
                    print(f'test mat was already generated, {testfile}')
                else:
                    np.savetxt(test_mat_path+'/'+testfile, mat_modified, fmt = '%10.19f')
                    
                if os.path.exists(test_imgs_path+'/'+outfile):
                    print(f'test image was already generated, {outfile}')
                else:
                    naf.doApplyXFM(raw_path+'/'+moving_file, test_mat_path+'/'+testfile, raw_path+'/'+ref_file, test_imgs_path+'/'+outfile, 'spline', img_type)

def compute_coreg_test_cost_vectors(datapath, subject, cost_func, image_type, voi_size, step_size):
    ''' computes cost vector for given combination of cost and registration type'''
    
    recompute = True # flag to recompute the cost values
    
    print(f'doing for cost {cost_func}\n')
    
    raw_path = datapath+'/'+subject+'/anat'
    for raw_file in os.listdir(raw_path):
        if raw_file.endswith('nu_corr.brain.nii') and 'hrT1' in raw_file:
            ref_file = raw_file
    
    if image_type == 'hrT2':
        required_folder = datapath+'/'+subject+'/test_imgs_T2_T1'+str(voi_size)+str(step_size)
        cost_folder = datapath+'/'+subject+'/test_cost_T2_T1'+str(voi_size)+str(step_size)
    elif image_type == 'hrFLAIR':
        required_folder = datapath+'/'+subject+'/test_imgs_FLAIR_T1'+str(voi_size)+str(step_size)
        cost_folder = datapath+'/'+subject+'/test_cost_FLAIR_T1'+str(voi_size)+str(step_size)
    
    if os.path.exists(required_folder) and os.listdir(required_folder):
        if not os.path.exists(cost_folder):
            os.makedirs(cost_folder)
        movingfiles = os.listdir(required_folder)
        for movingfile in movingfiles:
            print(f'{subject}, checking file: {movingfile}')
            cost_file = movingfile[0:-4]+f'.{cost_func}.data'
            
            if not recompute and os.path.exists(cost_folder+'/'+cost_file) and os.path.getsize(cost_folder+'/'+cost_file) > 0:
                print(f'cost values were already computed at {cost_file}')
            else:
                global_cost, local_cost = rcf.do_check_coregistration(raw_path+'/'+ref_file, required_folder+'/'+movingfile, cost_func, voi_size, step_size, masking = True, measure_global = True, measure_local = True)
                np.savetxt(cost_folder+'/'+cost_file, [global_cost, local_cost], fmt = '%1.6f')

def generate_test_images(datapath, subject, img_type, reg_type, voi_size, step_size, no_of_test_images):
    ''' generating test images for both align and affine for all kinds of structural scans'''
    
    mat_path = datapath+'/'+subject+'/mat'
    raw_path = datapath+'/'+subject+'/anat'
    
    for raw_file in os.listdir(raw_path):
        if raw_file.endswith('reoriented.nii') and img_type in raw_file:
            infile = raw_file
    
    if reg_type == 'align':
        if img_type == 'hrT1':
            mat_tag = 't1-align.mat'
            test_mat_path = datapath+'/'+subject+'/test_mat_T1_align'+str(voi_size)+str(step_size)
            test_imgs_path = datapath+'/'+subject+'/test_imgs_T1_align'+str(voi_size)+str(step_size)
        elif img_type == 'hrT2':
            mat_tag = 't2-align.mat'
            test_mat_path = datapath+'/'+subject+'/test_mat_T2_align'+str(voi_size)+str(step_size)
            test_imgs_path = datapath+'/'+subject+'/test_imgs_T2_align'+str(voi_size)+str(step_size)
        elif img_type == 'hrFLAIR':
            mat_tag = 'flair-align.mat'
            test_mat_path = datapath+'/'+subject+'/test_mat_FLAIR_align'+str(voi_size)+str(step_size)
            test_imgs_path = datapath+'/'+subject+'/test_imgs_FLAIR_align'+str(voi_size)+str(step_size)
    elif reg_type == 'mni':
        if img_type == 'hrT1':
            mat_tag = 't1-mni.mat'
            test_mat_path = datapath+'/'+subject+'/test_mat_T1_mni'+str(voi_size)+str(step_size)
            test_imgs_path = datapath+'/'+subject+'/test_imgs_T1_mni'+str(voi_size)+str(step_size)
        elif img_type == 'hrT2':
            mat_tag = 't2-mni.mat'
            test_mat_path = datapath+'/'+subject+'/test_mat_T2_mni'+str(voi_size)+str(step_size)
            test_imgs_path = datapath+'/'+subject+'/test_imgs_T2_mni'+str(voi_size)+str(step_size)
        elif img_type == 'hrFLAIR':
            mat_tag = 'flair-mni.mat'
            test_mat_path = datapath+'/'+subject+'/test_mat_FLAIR_mni'+str(voi_size)+str(step_size)
            test_imgs_path = datapath+'/'+subject+'/test_imgs_FLAIR_mni'+str(voi_size)+str(step_size)
           
    for test_matfile in sorted(os.listdir(mat_path)):
        if test_matfile.endswith(mat_tag):
            
            if not os.path.exists(test_imgs_path):
                os.makedirs(test_imgs_path)
            if not os.path.exists(test_mat_path):
                os.makedirs(test_mat_path)
                
            mat_orig = np.loadtxt(mat_path+'/'+test_matfile)
            scales, trans, rots = dctm.decompose(mat_orig, angles = True) # decomposing the transformation matrix into scales, translations and rotations
            for t in range(no_of_test_images):
                testfile = f'{test_matfile[0:-4]}.test{t+1}.mat'
                outfile = f'{infile[0:-4]}.test{t+1}.nii'
                
                scales_new, trans_new, rots_new = generate_new_param(scales, trans, rots, t)
                
                mat_modified = dctm.compose(scales_new, trans_new, rots_new, origin = None)
                #print(mat_orig - mat_modified)
                
                if os.path.exists(test_mat_path+'/'+testfile):
                    print(f'test mat was already generated, {testfile}')
                else:
                    np.savetxt(test_mat_path+'/'+testfile, mat_modified, fmt = '%10.19f')
                    
                if os.path.exists(test_imgs_path+'/'+outfile):
                    print(f'test image was already generated, {outfile}')
                else:
                    naf.doApplyXFM(raw_path+'/'+infile, test_mat_path+'/'+testfile, ref, test_imgs_path+'/'+outfile, 'spline', img_type)
            
def compute_test_cost_vectors(datapath, subject, reg_type, cost_func, image_type, voi_size, step_size):
    ''' computes cost vector for given combination of cost and registration type'''
    
    recompute = True # flag to recompute the stuff if required
    
    print(f'doing for {reg_type} and cost {cost_func}\n')
    if reg_type == 'align':
        if image_type == 'hrT1':
            required_folder = datapath+'/'+subject+'/test_imgs_T1_align'+str(voi_size)+str(step_size)
            cost_folder = datapath+'/'+subject+'/test_cost_T1_align'+str(voi_size)+str(step_size)
        elif image_type == 'hrT2':
            required_folder = datapath+'/'+subject+'/test_imgs_T2_align'+str(voi_size)+str(step_size)
            cost_folder = datapath+'/'+subject+'/test_cost_T2_align'+str(voi_size)+str(step_size)
        elif image_type == 'hrFLAIR':
            required_folder = datapath+'/'+subject+'/test_imgs_FLAIR_align'+str(voi_size)+str(step_size)
            cost_folder = datapath+'/'+subject+'/test_cost_FLAIR_align'+str(voi_size)+str(step_size)
    elif reg_type == 'mni':
        if image_type == 'hrT1':
            required_folder = datapath+'/'+subject+'/test_imgs_T1_mni'+str(voi_size)+str(step_size)
            cost_folder = datapath+'/'+subject+'/test_cost_T1_mni'+str(voi_size)+str(step_size)
        elif image_type == 'hrT2':
            required_folder = datapath+'/'+subject+'/test_imgs_T2_mni'+str(voi_size)+str(step_size)
            cost_folder = datapath+'/'+subject+'/test_cost_T2_mni'+str(voi_size)+str(step_size)
        elif image_type == 'hrFLAIR':
            required_folder = datapath+'/'+subject+'/test_imgs_FLAIR_mni'+str(voi_size)+str(step_size)
            cost_folder = datapath+'/'+subject+'/test_cost_FLAIR_mni'+str(voi_size)+str(step_size)
    
    if os.path.exists(required_folder) and os.listdir(required_folder):
        if not os.path.exists(cost_folder):
            os.makedirs(cost_folder)
        movingfiles = os.listdir(required_folder)
        for movingfile in movingfiles:
            print(f'{subject}, checking file: {movingfile}')
            cost_file = movingfile[0:-4]+f'.{cost_func}.data'
            
            if not recompute and os.path.exists(cost_folder+'/'+cost_file) and os.path.getsize(cost_folder+'/'+cost_file) > 0:
                print(f'cost values were already computed at {cost_file}')
            else:
                global_cost, local_cost = rcf.do_check_registration(refpath, required_folder+'/'+movingfile, cost_func, voi_size, step_size, masking = True, measure_global = True, measure_local = True)
                np.savetxt(cost_folder+'/'+cost_file, [global_cost, local_cost], fmt = '%1.6f')

def main(data_dir, subject):
    '''
    Parameters
    ----------
    data_dir : str
        string representing the data directory.
    subject : str
        subject ID.

    Returns
    -------
    test transformation matrics, test images and computed cost values on test images.
    
    '''
    image_types = ['hrT1', 'hrT2', 'hrFLAIR']
    costs = ['ncc', 'nmi', 'cor']
    reg_types = ['align', 'mni']
    
    # VOI size and step size for local cost computation
    voi_size = 3
    step_size = 3 # stride

    for image_type in image_types:
        for reg_type in reg_types:
            # generating test images for each img_type
            generate_test_images(data_dir, subject, image_type, reg_type, voi_size, step_size, no_of_test_images = 5) # generate test images for each subject
            for cost in costs:
                # computing cost for test images (T1, T2 and FLAIR)
                compute_test_cost_vectors(data_dir, subject, reg_type, cost, image_type, voi_size, step_size)
            
# for image_type in image_types[1:]:
#     # genrating test images for co-reg of T2/FLAIR brain to T1 brain
#     generate_coreg_test_images(image_type, voi_size, step_size, no_of_test_images = 5) # generate test images for each subject
#     for cost in costs:
#         # computing cost for test images of T2/FLAIR brain aligned to T1 brain
#         compute_coreg_test_cost_vectors(cost, image_type, voi_size, step_size)
    
print('done with computations\n')
       

