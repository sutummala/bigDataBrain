# Code created by Sudhakar on April 2020
# Computes cost-function for checking the registration

import progressbar
import numpy as np
import nibabel as nib
import all_cost_functions as acf

# Computing local similarity values
def compute_local_similarity(ref_image, moving_image, cost_func, voi_size):
    x,y,z = np.shape(ref_image)
    cost_vector = []
    bar = progressbar.ProgressBar(maxval= x, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in range(x-voi_size):
        bar.update(i+1)
        for j in range(y-voi_size):
            for k in range(z-voi_size):
                ref_voi = ref_image[i:i+voi_size, j:j+voi_size, k:k+voi_size]
                moving_voi = moving_image[i:i+voi_size, j:j+voi_size, k:k+voi_size]
                if all(np.ndarray.flatten(ref_voi) == 0) or all(np.ndarray.flatten(moving_voi) == 0):
                    continue
                if cost_func == 'ssd':
                    cost_vector.append(acf.ssd(ref_voi, moving_voi))
                elif cost_func == 'cc':
                    cost_vector.append(acf.cc(ref_voi, moving_voi))
                elif cost_func == 'ncc':
                    cost_vector.append(np.abs(acf.ncc(ref_voi, moving_voi, 'spearman')))
                elif cost_func == 'mi':
                    cost_vector.append(acf.mi(ref_voi, moving_voi)[0])
                elif cost_func == 'nmi':
                    cost_vector.append(acf.nmi(ref_voi, moving_voi))
                else:
                    print('cost function is not defined\n')
    bar.finish()
    print(f'local similarity ({cost_func}) between reference and moving computed is: {np.average(cost_vector[~np.isnan(cost_vector)])}\n')
    cost_vector = np.array(cost_vector)
    return np.average(cost_vector[~np.isnan(cost_vector)]) # removing nan values if any
 
# Computing global similarity value                   
def compute_global_similarity(ref_image, moving_image, cost_func):                           
    if cost_func == 'ssd':
        # 1. Sum of squared differences(SSD)
        similarity = acf.ssd(ref_image, moving_image)
    elif cost_func == 'cc':
        # 2. Cross Correlation 
        similarity = acf.cc(ref_image, moving_image)
    elif cost_func == 'ncc':
        # 3. Normalited Cross Correlation
        similarity = acf.ncc(ref_image, moving_image, 'spearman')
    elif cost_func == 'mi':
        # 4. Mutual Information
        similarity, _, _ = acf.mi(ref_image, moving_image)
    elif cost_func == 'nmi':   
        # 5. Normalized Mutual Information
        similarity = acf.nmi(ref_image, moving_image)
    
    print(f'global similarity ({cost_func}) between reference and moving computed is: {similarity}\n')
    return similarity
    
# Checking registration to MNI template 
def do_check_registration(refpath, movingfile, cost_func, masking, measure_global, measure_local):
    
    ref = refpath+'/'+'MNI152_T1_1mm.nii.gz' # Whole brain MN
    refmask = refpath+'/'+'MNI152_T1_1mm_brain_mask.nii.gz' # MNI brain mask
    
    reference = nib.load(ref)
    ref_image = reference.get_fdata() # reference image
    #ref_image = ref_image/max(np.ndarray.flatten(ref_image))
    
    ref_mask = nib.load(refmask)
    refmask_image = ref_mask.get_fdata() # mask based on reference
    
    moving = nib.load(movingfile)
    moving_image = moving.get_fdata() # moving image (image under check)
    #moving_image = moving_image/max(np.ndarray.flatten(moving_image))
    
    if masking:
        ref_image = ref_image * refmask_image
        moving_image = moving_image * refmask_image
    
    local_cost = []
    global_cost = []
    if ref_image.shape == moving_image.shape:
        if measure_local:
            local_similarity = compute_local_similarity(ref_image, moving_image, cost_func, voi_size = 5) # local similarity
            local_cost.append(local_similarity)
        else:
            print('local similarity measure is not requested\n')
        if measure_global:
            global_similarity = compute_global_similarity(ref_image, moving_image, cost_func) # global similarity
            global_cost.append(global_similarity)
        else:
            print('global similarity meausre is not requested\n')
    else:
        print('image shape mismatch!\n')
    return global_cost, local_cost
        
# Checking co-registration of T2/FLAIR to T1 brain or T1/T2/FLAIR to MNI if full path is given in ref
def do_check_coregistration(ref, movingfile, cost_func, masking, measure_global, measure_local):
       
    reference = nib.load(ref)
    ref_image = reference.get_fdata() # reference image
    #ref_image = ref_image/max(np.ndarray.flatten(ref_image)) # Intensity normalization
    
    moving = nib.load(movingfile)
    moving_image = moving.get_fdata() # moving image (image under check)
    #moving_image = moving_image/max(np.ndarray.flatten(moving_image))
    
    if masking:
        pass
    
    local_cost = []
    global_cost = []
    if ref_image.shape == moving_image.shape:
        if measure_local:
            local_similarity = compute_local_similarity(ref_image, moving_image, cost_func, voi_size = 5) # local similarity
            local_cost.append(local_similarity)
        else:
            print('local similarity measure is not requested\n')
        if measure_global:
            global_similarity = compute_global_similarity(ref_image, moving_image, cost_func) # global similarity
            global_cost.append(global_similarity)
        else:
            print('global similarity meausre is not requested\n')
    else:
        print('image shape mismatch!\n')
    return global_cost, local_cost


        
