# Code created by Sudhakar on April and modified on October 2020
# Computes cost-function for checking the registration quality

import progressbar
import numpy as np
import nibabel as nib
import all_cost_functions as acf

# Computing local similarity values
def compute_local_similarity(ref_image, moving_image, cost_func, voi_size, step_size):
    '''
    Parameters
    ----------
    ref_image : str
        path to the reference image.
    moving_image : str
        path to the moving image.
    cost_func : str
        cost function.
    voi_size : int
        size of the volume of interest, e.g. 3 or 5 or 7 etc.
    step_size : int
        size of the step size in sliding the VOI over the image, ideally it may be same as voi_size, but that is not the strict requirement.
        
    Returns
    -------
    float
        returns local cost.
    '''
    x,y,z = np.shape(ref_image)
    cost_vector = []
    #bar = progressbar.ProgressBar(maxval= x, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #bar.start()
    for i in range(round(x/step_size)-voi_size):
        #bar.update(i+1)
        for j in range(round(y/step_size)-voi_size):
            for k in range(round(z/step_size)-voi_size):
                ref_voi = ref_image[i*step_size:(i*step_size)+voi_size, j*step_size:(j*step_size)+voi_size, k*step_size:(k*step_size)+voi_size]
                moving_voi = moving_image[i*step_size:(i*step_size)+voi_size, j*step_size:(j*step_size)+voi_size, k*step_size:(k*step_size)+voi_size]
                if all(np.ndarray.flatten(ref_voi) == 0) or all(np.ndarray.flatten(moving_voi) == 0):
                    continue
                if cost_func == 'ssd':
                    cost_vector.append(acf.ssd(ref_voi, moving_voi))
                elif cost_func == 'cc':
                    cost_vector.append(acf.cc(ref_voi, moving_voi))
                elif cost_func == 'ncc':
                    cost_vector.append(np.abs(acf.ncc(ref_voi, moving_voi, 'pearson'))) 
                elif cost_func == 'mi':
                    cost_vector.append(acf.mi(ref_voi, moving_voi)[0])
                elif cost_func == 'nmi':
                    cost_vector.append(acf.nmi(ref_voi, moving_voi))
                elif cost_func == 'cor':
                    cost_vector.append(acf.cor(ref_voi, moving_voi))
                else:
                    print('cost is not defined\n')
    #bar.finish()
    cost_vector = np.array(cost_vector)
    
    print(f'local similarity ({cost_func}) between reference and moving computed is: {np.average(cost_vector[~np.isnan(cost_vector)])}\n')
    return np.average(cost_vector[~np.isnan(cost_vector)]) # removing nan values if any
 
# Computing global similarity value                   
def compute_global_similarity(ref_image, moving_image, cost_func):
    '''
    Parameters
    ----------
    ref_image : str
        path to the reference image.
    moving_image : str
        path to the moving image.
    cost_func : str
        cost function.

    Returns
    -------
    similarity : float
        return global cost.
    '''                           
    if cost_func == 'ssd':
        # 1. Sum of squared differences(SSD)
        similarity = acf.ssd(ref_image, moving_image)
    elif cost_func == 'cc':
        # 2. Cross Correlation 
        similarity = acf.cc(ref_image, moving_image)
    elif cost_func == 'ncc':
        # 3. Normalited Cross Correlation
        similarity = acf.ncc(ref_image, moving_image, 'pearson') 
    elif cost_func == 'mi':
        # 4. Mutual Information
        similarity, _, _ = acf.mi(ref_image, moving_image)
    elif cost_func == 'nmi':   
        # 5. Normalized Mutual Information
        similarity = acf.nmi(ref_image, moving_image)
    elif cost_func == 'cor':
        # 6. Correlation Ratio
        similarity = acf.cor(ref_image, moving_image)
    else:
        print('cost if not defined\n')
    
    print(f'global similarity ({cost_func}) between reference and moving computed is: {similarity}\n')
    return similarity

# main function for computing local and global registration 
def do_compute_main(ref_image, moving_image, cost_func, voi_size, step_size, measure_global, measure_local):
    '''
    Parameters
    ----------
    ref_image : str
        fixed image.
    moving_image : str
        image under check.
    cost_func : str
        string describing the cost function.
    voi_size : int
        size of volume of interest, e.g. 3, 5 etc.
    step_size : int
        size of the step size in sliding the VOI over the image, ideally it is same as voi_size, but that is not the strict requirement.
    masking : boolean
        if true, brain masking will be performed i.e computation will be performed with in the actual brain region.
    measure_global : boolean
        if true global cost value will be computed.
    measure_local : boolean
        if true local cost value will be computed.

    Returns
    -------
    global_cost : float
        value of the global cost.
    local_cost : float
        value of the local cost.
    '''
    local_cost = []
    global_cost = []
    if ref_image.shape == moving_image.shape:
        if measure_local:
            local_similarity = compute_local_similarity(ref_image, moving_image, cost_func, voi_size, step_size) # local similarity
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
    
# Checking registration to MNI template 
def do_check_registration(refpath, movingfile, cost_func, voi_size, step_size, masking, measure_global, measure_local):
    '''
    Parameters
    ----------
    refpath : str
        path to the MNI standard template
    movingfile : str
        path to the moving file
    cost_func : str
        string describing the cost function.
    voi_size : int
        size of volume of interest, e.g. 3, 5 etc.
    step_size : int
        size of the step size in sliding the VOI over the image, ideally it is same as voi_size, but that is not the strict requirement.
    masking : boolean
        if true, brain masking will be performed i.e computation will be performed with in the actual brain region.
    measure_global : boolean
        if true global cost value will be computed.
    measure_local : boolean
        if true local cost value will be computed.

    Returns
    -------
    global_cost : float
        value of the global cost.
    local_cost : float
        value of the local cost.
    '''
    
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
    
    return do_compute_main(ref_image, moving_image, cost_func, voi_size, step_size, measure_global, measure_local)
        
# Checking co-registration of T2/FLAIR to T1 brain or T1/T2/FLAIR to MNI if full path is given in ref
def do_check_coregistration(ref, movingfile, cost_func, voi_size, step_size, masking, measure_global, measure_local):
    '''
    Parameters
    ----------
    ref : str
        path to the corresponding T1-weighted image
    movingfile : str
        path to the moving file
    cost_func : str
        string describing the cost function.
    voi_size : int
        size of volume of interest, e.g. 3, 5 etc.
    step_size : int
        size of the step size in sliding the VOI over the image, ideally it is same as voi_size, but that is not the strict requirement.
    masking : boolean
        if true, brain masking will be performed i.e computation will be performed with in the actual brain region.
    measure_global : boolean
        if true global cost value will be computed.
    measure_local : boolean
        if true local cost value will be computed.

    Returns
    -------
    global_cost : float
        value of the global cost.
    local_cost : float
        value of the local cost.
    '''
       
    reference = nib.load(ref)
    ref_image = reference.get_fdata() # reference image
    #ref_image = ref_image/max(np.ndarray.flatten(ref_image)) # Intensity normalization
    
    moving = nib.load(movingfile)
    moving_image = moving.get_fdata() # moving image (image under check)
    #moving_image = moving_image/max(np.ndarray.flatten(moving_image))
    
    if masking:
        pass
    
    return do_compute_main(ref_image, moving_image, cost_func, voi_size, step_size, measure_global, measure_local)
    


        
