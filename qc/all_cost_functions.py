# code created by Sudhakar on May/July 2020
# cost functions for checking registration and image quality

import os
import numpy as np
import pandas as pd
import scipy.stats
import nibabel as nib
import nipype_all_functions as naf

def get_file_name_and_extension(infile):
    '''
    Parameters
    ----------
    infile : str
        filename.

    Returns
    -------
    TYPE: str
        main name of the filename.
    TYPE: str
        extension of the filename.
    '''
    main_1, ext_1 = os.path.splitext(infile)
    if ext_1 == '.gz':
        main_2, ext_2 = os.path.splitext(main_1)
        return main_2, ext_2+ext_1
    else:
        return main_1, ext_1
    
def compute_weight(refimage, movingimage):
    '''
    Parameters
    ----------
    refimage : float
        volume of interest.
    movingimage : float
        volume of interest.

    Returns
    -------
    TYPE: float
        weight value.
    '''
    non_zeros = np.max(np.count_nonzero(refimage), np.count_nonzero(movingimage)) # generally, the number of non-zeros are same in both volume of interests. 
    return weight = 1/(1+(1000*np.exp(-non_zeros)))
    
## cost functions for checking goodness of registraitons

# 1. Sum of squared differences(SSD)
def ssd(refimage, movingimage):
    
    if refimage.shape != movingimage.shape:
        print('images shape mismatch')
    else:
        return compute_weight(refimage, movingimage)*np.sum((np.ndarray.flatten(refimage)-np.ndarray.flatten(movingimage))**2)
    
# 2. Cross Correlation 
def cc(refimage, movingimage):
    
    if refimage.shape != movingimage.shape:
        print('images shape mismatch')
    else:
        return compute_weight(refimage, movingimage)*np.sum(np.abs(np.correlate(np.ndarray.flatten(refimage), np.ndarray.flatten(movingimage), 'full')))

# 3. Normalized Cross Correlation (NCC)
def ncc(refimage, movingimage, cor_type):
    
    if refimage.shape != movingimage.shape:
        print('images shape mismatch')
    else:
        if cor_type == 'pearson':
            return compute_weight(refimage, movingimage)*np.corrcoef(np.ndarray.flatten(refimage), np.ndarray.flatten(movingimage))[0,1] # pearson's correlation coefficient (this could also be implemented using scipy.stats.pearsonr)
        elif cor_type == 'spearman':
            compute_weight(refimage, movingimage)*return scipy.stats.spearmanr(np.ndarray.flatten(refimage), np.ndarray.flatten(movingimage))[0] # spearman correlation coefficient
    
# Entropy for MI and NMI
def entropy(hist):
    
    hist_normalized = hist/float(np.sum(hist))
    hist_normalized = hist_normalized[np.nonzero(hist_normalized)]
    return -sum(hist_normalized * np.log2(hist_normalized))

# 4. Mutual Information 
def mi(refimage, movingimage):
    
    if refimage.shape != movingimage.shape:
        print('images shape mismatch')
    else:
        hist_joint = np.histogram2d(np.ndarray.flatten(refimage), np.ndarray.flatten(movingimage), bins = 10)[0]
        hist_ref = np.histogram(np.ndarray.flatten(refimage), bins = 10)[0]
        hist_moving = np.histogram(np.ndarray.flatten(movingimage), bins = 10)[0]
        
        entropy_joint = entropy(hist_joint)
        entropy_ref = entropy(hist_ref)
        entropy_moving = entropy(hist_moving)
        return entropy_ref + entropy_moving - entropy_joint, entropy_ref, entropy_moving

# 5. Normalized Mutual Information (NMI)
def nmi(refimage, movingimage):
    
    mutual_info, ent_ref, ent_moving = mi(refimage, movingimage)
    return compute_weight(refimage, movingimage)*(mutual_info/((ent_ref + ent_moving)*0.5))

# Correlation Ratio main function
def correlation_ratio_main(refimage, movingimage):
    fcat, _ = pd.factorize(refimage)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = movingimage[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(movingimage,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

# 6. Correlation Ratio (CR)
def cor(refimage, movingimage):
    
    if refimage.shape != movingimage.shape:
        print('images shape mismatch')
    else:
        return compute_weight(refimage, movingimage)*correlation_ratio_main(np.ndarray.flatten(refimage), np.ndarray.flatten(movingimage))
    
# 7. Entropy of Intensity Differnces (EID)
def eid(refimage, movingimage):
    
    if refimage.shape != movingimage.shape:
        print('images shape mismatch')
    else:
        intensity_diff = np.abs(np.ndarray.flatten(refimage) - np.ndarray.flatten(movingimage))
        hist_diff = np.histogram(intensity_diff, bins = 10)[0]
        return entropy(hist_diff)
    
## cost functions for checking image quality

# 1. mean signal intensity (MSI) in a specified direction (sagittal, coronal or axial)
def msi(in_image, direction):
    
    x, y, z = np.shape(in_image)
    
    mean_signal_intensity_x = []
    mean_signal_intensity_y = []
    mean_signal_intensity_z = []
    
    if direction == 'sagittal':
        for i in range(x):
            mean_signal_intensity_x.append(np.average(in_image[i, :, :]))
    elif direction == 'coronal':
        for j in range(y):
            mean_signal_intensity_y.append(np.average(in_image[:, j, :]))
    elif direction == 'axial':
        for k in range(z):
            mean_signal_intensity_z.append(np.average(in_image[:, :, k]))
                    
    if direction == 'sagittal':
        return np.average(mean_signal_intensity_x)
    elif direction == 'coronal':
        return np.average(mean_signal_intensity_y)
    elif direction == 'axial':
        return np.average(mean_signal_intensity_z)
            
# 2. signal to noise ratio (SNR) and signal-variance-to-noise-variance ratio (SVNR)
def snr_and_svnr(*paths):
    
    infile = os.path.join(paths[0], paths[1])
    
    # main, ext = get_file_name_and_extension(infile)
    # cropped_file = main+'.crop.nii'
    # outmat = main+'.crop.mat'
    # naf.doCropping(infile, cropped_file, outmat)
    # os.remove(os.path.join(paths[0], outmat))
    
    # input image after cropping 
    input_image = nib.load(infile)
    input_image_data = input_image.get_fdata()
    input_image_data_vector = np.sort(np.ndarray.flatten(input_image_data))
    
    threshold = 0.1 # threshold for background intensity cut-off
       
    input_image_data_vector_unique = np.unique(input_image_data_vector)
    hist, bin_edges = np.histogram(input_image_data_vector, bins = len(input_image_data_vector_unique))
    
    threshold_number = int(threshold*sum(hist))
    
    for i in range(len(hist)):
        if sum(hist[0:i]) >= threshold_number:
            break
        
    threshold_intensity = np.round(bin_edges[i]) # Intensity values less than or equal this threshold are considered as background
            
    background_data_vector = input_image_data_vector[input_image_data_vector <= threshold_intensity]
    head_data_vector = input_image_data_vector[input_image_data_vector > threshold_intensity]
    
    print(f'average of head region is {np.average(head_data_vector)} and standard deviation of background is {np.std(background_data_vector)}\n')
    print(f'variance of head region is {np.var(head_data_vector)} and variance of background is {np.var(background_data_vector)}\n')
    
    return np.average(head_data_vector)/np.std(background_data_vector), np.var(head_data_vector)/np.var(background_data_vector), np.var(background_data_vector)

# 3. contrast to noise ratio (CNR), contrast variance to noise variance ratio (CVNR) and tissue contrast to tissue intensity variation (TCTV)
def cnr_and_cvnr_and_tctv(*paths):
    
    infile = os.path.join(paths[0], paths[1])
    
    _, _, sd_background = snr_and_svnr(paths[0], paths[1])
    
    # input image
    input_image = nib.load(infile)
    input_image_data = input_image.get_fdata()
    input_image_data_vector = np.ndarray.flatten(input_image_data)
    
    # # doing reorientation
    # infile_r, ext_r = get_file_name_and_extension(infile)
    # outfile_r = infile_r+'.reoriented.nii'
    
    # naf.reOrientation(infile, outfile_r)
    
    # main, ext = get_file_name_and_extension(infile)
    # cropped_file = main+'.crop.nii'
    # outmat = main+'.crop.mat'
    # naf.doCropping(infile, cropped_file, outmat)
    # os.remove(os.path.join(paths[0], outmat))
    
    if paths[1].find('T1') or paths[1].find('FLAIR'):
        fraction = 0.4
    elif paths[1].find('T2') or paths[1].find('PD'):
        fraction = 0.2
    
    main, ext = get_file_name_and_extension(infile)
    maskfile = main+'.brain.mask.nii' 
    skullfile = main+'.brain.skull.nii'
    outfile = main+'.brain.nii'
    
    # doing brain extraction
    naf.do_Brain_Skull_Extraction(infile, maskfile, skullfile, outfile, fraction)
    
    brain_image = nib.load(outfile)
    brain_image_data = brain_image.get_fdata()
    brain_image_data_vector = np.ndarray.flatten(brain_image_data)
    
    # doing tissue (gray matter, white matter and CSF) segmentation
    naf.do_Tissue_Segmentation(outfile)
    
    seg_file = main+'.brain_seg.nii'
    seg_image = nib.load(seg_file)
    seg_image_data = seg_image.get_fdata()
    seg_image_data_vector = np.ndarray.flatten(seg_image_data)
    
    # extracting gray and white matter intensities
    if seg_file.find('hrT1') != -1:
        gray_matter = brain_image_data_vector[seg_image_data_vector == 2]
        white_matter = brain_image_data_vector[seg_image_data_vector == 3]
    elif seg_file.find('hrT2') != -1:
        gray_matter = brain_image_data_vector[seg_image_data_vector == 1]
        white_matter = brain_image_data_vector[seg_image_data_vector == 2]
    
    pooled_std = np.sqrt(np.std(white_matter)**2 + np.std(gray_matter)**2)
    print(f'difference of means of GM and WM is {np.abs(np.average(gray_matter)-np.average(white_matter))} & std of background is {sd_background}\n')
    print(f'difference of stds of GM and WM is {np.abs(np.std(gray_matter)-np.std(white_matter))} & std of background is {sd_background}\n')
    print(f'difference of stds of GM and WM is {np.abs(np.std(gray_matter)-np.std(white_matter))} & pooled std is {pooled_std}\n')
    
    return np.abs(np.average(gray_matter)-np.average(white_matter))/sd_background, np.abs(np.std(gray_matter)-np.std(white_matter))/sd_background, np.abs(np.std(gray_matter)-np.std(white_matter))/pooled_std
    
# 4. full width half maximum (FWHM) using AFNI
def fwhm(*paths):
    
    infile = os.path.join(paths[0], paths[1])
    
    main, ext = get_file_name_and_extension(infile)
    
    outfile = main+'.fwhm'
    
    # compute fwhm
    naf.do_Compute_FWHM(infile, outfile)
    
    return np.linalg.norm(np.loadtxt(outfile))

# 5. entropy (ent)
def ent(*paths):
    
    infile = os.path.join(paths[0], paths[1])
    
    input_image = nib.load(infile)
    input_image_data = input_image.get_fdata()
    
    hist_in = np.histogram(np.ndarray.flatten(input_image_data), bins = 10)[0]
    
    return entropy(hist_in)
    
    
    
    
    

