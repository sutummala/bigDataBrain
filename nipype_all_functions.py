
# Code created by Sudhakar on March 2020
# Nipype functions for processing for structural data (T1, T2 and FLAIR)

import os
import json
import nibabel as nib

from nipype.interfaces import spm, fsl, freesurfer, ants
fsl.FSLCommand.set_default_output_type('NIFTI')

def doAverage(infile1, infile2, outfile): # Doing average
    '''
    Parameters
    ----------
    infile1 : str
        path containing the input image one.
    infile2 : str
        path containing the input image two.
    outfile : str
        path to save the average of two input images.
    
    Returns
    -------
    an image which is average of two input images
    '''
    print('doing average of', infile1, infile2)
    avg = fsl.MultiImageMaths()
    avg.inputs.in_file = infile1
    avg.inputs.operand_files = infile2
    avg.inputs.op_string = "-add %s -div 2"
    avg.inputs.out_file = outfile
    avg.inputs.output_type = 'NIFTI_GZ'
    avg.run()
    print('averaging done', outfile, '\n')
    
def reOrientation(infile, outfile): # Doing Re-orientation
    '''
    Parameters
    ----------
    infile : str
        path containing the input image.
    outfile : str
        path to save the image after re-orientation.
    
    Returns
    -------
    an image which is the re-oriented version of input image
    '''
    print('doing re-orientation', infile)
    reorient = fsl.utils.Reorient2Std()
    reorient.inputs.in_file = infile
    reorient.inputs.out_file = outfile
    reorient.inputs.output_type = 'NIFTI'
    reorient.run()
    print('re-orientation done', outfile, '\n')
    
def doCropping(infile, outfile, outmat): # Doing Cropping
    '''
    Parameters
    ----------
    infile : str
        path containing the input image.
    outfile : str
        path to save the output image.
    outmat : str
        path to save the cropping information as a matrix.
    
    Returns
    -------
    an image which is cropped version of input image (removing neck and lower head parts)
    a text file containing the cropping information
    '''
    print('doing cropping', infile)
    robustfov = fsl.utils.RobustFOV()
    robustfov.inputs.in_file = infile
    robustfov.inputs.out_roi = outfile
    robustfov.inputs.brainsize = 170 # 170 mm above the neck will be chopped off
    robustfov.inputs.output_type = 'NIFTI'
    robustfov.inputs.out_transform = outmat
    robustfov.run()
    print('cropping done', outfile, '\n')
    
def doRemoveNegativeValues(infile): # Removes negative values for N4 bias-correction (as the intensities are log transformed)
    '''
    Parameters
    ----------
    infile : str
        path containing the input image.
        
    Returns
    -------
    an image in which all negative values are removed
    '''
    print('removing negative values and values close to zero before N4 in', infile)
    tempImage = nib.load(infile)
    tempVol = tempImage.get_fdata()
    tempVol_all_positve = tempVol.clip(min = 1) # making any values below one to zero

    # Nifti1
    if tempImage.header['sizeof_hdr'] == 348:
        tempImage_all_positive = nib.Nifti1Image(tempVol_all_positve, tempImage.affine, tempImage.header)
    # Nifti2
    elif tempImage.header['sizeof_hdr'] == 540:
        tempImage_all_positive = nib.Nifti2Image(tempVol_all_positve, tempImage.affine, tempImage.header)
    else:
        raise IOError('input image header problem in saving the file', infile)

    nib.save(tempImage_all_positive, infile)
    print('negative values are removed for N4 in', infile, '\n')
    
def doN4BiasFieldCorrection(infile, outfile): # Doing N4 Bias-Field Correction
    '''
    Parameters
    ----------
    infile : str
        path containing the input image.
    outfile : str
        path to save the image after N4 bias-field correction (N4 is from ANTS).
    
    Returns
    -------
    an image which is the bias-field corrected version of input image
    '''
    print('doing bias correction using N4 for', infile)
    n4 = ants.N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = infile
    n4.inputs.save_bias = False
    n4.inputs.output_image = outfile
    n4.inputs.bspline_fitting_distance = 100
    n4.inputs.rescale_intensities = True
    n4.inputs.convergence_threshold = 0
    n4.inputs.shrink_factor = 2
    n4.inputs.n_iterations = [50,50,50,50]
    n4.inputs.histogram_sharpening = (0.14, 0.01, 200)
    n4.run()
    print('bias-corection done', outfile, '\n')
    
def doN3BiasFieldCorrection(infile, outfile): # Doing N3 Bias-Field Correction
    '''
    Parameters
    ----------
    infile : str
        path containing the input image.
    outfile : str
        path to save the image after N3 bias-field correction (N3 is from Free Surfer).
    
    Returns
    -------
    an image which is the bias-field corrected version of input image
    '''
    print('doing bias correction using N3 for', infile)
    n3 = freesurfer.MNIBiasCorrection()
    n3.inputs.in_file = infile
    n3.inputs.iterations = 4
    n3.inputs.distance = 50
    n3.inputs.out_file = outfile
    n3.inputs.protocol_iterations = 1000
    n3.run()
    print('bias-corection done', outfile, '\n')
    
def doBrainExtraction(infile, maskfile, outfile, fraction): # Doing Brain Extraction
    '''
    Parameters
    ----------
    infile : str
        path containing the input image.
    maskfile : str
        path to save the mask after brain extraction.
    outfile : str
        path to save the image after brain extraction (BET tool from FSL is used).
    fraction : float
        fraction that controls extent of brain extraction (lower value removes more outer brain).
    
    Returns
    -------
    an image which is the brain extracted version of input image
    a binary image which is the mask for extracted brain
    '''
    print('doing brain extraction for', infile)
    btr = fsl.BET()
    btr.inputs.in_file = infile
    btr.inputs.frac = fraction
    btr.inputs.robust = True
    btr.inputs.out_file = outfile
    btr.inputs.mask = True
    btr.inputs.output_type = 'NIFTI'
    btr.run()
    os.rename(maskfile[0:-9]+'_mask.nii', maskfile)
    print('brain extraction completed', outfile, '\n')
    
def doFLIRT(infile, reference, outfile, outmat, dof, costfunc, interpfunc, tag): # Doing Affine and Rigid Transformation
    '''
    Parameters
    ----------
    infile : str
        path containing the input image.
    reference : str
        path containing the reference to which the input image will be registered.
    outfile : str
        path to save the output image.
    outmat : str
        path to save the transformation matrix.
    dof : int
        degrees of freedom (6 for rigid and 12 for affine)
    costfunc : str
        string describes the cost function
    interpfunc : str
        str containing the interpolation type
    tag : str
        tag to identify the image type. e.g. T1, T2, FLAIR etc.
    
    Returns
    -------
    an image which is the transformed version of input image
    the corresponding transformation matrix file
    '''
    flt = fsl.FLIRT()
    flt.inputs.in_file = infile
    if tag == "T1":
        if dof == 12:
            print('doing affine transformation of T1 to MNI...')
        elif dof == 6:
            print('doing rigid transformation of T1 brain to MNI brain...')
    elif tag == "T2":
        print('aligning T2 brain to T1 brain using rigid transformation...')
    elif tag == "PD":
        print('aligning PD brain to T1 brain using rigid transformation...')
    elif tag == "FLAIR":
        print('aligning FLAIR brain to T1 brain using rigid transformation...')
    elif tag == "alignment":
        print('aligning second series to first series using rigid transformation...')
    else: 
        print('Incorrect tag is chosen')
    flt.inputs.dof = dof
    flt.inputs.cost_func = costfunc
    flt.inputs.reference = reference
    flt.inputs.out_file = outfile
    flt.inputs.out_matrix_file = outmat
    flt.inputs.output_type = 'NIFTI'
    flt.inputs.no_resample = True
    flt.inputs.no_resample_blur = True
    flt.inputs.interp = interpfunc
    flt.run()
    print('transformation done ', outfile, '\n')
    
def doInverseXFM(inmat, outmat): # Doing Inverse of transformation matrix
    '''
    Parameters
    ----------
    inmat : str
        path containing the input transformation matrix.
    outmat : str
        path to store output transformation matrix.
    
    Returns
    -------
    an inverse transformation matrix of the input matrix
    '''
    print('doing inverse of', inmat)
    invt = fsl.ConvertXFM() 
    invt.inputs.in_file = inmat
    invt.inputs.invert_xfm = True
    invt.inputs.out_file = outmat
    invt.run()
    print('inverse finished', outmat, '\n')
    
def doConcatXFM(inmat1, inmat2, outmat): # Doing Concatenation of transformation matrices
    '''
    Parameters
    ----------
    inmat1 : str
        path containing the first input transformation matrix.
    inmat2 : str
        path containing the second input transformation matrix.
    outmat : str
        path to store output transformation matrix.
    
    Returns
    -------
    a combined transformation matrix of the two input matrices
    '''
    print('doing concat of', inmat1, inmat2)
    cont = fsl.ConvertXFM() 
    cont.inputs.in_file = inmat1
    cont.inputs.in_file2 = inmat2
    cont.inputs.concat_xfm = True
    cont.inputs.out_file = outmat
    cont.run()
    print('Concatenation finished', outmat, '\n')
    
def doApplyMasking(infile, maskfile, outfile): # Doing Masking of input image with a given mask
    '''
    Parameters
    ----------
    infile : str
        path containing the image to be masked.
    maskfile : str
        path to store mask of the input image.
    outfile : str
        path to store masked version of the input image.
    
    Returns
    -------
    both mask and masked version of input image
    '''
    print('doing masking of', infile)
    mask = fsl.ApplyMask() 
    mask.inputs.in_file = infile
    mask.inputs.mask_file = maskfile
    mask.inputs.out_file = outfile
    mask.run()
    print('masking finished', outfile, '\n')
    
def doApplyXFM(infile, inmat, ref, outfile, intertype, tag): # Doing transformation using existing mat files
    '''
    Parameters
    ----------
    infile : str
        path containing the input image.
    inmat : str
        path containing the input transformation matrix.
    ref : str
        path containing the standard MNI_T1_1mm template.
    outfile : str
        path to save the output image.
    intertype : str
        string containing the interpolation type during transformation. e.g. spline, etc.
    tag : str
        tag to identify the image type.e.g. T1, T2, FLAIR etc.
    
    Returns
    -------
    an image with input transformation applied
    '''
    print('doing transformation/cropping using trasformation/cropping matrix', tag, infile)
    applyxfm = fsl.ApplyXFM()
    applyxfm.inputs.apply_xfm = True
    applyxfm.inputs.reference = ref
    applyxfm.inputs.in_file = infile
    applyxfm.inputs.out_file = outfile   
    applyxfm.inputs.in_matrix_file = inmat
    applyxfm.inputs.out_matrix_file = inmat # There will be no change in the in_matrix_file since it is applyxfm
    applyxfm.inputs.no_resample = True
    applyxfm.inputs.no_resample_blur = True
    applyxfm.inputs.interp = intertype
    applyxfm.run()
    print(tag, 'is transformed/cropped', outfile, '\n')
    
def do_json_combine(reg_folder, subject, remove_individual_json): # combine all json files into a single json
    '''
    Parameters
    ----------
    reg_folder : str
        path containing the json files.
    subject : str
        subject id.
    remove_individual_json: boolean
        removes individual json files if true

    Returns
    -------
        a json file with individual json files merged together
    '''
    result = []
    result.append({'subject_id': subject})
    print(f'merging all available json files at {reg_folder}\n')
    for f in os.listdir(reg_folder):
        with open(reg_folder+'/'+f, 'r') as infile:
            result.append(json.load(infile))
        if remove_individual_json:
            os.remove(reg_folder+'/'+f)

    with open(reg_folder+'/'+'merged_file.json', 'w') as merged_json:
        json.dump(result, merged_json, indent = 4)
    print(f'all json files are merged into {merged_json}\n')
    
def do_spm_new_segment(tpm_file, infile):
    '''
    Parameters
    ----------
    tpm_file : str
        full path to the TPM.nii file in the SPM12 directory.
    infile : str
        full path to the T1-weighted image.

    Returns
    -------
    gray matter, white matter and CSF probability maps in native space.

    '''
    seg = spm.NewSegment()
    seg.inputs.affine_regularization = 'mni'
    seg.inputs.sampling_distance = 2
    seg.inputs.channel_files = infile # T1-weighted image
    seg.inputs.channel_info = (0.0001, 60, (False, False))
    tissue1 = ((tpm_file, 1), 2, (True,False), (False, False))
    tissue2 = ((tpm_file, 2), 2, (True,False), (False, False))
    tissue3 = ((tpm_file, 3), 2, (True,False), (False, False))
    tissue4 = ((tpm_file, 4), 2, (False,False), (False, False))
    tissue5 = ((tpm_file, 5), 2, (False,False), (False, False))
    seg.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5]
    seg.run()
    
def do_spm_new_segment_multi_channel(tpm_file, *infiles):
    '''
    Parameters
    ----------
    tpm_file : str
        full path to the TPM.nii file in the SPM12 directory.
    *infiles : str
        full paths to the T1-weighted image and the corresponding co-registered T2/FLAIR image.

    Returns
    -------
    gray matter, white matter and CSF probability maps in native space.

    '''
    seg = spm.MultiChannelNewSegment()
    seg.inputs.affine_regularization = 'mni'
    seg.inputs.sampling_distance = 2
    channel1= (infiles[0],(0.0001, 60, (False, False))) # T1-weighted image
    channel2= (infiles[1],(0.0001, 60, (False, False))) # T2/FLAIR and it should be co-registered to the corresponding T1 before segmentation
    seg.inputs.channels = [channel1, channel2]
    tissue1 = ((tpm_file, 1), 2, (True,False), (False, False))
    tissue2 = ((tpm_file, 2), 2, (True,False), (False, False))
    tissue3 = ((tpm_file, 3), 2, (True,False), (False, False))
    tissue4 = ((tpm_file, 4), 2, (False,False), (False, False))
    tissue5 = ((tpm_file, 5), 2, (False,False), (False, False))
    seg.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5]
    seg.run() 
    