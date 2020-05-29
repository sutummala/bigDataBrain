
# Code created by Sudhakar on March 2020
# Nipype functions for processing for structural data (T1, T2 and FLAIR)

import os
import nibabel as nib

from nipype.interfaces import spm, fsl, freesurfer, ants
fsl.FSLCommand.set_default_output_type('NIFTI')

def doAverage(infile1, infile2, outfile): # Doing average 
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
    print('doing re-orientation', infile)
    reorient = fsl.utils.Reorient2Std()
    reorient.inputs.in_file = infile
    reorient.inputs.out_file = outfile
    reorient.inputs.output_type = 'NIFTI'
    reorient.run()
    print('re-orientation done', outfile, '\n')
    
def doCropping(infile, outfile, outmat): # Doing Cropping
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
    flt = fsl.FLIRT()
    flt.inputs.in_file = infile
    if tag == "T1":
        if dof == 12:
            print('doing affine transformation of T1 to MNI...')
        elif dof == 6:
            print('doing rigid transformation of T1 brain to MNI brain...')
    elif tag == "T2":
        print('aligning T2 brain to T1 brain using rigid transformation...')
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
    print('doing inverse of', inmat)
    invt = fsl.ConvertXFM() 
    invt.inputs.in_file = inmat
    invt.inputs.invert_xfm = True
    invt.inputs.out_file = outmat
    invt.run()
    print('inverse finished', outmat, '\n')
    
def doConcatXFM(inmat1, inmat2, outmat): # Doing Concatenation of transformation matrices
    print('doing concat of', inmat1, inmat2)
    cont = fsl.ConvertXFM() 
    cont.inputs.in_file = inmat1
    cont.inputs.in_file2 = inmat2
    cont.inputs.concat_xfm = True
    cont.inputs.out_file = outmat
    cont.run()
    print('Concatenation finished', outmat, '\n')
    
def doApplyMasking(infile, maskfile, outfile): # Doing Masking of input image with a given mask
    print('doing masking of', infile)
    mask = fsl.ApplyMask() 
    mask.inputs.in_file = infile
    mask.inputs.mask_file = maskfile
    mask.inputs.out_file = outfile
    mask.run()
    print('masking finished', outfile, '\n')
    
def doApplyXFM(infile, inmat, ref, outfile, intertype, tag): # Doing transformation using existing mat files
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
    
def doCalculateSimilarity(infile, reference, inmask, refmask, costfun, tag):
    print('calculating the registration similarity b/w', infile, reference, '\n')
    sim = ants.MeasureImageSimilarity()
    sim.inputs.dimension = 3
    sim.inputs.metric = costfun
    sim.inputs.fixed_image = reference
    sim.inputs.moving_image = infile
    sim.inputs.metric_weight = 1.0
    sim.inputs.radius_or_number_of_bins = 5
    sim.inputs.sampling_strategy = 'Regular'
    sim.inputs.sampling_percentage = 1.0
    sim.inputs.fixed_image_mask = refmask
    sim.inputs.moving_image_mask = inmask
    sim.run()
