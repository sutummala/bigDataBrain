
# Code created by Sudhakar on March 2020
# pre-processing structural data, alignment of T1, T2/FLAIR to MNI

import os
import numpy as np
import nipype_all_functions as naf
import registration_cost_function as rcf

# checking registration
def do_registration_quality(*args):
    print(f'checking registration quality b/w {args[0], args[1]} with cost {args[2]}\n')
    global_cost, local_cost = rcf.do_check_coregistration(args[0], args[1], args[2], masking = True, measure_global = True, measure_local = True)
    
    print('doing quality check...\n')
    
    if args[2] == 'ncc':
        thr = 0.40
        thr_critical = 0.2
    elif args[2] == 'nmi':
        thr = 0.30
        thr_critical = 0.15
    
    print(f'actual thr is {local_cost[0]} and cut-off thr is {thr}, critical thr is {thr_critical}\n')
        
    split_path = args[1].split('/')
    
    saving_folder = os.path.join('/', split_path[1], split_path[2], split_path[3], split_path[4], split_path[5], 'reg_check')
    print(f'saving flags at {saving_folder}\n')
    
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
        
    save_file = split_path[7][:-4]+'.reg'
    print(f'reg flag file saved is {save_file}')
    
    if local_cost[0] >= thr:
        print(f'registration between {args[0]} and {args[1]} looks fine\n')
        np.savetxt(saving_folder+'/'+save_file, [1], fmt = '%d')
    elif local_cost[0] < thr and local_cost[0] >= thr_critical:
        print(f'registration needs manual checking for {args[1]}\n')
        np.savetxt(saving_folder+'/'+save_file, [0], fmt = '%d')
    elif local_cost[0] < thr_critical:
        print(f'registration is bad for {args[1]}, ignore the subject\n')
        np.savetxt(saving_folder+'/'+save_file, [-1], fmt = '%d')
                
# realignning and averaging if two series are found
def doAlignAverage(datapath, datapathMat, infile1, infile2, outfile):
    
    if infile2.endswith('.gz'):
        outfile1 = infile2[0:-6]+'alignedtoA.nii' # output in .nii format
    elif any([infile2.endswith('.nii'), infile2.endswith('.img')]):
        outfile1 = infile2[0:-3]+'alignedtoA.nii'
    matfile = datapathMat+'/'+infile1[0:-8]+'B_realignedTo_A.mat'
        
    if os.path.exists(datapath+'/'+outfile1):
        print('series 2 is realigned to series 1 already\n')
    else:
        naf.doFLIRT(datapath+'/'+infile2, datapath+'/'+infile1, datapath+'/'+outfile1, matfile, 6, 'corratio', 'spline', 'alignment')
        
    if os.path.exists(datapath+'/'+outfile):
        print('alignment and averaging already done\n')
    else:
        naf.doAverage(datapath+'/'+infile1, datapath+'/'+outfile1, datapath+'/'+outfile)
    
# preProcessing of T1 and T2/FLAIR Images
def preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, image, tag, maintag, nu_corr):
    
    ref = refpath+'/'+'MNI152_T1_1mm.nii.gz' # Whole brain MNI
    refbrain = refpath+'/'+'MNI152_T1_1mm_brain.nii.gz' # MNI brain
    refmask = refpath+'/'+'MNI152_T1_1mm_brain_mask.nii.gz' # MNI brain mask
    
    alltags = {'hrT1', 'hrT1.A', 'hrT1.B','hrT1.M','hrT2', 'hrT2.A', 'hrT2.B', 'hrT2.M', 'hrFLAIR', 'hrFLAIR.A', 'hrFLAIR.B', 'hrFLAIR.M', 'hrPD', 'hrPD.A', 'hrPD.B', 'hrPD.M'}
    
    if image.endswith('nii.gz'):
        if maintag in alltags: # It is used to properly assign extensions (by getting rid of nii.gz) for files generated during processing
            idx = -7
            idxc = idx-len(maintag)
    elif image.endswith('nii') or image.endswith('img'):
        if maintag in alltags: # It is used to properly assign extensions (by getting rid of .nii or .img) for files generated during processing
            idx = -4
            idxc = idx-len(maintag)
                
            
    # Pre-processig Steps
    # 1. Re-orientation (global rotation) using FSL
    fileRaw = datapath+'/'+image
    fileReo = datapath+'/'+image[0:idx]+'.reoriented.nii'
    if not os.path.exists(fileReo):
        naf.reOrientation(fileRaw, fileReo)
    else:
        print('re-orientation already done', fileReo, '\n')
        
    # 1a. Doing Bias-field correction for cropping 
    fileRenucorr = datapath+'/'+image[0:idx]+'.reoriented.nu_corr.nii'
    if not os.path.exists(fileRenucorr):
        if nu_corr == 'N3':
            naf.doN3BiasFieldCorrection(fileReo, fileRenucorr)
        elif nu_corr == 'N4':
            naf.doRemoveNegativeValues(fileReo) # Removing negative values for N4
            naf.doN4BiasFieldCorrection(fileReo, fileRenucorr)
    else:
        print('bias correction already done before cropping', fileRenucorr, '\n')
        
    # 2. Cropping (removing neck and lower head) using FSL
    fileCro = datapath+'/'+image[0:idx]+'.reoriented.nu_corr.cropped.nii'
    filecropped = datapath+'/'+image[0:idx]+'.reoriented.cropped.nii'
    
    matCroT1 = datapathMat+'/'+image[0:idxc]+'hrT1.cropped-t1.mat'
    matT1Cro = datapathMat+'/'+image[0:idxc]+'hrT1.t1-cropped.mat'
    
    if tag == "T1":
        if not (os.path.exists(fileCro) and os.path.exists(matCroT1)):
            naf.doCropping(fileRenucorr, fileCro, matCroT1)
        else:
            print('cropping already done for T1', fileCro, '\n')
        
    if tag == "T2":
        matCroT2 = datapathMat+'/'+image[0:idx]+'.cropped-t2.mat'
        matT2Cro = datapathMat+'/'+image[0:idx]+'.t2-cropped.mat'
        if not (os.path.exists(fileCro) and os.path.exists(matCroT2)):
            naf.doCropping(fileRenucorr, fileCro, matCroT2)
        else:
            print('cropping already done for T2', fileCro, '\n')
    elif tag == "FLAIR":
        matCroFlair = datapathMat+'/'+image[0:idx]+'.cropped-flair.mat'
        matFlairCro = datapathMat+'/'+image[0:idx]+'.flair-cropped.mat'
        if not (os.path.exists(fileCro) and os.path.exists(matCroFlair)):
            naf.doCropping(fileRenucorr, fileCro, matCroFlair)
        else:
            print('cropping already done for FLAIR', fileCro, '\n')
    elif tag == "PD":
        matCroPD = datapathMat+'/'+image[0:idx]+'.cropped-pd.mat'
        matPDCro = datapathMat+'/'+image[0:idx]+'.pd-cropped.mat'
        if not (os.path.exists(fileCro) and os.path.exists(matCroPD)):
            naf.doCropping(fileRenucorr, fileCro, matCroPD)
        else:
            print('cropping already done for PD', fileCro, '\n')
             
    if tag == "T1":
        if not (os.path.exists(matT1Cro) and os.path.exists(filecropped)):
            naf.doInverseXFM(matCroT1, matT1Cro)
            naf.doApplyXFM(fileReo, matT1Cro, fileCro, filecropped, 'spline', tag)
        else:
            print('inverse transformation matrix for cropping already computed and applied for T1', matT1Cro, '\n')
            
    if tag == "T2":
        if not (os.path.exists(matT2Cro) and os.path.exists(filecropped)):
            naf.doInverseXFM(matCroT2, matT2Cro)
            naf.doApplyXFM(fileReo, matT2Cro, fileCro, filecropped, 'spline', tag)
        else:
            print('inverse transformation matrix for cropping already computed and applied for T2', matT2Cro, '\n')
            
    if tag == "FLAIR":
        if not (os.path.exists(matFlairCro) and os.path.exists(filecropped)):
            naf.doInverseXFM(matCroFlair, matFlairCro)
            naf.doApplyXFM(fileReo, matFlairCro, fileCro, filecropped, 'spline', tag)
        else:
            print('inverse transformation matrix for cropping already computed and applied for FLAIR', matFlairCro, '\n')
    
    if tag == "PD":
        if not (os.path.exists(matPDCro) and os.path.exists(filecropped)):
            naf.doInverseXFM(matCroPD, matPDCro)
            naf.doApplyXFM(fileReo, matPDCro, fileCro, filecropped, 'spline', tag)
        else:
            print('inverse transformation matrix for cropping already computed and applied for PD', matPDCro, '\n')
                   
    # 3. Bias-field Correction using N3 (FS)/N4 (ANTS) after cropping 
    filenucorr = datapath+'/'+image[0:idx]+'.reoriented.cropped.nu_corr.nii'
    if not os.path.exists(filenucorr):
        if nu_corr == 'N3':
            naf.doN3BiasFieldCorrection(filecropped, filenucorr)
        elif nu_corr == 'N4':
            naf.doRemoveNegativeValues(filecropped) # removes negative values if any for N4
            naf.doN4BiasFieldCorrection(filecropped, filenucorr)
    else:
        print('bias-correction already done', filenucorr, '\n')
    
    # 4. Rigid and Affine Transformation of brains using FLIRT in FSL
    matCroppedtoMNI = datapathMat+'/'+image[0:idxc]+'hrT1.cropped-mni.mat'
    matMNItoCropped = datapathMat+'/'+image[0:idxc]+'hrT1.mni-cropped.mat'
    matT1toMNI = datapathMat+'/'+image[0:idxc]+'hrT1.t1-mni.mat'
    matMNItoT1 = datapathMat+'/'+image[0:idxc]+'hrT1.mni-t1.mat'
    
    matCroppedtoAlign = datapathMat+'/'+image[0:idxc]+'hrT1.cropped-align.mat'
    matAligntoCropped = datapathMat+'/'+image[0:idxc]+'hrT1.align-cropped.mat'
    matT1toAlign = datapathMat+'/'+image[0:idxc]+'hrT1.t1-align.mat'
    matAligntoT1 = datapathMat+'/'+image[0:idxc]+'hrT1.align-t1.mat'
    
    filemni = datapathMni+'/'+image[0:idx]+'.reoriented.cropped.nu_corr.mni.nii'
    filemnibrain = datapathMni+'/'+image[0:idx]+'.reoriented.cropped.nu_corr.mni.brain.nii'
    fileRawmnibrain = datapathMni+'/'+image[0:idx]+'.reoriented.mni.brain.nii'
    filenucorrmnibrain = datapathMni+'/'+image[0:idx]+'.reoriented.nu_corr.mni.brain.nii'
    filebrainRaw = datapath+'/'+image[0:idx]+'.reoriented.brain.nii'
    filebrain = datapath+'/'+image[0:idx]+'.reoriented.cropped.nu_corr.brain.nii'
    filebrain1 = datapath+'/'+image[0:idx]+'.reoriented.nu_corr.cropped.brain.nii'
    
    maskfilemni = datapathMni+'/'+image[0:idx]+'.reoriented.cropped.nu_corr.mni.brain.mask.nii'
    maskfileRawmni = datapathMni+'/'+image[0:idx]+'.reoriented.mni.brain.mask.nii'
    maskfileRawalign = datapathAlign+'/'+image[0:idx]+'.reoriented.align.brain.mask.nii'
    maskfilenucorrmni = datapathMni+'/'+image[0:idx]+'.reoriented.nu_corr.mni.brain.mask.nii'
    maskfilealign = datapathAlign+'/'+image[0:idx]+'.reoriented.nu_corr.align.brain.mask.nii'
    maskfile = datapath+'/'+image[0:idx]+'.reoriented.cropped.nu_corr.brain.mask.nii'
    maskfile1 = datapath+'/'+image[0:idx]+'.reoriented.nu_corr.cropped.brain.mask.nii'
    maskfileRaw = datapath+'/'+image[0:idx]+'.reoriented.brain.mask.nii'
        
    filebrainalign = datapathAlign+'/'+image[0:idx]+'.reoriented.cropped.nu_corr.brain.align.nii'
    filealignbrain = datapathAlign+'/'+image[0:idx]+'.reoriented.nu_corr.align.brain.nii'
    fileRenucorrAlign = datapathAlign+'/'+image[0:idx]+'.reoriented.nu_corr.align.nii'
    fileRawAlign = datapathAlign+'/'+image[0:idx]+'.reoriented.align.nii'
    fileRawAlignbrain = datapathAlign+'/'+image[0:idx]+'.reoriented.align.brain.nii'
    
    fileRenucorrMNI = datapathMni+'/'+image[0:idx]+'.reoriented.nu_corr.mni.nii'
    fileRawMNI = datapathMni+'/'+image[0:idx]+'.reoriented.mni.nii'
    
    if tag in {"T2", "PD", "FLAIR"}:
        anatfiles = os.listdir(datapath)
        for reffile in anatfiles:
            if reffile.endswith('nu_corr.brain.nii') and 'hrT1' in reffile:
                reference = datapath+'/'+reffile # T2, PD, FLAIR will be co-registered to the corresponding T1
        
    if tag == "T1":
        if os.path.exists(filemni) and os.path.exists(matCroppedtoMNI):
            print('affine registration of T1 already done', filemni, '\n')
        else:
            naf.doFLIRT(filenucorr, ref, filemni, matCroppedtoMNI, 12, 'corratio', 'spline', tag) # Cropped T1 to MNI
            
        if os.path.exists(matT1toMNI):
            print('transformation matrix from T1 to MNI already computed', matT1toMNI, '\n')
        else:
            naf.doConcatXFM(matT1Cro, matCroppedtoMNI, matT1toMNI) # T1 to MNI
            
        if os.path.exists(matMNItoCropped):
            print('transformation matrix from MNI to T1 Cropped already computed', matMNItoCropped, '\n')
        else:
            naf.doInverseXFM(matCroppedtoMNI, matMNItoCropped) # MNI to Cropped T1
            
        if os.path.exists(matMNItoT1):
            print('transformation matrix from MNI to T1 already computed', matMNItoT1, '\n')
        else:
            naf.doInverseXFM(matT1toMNI, matMNItoT1) # MNI to T1
            
        if os.path.exists(filemnibrain) and os.path.exists(maskfilemni):
            print('brain extraction of MNI T1 already done', filemnibrain, '\n')
        else:   
            naf.doBrainExtraction(filemni, maskfilemni, filemnibrain, 0.4) # doing brain extraction in MNI
            
        if os.path.exists(maskfile):
            print('brain mask in native space is already computed', maskfile, '\n')
        else:
            naf.doApplyXFM(maskfilemni, matMNItoCropped, filenucorr, maskfile, 'nearestneighbour', tag) # creating brain mask in cropped native space
            
        if os.path.exists(filebrain):
            print('masking in native space after cropping and bias-correction already finished', filebrain, '\n')
        else:
            naf.doApplyMasking(filenucorr, maskfile, filebrain) # doing masking in native space after cropping and bias correction
            
        if os.path.exists(maskfileRaw):
            print('brain mask in native space already finished', maskfileRaw, '\n')
        else:
            naf.doApplyXFM(maskfilemni, matMNItoT1, fileRaw, maskfileRaw, 'nearestneighbour', tag) # creating brain mask in native space
            
        if os.path.exists(filebrainRaw):
            print('masking in native space already finished', filebrainRaw, '\n')
        else:
            naf.doApplyMasking(fileRaw, maskfileRaw, filebrainRaw) # doing masking in native space
        
        if os.path.exists(fileRenucorrMNI):
            print('cropped and bias-corrected in mni is already generated', fileRenucorrMNI, '\n')
        else:
            naf.doApplyXFM(filenucorr, matCroppedtoMNI, ref, fileRenucorrMNI, 'spline', tag) # doing bias-corrected mni transformation from cropped.nu_corr
        do_registration_quality(ref, fileRenucorrMNI, 'nmi') # checking quality of registration
        
        if os.path.exists(filenucorrmnibrain):
            print('cropped and bias-corrected brain extraction already finished', filenucorrmnibrain, '\n')
        else:
            naf.doBrainExtraction(fileRenucorrMNI, maskfilenucorrmni, filenucorrmnibrain, 0.4) # doing brain extraction of nu_corr in mni
            os.rename(maskfilenucorrmni, maskfileRawmni) 
            
        if os.path.exists(fileRawMNI):
            print('transformation to MNI space already finished', fileRawMNI, '\n')
        else:
            naf.doApplyXFM(fileReo, matT1toMNI, ref, fileRawMNI, 'spline', tag) # transforming raw image to mni
        do_registration_quality(ref, fileRawMNI, 'nmi') # checking quality of registration
        
        if os.path.exists(fileRawmnibrain):
            print('masking in mni space for raw image already finished', fileRawmnibrain, '\n')
        else:
            naf.doApplyMasking(fileRawMNI, maskfileRawmni, fileRawmnibrain) # doing raw brain masking in mni space
            
        if os.path.exists(matCroppedtoAlign):
             print('rigid transformation of T1 already done', filebrainalign, '\n')
        else:
             naf.doFLIRT(filebrain, refbrain, filebrainalign, matCroppedtoAlign, 6, 'corratio', 'spline', tag) # T1 brain to align
             if True:
                 os.remove(filebrainalign) # Removing the brain aligned to MNI brain
        
        if os.path.exists(matT1toAlign):
            print('concat already', matT1toAlign, '\n')
        else:
            naf.doConcatXFM(matT1Cro, matCroppedtoAlign, matT1toAlign) # Cropped T1 to Align
             
        if os.path.exists(fileRenucorrAlign):
            print('trasformation of bias-corrected T1 to align already done', fileRenucorrAlign, '\n')
        else:
            naf.doApplyXFM(fileRenucorr, matT1toAlign, ref, fileRenucorrAlign, 'spline', tag) # Nucorr to Align
        do_registration_quality(ref, fileRenucorrAlign, 'nmi') # checking quality of registration

        if os.path.exists(fileRawAlign):
            print('trasformation of raw T1 to align already done', fileRawAlign, '\n')
        else:
             naf.doApplyXFM(fileRaw, matT1toAlign, ref, fileRawAlign, 'spline', tag) # Original to Align
        do_registration_quality(ref, fileRawAlign, 'nmi') # checking quality of registration

        if os.path.exists(filealignbrain):
            print('brain extraction in align space is already done', filealignbrain, '\n')
        else:
            naf.doBrainExtraction(fileRenucorrAlign, maskfilealign, filealignbrain, 0.4) # Doing brain extraction in nucorr align space
            os.rename(maskfilealign, maskfileRawalign)
            
        if os.path.exists(fileRawAlignbrain):
            print('brain extraction of raw brain in align already done', fileRawAlignbrain, '\n')
        else:
            naf.doApplyMasking(fileRawAlign, maskfileRawalign, fileRawAlignbrain) # masking in Raw align space
    elif tag in {"T2"}:
        tag1 = tag.lower()
        fileT2toT1 = datapath+'/'+image[0:idx]+'.reoriented.cropped.nu_corr.brain.alignedToT1.nii'
        matT2toT1 = datapathMat+'/'+image[0:idx]+'.t2-t1.mat'
        matT2croppedtoT1 = datapathMat+'/'+image[0:idx]+'.t2_cropped-t1.mat'
        matT2croppedtoT1cropped = datapathMat+'/'+image[0:idx]+'.t2_cropped-t1_cropped.mat'
        matT2toMNI = datapathMat+'/'+image[0:idx]+'.t2-mni.mat'
        matT2toAlign = datapathMat+'/'+image[0:idx]+'.t2-align.mat'
        matT2croppedtoMNI = datapathMat+'/'+image[0:idx]+'.t2_cropped-mni.mat'
        matT2croppedtoAlign = datapathMat+'/'+image[0:idx]+'.t2_cropped-align.mat'
        
        if os.path.exists(filebrain1):
            print('T2 brain extracted from bias-corredted and cropped', filebrain1, '\n')
        else:
            naf.doBrainExtraction(fileCro, maskfile1, filebrain1, 0.2) # Brain extraction of T2 nu_corr.cropped
            os.remove(maskfile1)
        
        if os.path.exists(filebrain):
            print('T2 brain extracted from cropped and bias-corredted', filebrain, '\n')
        else:
            naf.doBrainExtraction(filenucorr, maskfile, filebrain, 0.2) # Brain extraction of T2 cropped.nu_corr
            os.remove(maskfile)
        
        if os.path.exists(fileT2toT1) and os.path.exists(matT2croppedtoT1cropped):
            print('T2 already aligned with T1\n')
        else:
            naf.doFLIRT(filebrain, reference, fileT2toT1, matT2croppedtoT1cropped, 6, 'normmi', 'spline', tag) # T2 to T1
            do_registration_quality(reference, fileT2toT1, 'nmi') # checking quality of registration

        if os.path.exists(matT2croppedtoT1):
            print('T2 cropped to T1 matrix already computed', matT2croppedtoT1, '\n')
        else:
            naf.doConcatXFM(matT2croppedtoT1cropped, matCroT1, matT2croppedtoT1) # T2 cropped to T1
            
        if os.path.exists(matT2toT1):
            print('T2 to T1 matrix already computed', matT2toT1, '\n')
        else:
            naf.doConcatXFM(matT2Cro, matT2croppedtoT1, matT2toT1) # T2 to T1
        
        if os.path.exists(matT2toAlign):
            print('T2 to MNI matrix already computed', matT2toAlign)
        else:
            naf.doConcatXFM(matT2toT1, matT1toAlign, matT2toAlign) # T2 to Align
            
        if os.path.exists(matT2toMNI):
            print('T2 to MNI matrix already computed', matT2toMNI)
        else:
            naf.doConcatXFM(matT2toT1, matT1toMNI, matT2toMNI) # T2 to MNI
        
        if os.path.exists(matT2croppedtoAlign):
            print('T2 to Align matrix already computed', matT2croppedtoAlign)
        else:
            naf.doConcatXFM(matT2croppedtoT1, matT1toAlign, matT2croppedtoAlign) # T2 cropped to Align
            
        if os.path.exists(matT2croppedtoMNI):
            print('T2 to MNI matrix already computed', matT2croppedtoMNI)
        else:
            naf.doConcatXFM(matT2croppedtoT1, matT1toMNI, matT2croppedtoMNI) # T2 cropped to MNI
            
        if os.path.exists(fileRenucorrAlign):
            print('bias-corrected T2 image transformed to align', fileRenucorrAlign, '\n')
        else:
            naf.doApplyXFM(fileRenucorr, matT2toAlign, ref, fileRenucorrAlign, 'spline', tag) # bias-corrected T2 to Align
            do_registration_quality(ref, fileRenucorrAlign, 'nmi') # checking quality of registration

        if os.path.exists(fileRawAlign):
            print('Raw T2 image transformed to align', fileRawAlign, '\n')
        else:
            naf.doApplyXFM(fileReo, matT2toAlign, ref, fileRawAlign, 'spline', tag) # raw T2 to Align
            do_registration_quality(ref, fileRawAlign, 'nmi') # checking quality of registration

        if os.path.exists(fileRenucorrMNI):
            print('bias-corrected T2 image transformed to mni', fileRenucorrMNI, '\n')
        else:
            naf.doApplyXFM(fileRenucorr, matT2toMNI, ref, fileRenucorrMNI, 'spline', tag) # bias-corrected T2 to MNI
            do_registration_quality(ref, fileRenucorrMNI, 'nmi') # checking quality of registration

        if os.path.exists(fileRawMNI):
            print('Raw T2 image transformed to align', fileRawMNI, '\n')
        else:
            naf.doApplyXFM(fileReo, matT2toMNI, ref, fileRawMNI, 'spline', tag) # raw T2 to MNI
            do_registration_quality(ref, fileRawMNI, 'nmi') # checking quality of registration

    elif tag == "FLAIR":
        fileFlairtoT1 = datapath+'/'+image[0:idx]+'.reoriented.cropped.nu_corr.brain.alignedToT1.nii'
        matFlairtoT1 = datapathMat+'/'+image[0:idx]+'.flair-t1.mat'
        matFlaircroppedtoT1 = datapathMat+'/'+image[0:idx]+'.flair_cropped-t1.mat'
        matFlaircroppedtoT1cropped = datapathMat+'/'+image[0:idx]+'.flair_cropped-t1_cropped.mat'
        matFlairtoMNI = datapathMat+'/'+image[0:idx]+'.flair-mni.mat'
        matFlairtoAlign = datapathMat+'/'+image[0:idx]+'.flair-align.mat'
        matFlaircroppedtoMNI = datapathMat+'/'+image[0:idx]+'.flair_cropped-mni.mat'
        matFlaircroppedtoAlign = datapathMat+'/'+image[0:idx]+'.flair_cropped-align.mat'
        
        if os.path.exists(filebrain1):
            print('FLAIR brain extracted from bias-corredted and cropped', filebrain1, '\n')
        else:
            naf.doBrainExtraction(fileCro, maskfile1, filebrain1, 0.4) # Brain extraction of FLAIR nu_corr.cropped
            os.remove(maskfile1)
        
        if os.path.exists(filebrain):
            print('FLAIR brain extracted from cropped and bias-corredted', filebrain, '\n')
        else:
            naf.doBrainExtraction(filenucorr, maskfile, filebrain, 0.4) # Brain extraction of FLAIR cropped.nu_corr
            os.remove(maskfile)
        
        if os.path.exists(fileFlairtoT1) and os.path.exists(matFlaircroppedtoT1cropped):
            print('FLAIR already aligned with T1\n')
        else:
            naf.doFLIRT(filebrain, reference, fileFlairtoT1, matFlaircroppedtoT1cropped, 6, 'normmi', 'spline', tag) # FLAIR to T1
            do_registration_quality(reference, fileFlairtoT1, 'nmi') # checking quality of registration

        if os.path.exists(matFlaircroppedtoT1):
            print('FLAIR cropped to T1 matrix already computed', matFlaircroppedtoT1, '\n')
        else:
            naf.doConcatXFM(matFlaircroppedtoT1cropped, matCroT1, matFlaircroppedtoT1) # FLAIR cropped to T1
            
        if os.path.exists(matFlairtoT1):
            print('FLAIR to T1 matrix already computed', matFlairtoT1, '\n')
        else:
            naf.doConcatXFM(matFlairCro, matFlaircroppedtoT1, matFlairtoT1) # FLAIR to T1
        
        if os.path.exists(matFlairtoAlign):
            print('FLAIR to Align matrix already computed', matFlairtoAlign, '\n')
        else:
            naf.doConcatXFM(matFlairtoT1, matT1toAlign, matFlairtoAlign) # FLAIR to Align
            
        if os.path.exists(matFlairtoMNI):
            print('FLAIR to MNI matrix already computed', matFlairtoMNI, '\n')
        else:
            naf.doConcatXFM(matFlairtoT1, matT1toMNI, matFlairtoMNI) # FLAIR to MNI
        
        if os.path.exists(matFlaircroppedtoAlign):
            print('FLAIR cropped to align matrix already computed', matFlaircroppedtoAlign, '\n')
        else:
            naf.doConcatXFM(matFlaircroppedtoT1, matT1toAlign, matFlaircroppedtoAlign) # FLAIR cropped to Align
            
        if os.path.exists(matFlaircroppedtoMNI):
            print('FLAIR cropped to MNI matrix already computed', matFlaircroppedtoMNI, '\n')
        else:
            naf.doConcatXFM(matFlaircroppedtoT1, matT1toMNI, matFlaircroppedtoMNI) # FLAIR cropped to MNI
            
        if os.path.exists(fileRenucorrAlign):
            print('bias-corrected FLAIR image transformed to align', fileRenucorrAlign, '\n')
        else:
            naf.doApplyXFM(fileRenucorr, matFlairtoAlign, ref, fileRenucorrAlign, 'spline', tag) # bias-corrected FLAIR to Align
            do_registration_quality(ref, fileRenucorrAlign, 'nmi') # checking quality of registration

        if os.path.exists(fileRawAlign):
            print('Raw FLAIR image transformed to align', fileRawAlign, '\n')
        else:
            naf.doApplyXFM(fileReo, matFlairtoAlign, ref, fileRawAlign, 'spline', tag) # raw FLAIR to Align
            do_registration_quality(ref, fileRawAlign, 'nmi') # checking quality of registration

        if os.path.exists(fileRenucorrMNI):
            print('bias-corrected FLAIR image transformed to mni', fileRenucorrMNI, '\n')
        else:
            naf.doApplyXFM(fileRenucorr, matFlairtoMNI, ref, fileRenucorrMNI, 'spline', tag) # bias-corrected FLAIR to MNI
            do_registration_quality(ref, fileRenucorrMNI, 'nmi') # checking quality of registration

        if os.path.exists(fileRawMNI):
            print('Raw FLAIR image transformed to align', fileRawMNI, '\n')
        else:
            naf.doApplyXFM(fileReo, matFlairtoMNI, ref, fileRawMNI, 'spline', tag) # raw FLAIR to MNI
            do_registration_quality(ref, fileRawMNI, 'nmi') # checking quality of registration
            
    elif tag == "PD":
        filePDtoT1 = datapath+'/'+image[0:idx]+'.reoriented.cropped.nu_corr.brain.alignedToT1.nii'
        matPDtoT1 = datapathMat+'/'+image[0:idx]+'.pd-t1.mat'
        matPDcroppedtoT1 = datapathMat+'/'+image[0:idx]+'.pd_cropped-t1.mat'
        matPDcroppedtoT1cropped = datapathMat+'/'+image[0:idx]+'.pd_cropped-t1_cropped.mat'
        matPDtoMNI = datapathMat+'/'+image[0:idx]+'.pd-mni.mat'
        matPDtoAlign = datapathMat+'/'+image[0:idx]+'.pd-align.mat'
        matPDcroppedtoMNI = datapathMat+'/'+image[0:idx]+'.pd_cropped-mni.mat'
        matPDcroppedtoAlign = datapathMat+'/'+image[0:idx]+'.pd_cropped-align.mat'
        
        if os.path.exists(filebrain1):
            print('PD brain extracted from bias-corredted and cropped', filebrain1, '\n')
        else:
            naf.doBrainExtraction(fileCro, maskfile1, filebrain1, 0.2) # Brain extraction of PD nu_corr.cropped
            os.remove(maskfile1)
        
        if os.path.exists(filebrain):
            print('PD brain extracted from cropped and bias-corredted', filebrain, '\n')
        else:
            naf.doBrainExtraction(filenucorr, maskfile, filebrain, 0.2) # Brain extraction of PD cropped.nu_corr
            os.remove(maskfile)
        
        if os.path.exists(filePDtoT1) and os.path.exists(matPDcroppedtoT1cropped):
            print('PD already aligned with T1\n')
        else:
            naf.doFLIRT(filebrain, reference, filePDtoT1, matPDcroppedtoT1cropped, 6, 'normmi', 'spline', tag) # PD to T1
            do_registration_quality(reference, filePDtoT1, 'nmi') # checking quality of registration

        if os.path.exists(matPDcroppedtoT1):
            print('PD cropped to T1 matrix already computed', matPDcroppedtoT1, '\n')
        else:
            naf.doConcatXFM(matPDcroppedtoT1cropped, matCroT1, matPDcroppedtoT1) # PD cropped to T1
            
        if os.path.exists(matPDtoT1):
            print('PD to T1 matrix already computed', matPDtoT1, '\n')
        else:
            naf.doConcatXFM(matPDCro, matPDcroppedtoT1, matPDtoT1) # PD to T1
        
        if os.path.exists(matPDtoAlign):
            print('PD to Align matrix already computed', matPDtoAlign, '\n')
        else:
            naf.doConcatXFM(matPDtoT1, matT1toAlign, matPDtoAlign) # PD to Align
            
        if os.path.exists(matPDtoMNI):
            print('PD to MNI matrix already computed', matPDtoMNI, '\n')
        else:
            naf.doConcatXFM(matPDtoT1, matT1toMNI, matPDtoMNI) # PD to MNI
        
        if os.path.exists(matPDcroppedtoAlign):
            print('PD cropped to align matrix already computed', matPDcroppedtoAlign, '\n')
        else:
            naf.doConcatXFM(matPDcroppedtoT1, matT1toAlign, matPDcroppedtoAlign) # PD cropped to Align
            
        if os.path.exists(matPDcroppedtoMNI):
            print('PD cropped to MNI matrix already computed', matPDcroppedtoMNI, '\n')
        else:
            naf.doConcatXFM(matPDcroppedtoT1, matT1toMNI, matPDcroppedtoMNI) # PD cropped to MNI
            
        if os.path.exists(fileRenucorrAlign):
            print('bias-corrected PD image transformed to align', fileRenucorrAlign, '\n')
        else:
            naf.doApplyXFM(fileRenucorr, matPDtoAlign, ref, fileRenucorrAlign, 'spline', tag) # bias-corrected PD to Align
            do_registration_quality(ref, fileRenucorrAlign, 'nmi') # checking quality of registration

        if os.path.exists(fileRawAlign):
            print('Raw PD image transformed to align', fileRawAlign, '\n')
        else:
            naf.doApplyXFM(fileReo, matPDtoAlign, ref, fileRawAlign, 'spline', tag) # raw PD to Align
            do_registration_quality(ref, fileRawAlign, 'nmi') # checking quality of registration

        if os.path.exists(fileRenucorrMNI):
            print('bias-corrected PD image transformed to mni', fileRenucorrMNI, '\n')
        else:
            naf.doApplyXFM(fileRenucorr, matPDtoMNI, ref, fileRenucorrMNI, 'spline', tag) # bias-corrected PD to MNI
            do_registration_quality(ref, fileRenucorrMNI, 'nmi') # checking quality of registration

        if os.path.exists(fileRawMNI):
            print('Raw PD image transformed to align', fileRawMNI, '\n')
        else:
            naf.doApplyXFM(fileReo, matPDtoMNI, ref, fileRawMNI, 'spline', tag) # raw PD to MNI
            do_registration_quality(ref, fileRawMNI, 'nmi') # checking quality of registration

    else: 
        print('Incorrect tag is chosen for FLIRT\n')