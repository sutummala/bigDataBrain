
import os
import numpy as np
import nipype_preprocessing_structural as nps
refpath = "/usr/users/nmri/tools/fsl/6.0.3/data/standard" # FSL template for affine registration

def preprocessing_main(data_dir, subject):
    datapath = data_dir+'/'+subject+'/anat' # raw image
    datapathAlign = data_dir+'/'+subject+'/align'# rigid transformed
    datapathAlignBB = data_dir+'/'+subject+'/align_bb'# rigid transformed with big bounding box
    datapathMat = data_dir+'/'+subject+'/mat' # all transformation matrices
    datapathMni = data_dir+'/'+subject+'/mni' # affine transformed
    if os.path.exists(datapath):
        strucImages = os.listdir(datapath)
        nFiles = np.size(strucImages)
        if nFiles == 0:
            print('no files in the ant folder for', subject, 'moving on to the next subject\n')
        else:
            useFirst = False # First Series
            useSecond = False # Second Series
            useAverage = True # Align and average if two series are found
            nu_corr = 'N3' # Bias-correction type (N3: freesurfer, N4: ANTS)
            
            # creating directories if they does not exist already
            if not os.path.exists(datapathAlign):
                os.makedirs(datapathAlign)
                
            if not os.path.exists(datapathAlignBB):
                os.makedirs(datapathAlignBB)
                
            if not os.path.exists(datapathMat):
                os.makedirs(datapathMat)
                
            if not os.path.exists(datapathMni):
                os.makedirs(datapathMni)
                
            # Pre-Processing of T1-weighted Image(s)
            onlyoneSeriesT1 = False 
            firstSeriesT1 = False
            secondSeriesT1 = False
            for image in strucImages:
                if any([image.endswith('hrT1.nii.gz'), image.endswith('hrT1.nii'), image.endswith('hrT1.img')]):
                    imageT1 = image
                    onlyoneSeriesT1 = True
                elif any([image.endswith('hrT1.A.nii.gz'), image.endswith('hrT1.A.nii'), image.endswith('hrT1.A.img')]):
                    imageT1_A = image
                    firstSeriesT1 = True
                elif any([image.endswith('hrT1.B.nii.gz'), image.endswith('hrT1.B.nii'), image.endswith('hrT1.B.img')]):
                    imageT1_B = image
                    secondSeriesT1 = True
                    
            if onlyoneSeriesT1:
                 print('found only one T1 series', imageT1, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageT1, 'T1', 'hrT1', nu_corr)
            elif useFirst and firstSeriesT1:
                 print('found two T1 series, using first one', imageT1_A, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageT1_A, 'T1', 'hrT1.A', nu_corr)
            elif useSecond and secondSeriesT1:
                 print('found two T1 series, using second one', imageT1_B, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageT1_B, 'T1', 'hrT1.B', nu_corr)
            elif (useAverage and firstSeriesT1 and secondSeriesT1):
                 print('found two T1 series, doing alignment and average before processing\n')
                 imageT1_Average = imageT1_A.replace('.A.', '.M.')
                 nps.doAlignAverage(datapath, datapathMat, imageT1_A, imageT1_B, imageT1_Average)
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageT1_Average, 'T1', 'hrT1.M', nu_corr)
                
            
            # Pre-Processing of T2-weighted Image(s)
            onlyoneSeriesT2 = False 
            firstSeriesT2 = False
            secondSeriesT2 = False
            for image in strucImages:
                if any([image.endswith('hrT2.nii.gz'), image.endswith('hrT2.nii'), image.endswith('hrT2.img')]):
                    imageT2 = image
                    onlyoneSeriesT2 = True
                elif any([image.endswith('hrT2.A.nii.gz'), image.endswith('hrT2.A.nii'), image.endswith('hrT2.A.img')]):
                    imageT2_A = image
                    firstSeriesT2 = True
                elif any([image.endswith('hrT2.B.nii.gz'), image.endswith('hrT2.B.nii'), image.endswith('hrT2.B.img')]):
                    imageT2_B = image
                    secondSeriesT2 = True
                    
            if onlyoneSeriesT2:
                 print('found only one T2 series', imageT2, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageT2, 'T2', 'hrT2', nu_corr)
            elif useFirst and firstSeriesT2:
                 print('found two T2 series, using first one', imageT2_A, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageT2_A, 'T2', 'hrT2.A', nu_corr)
            elif useSecond and secondSeriesT2:
                 print('found two T2 series, using second one', imageT2_B, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageT2_B, 'T2', 'hrT2.B', nu_corr)
            elif (useAverage and firstSeriesT2 and secondSeriesT2):
                 print('found two T2 series, doing alignment and average before processing\n')
                 imageT2_Average = imageT2_A.replace('.A.', '.M.')
                 nps.doAlignAverage(datapath, datapathMat, imageT2_A, imageT2_B, imageT2_Average)
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageT2_Average, 'T2', 'hrT2.M', nu_corr)
                   
            
            # Pre-Processing of FLAIR Image(s)
            onlyoneSeriesFLAIR = False 
            firstSeriesFLAIR = False
            secondSeriesFLAIR = False
            for image in strucImages:
                if any([image.endswith('hrFLAIR.nii.gz'), image.endswith('hrFLAIR.nii'), image.endswith('hrFLAIR.img')]):
                    imageFLAIR = image
                    onlyoneSeriesFLAIR = True
                elif any([image.endswith('hrFLAIR.A.nii.gz'), image.endswith('hrFLAIR.A.nii'), image.endswith('hrFLAIR.A.img')]):
                    imageFLAIR_A = image
                    firstSeriesFLAIR = True
                elif any([image.endswith('hrFLAIR.B.nii.gz'), image.endswith('hrFLAIR.B.nii'), image.endswith('hrFLAIR.B.img')]):
                    imageFLAIR_B = image
                    secondSeriesFLAIR = True
                    
            if onlyoneSeriesFLAIR:
                 print('found only one FLAIR series', imageFLAIR, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageFLAIR, 'FLAIR', 'hrFLAIR', nu_corr)
            elif useFirst and firstSeriesFLAIR:
                 print('found two FLAIR series, using first one', imageFLAIR_A, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageFLAIR_A, 'FLAIR', 'hrFLAIR.A', nu_corr)
            elif useSecond and secondSeriesFLAIR:
                 print('found two FLAIR series, using second one', imageFLAIR_B, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageFLAIR_B, 'FLAIR', 'hrFLAIR.B', nu_corr)
            elif (useAverage and firstSeriesFLAIR and secondSeriesFLAIR):
                 print('found two FLAIR series, doing alignment and average before processing\n')
                 imageFLAIR_Average = imageFLAIR_A.replace('.A.', '.M.')
                 nps.doAlignAverage(datapath, datapathMat, imageFLAIR_A, imageFLAIR_B, imageFLAIR_Average)
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imageFLAIR_Average, 'FLAIR', 'hrFLAIR.M', nu_corr)
                 
            # Pre-Processing of PD (Proton-Density) Image(s)
            onlyoneSeriesPD = False 
            firstSeriesPD = False
            secondSeriesPD = False
            for image in strucImages:
                if any([image.endswith('hrPD.nii.gz'), image.endswith('hrPD.nii'), image.endswith('hrPD.img')]):
                    imagePD = image
                    onlyoneSeriesPD = True
                elif any([image.endswith('hrPD.A.nii.gz'), image.endswith('hrPD.A.nii'), image.endswith('hrPD.A.img')]):
                    imagePD_A = image
                    firstSeriesPD = True
                elif any([image.endswith('hrPD.B.nii.gz'), image.endswith('hrPD.B.nii'), image.endswith('hrPD.B.img')]):
                    imagePD_B = image
                    secondSeriesPD = True
                    
            if onlyoneSeriesPD:
                 print('found only one PD series', imagePD, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imagePD, 'PD', 'hrPD', nu_corr)
            elif useFirst and firstSeriesPD:
                 print('found two PD series, using first one', imagePD_A, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imagePD_A, 'PD', 'hrPD.A', nu_corr)
            elif useSecond and secondSeriesPD:
                 print('found two PD series, using second one', imagePD_B, '\n')
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imagePD_B, 'PD', 'hrPD.B', nu_corr)
            elif (useAverage and firstSeriesPD and secondSeriesPD):
                 print('found two PD series, doing alignment and average before processing\n')
                 imagePD_Average = imagePD_A.replace('.A.', '.M.')
                 nps.doAlignAverage(datapath, datapathMat, imagePD_A, imagePD_B, imagePD_Average)
                 nps.preProcessing(datapath, datapathAlign, datapathMat, datapathMni, refpath, imagePD_Average, 'PD', 'hrPD.M', nu_corr)
    else:
        print('no anat folder exist for', subject, 'moving on to the next subject\n')
    
    
