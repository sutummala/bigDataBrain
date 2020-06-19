
# Code created by Sudhakar on April 2020
# FS recon-all including hippocampal subfileds

import os
from nipype.interfaces import freesurfer

def doCheckforT2andFLAIR(path):
    '''
    Parameters
    ----------
    path : str
        path to the input image.

    Returns
    -------
    isT2 : boolean
        returns true if a T2 image is present.
    isFLAIR : boolean
        returns true if a FLAIR image is present.
    '''
    images = os.listdir(path)
    isT2 = False
    isFLAIR = False
    for image in images:
        if 'hrT2' in image:
            isT2 = True
        elif 'hrFLAIR' in image:
            isFLAIR = True
    return isT2, isFLAIR

def fs_Processing(data_dir, subject, tag):
    '''
    Parameters
    ----------
    data_dir : str
        path to the subject directory.
    subject : str
        subject id.
    tag : str
        containing the image type.

    Returns
    -------
    all freesufer files at FS_Subjects_DIR
    '''
    datapath = data_dir+'/'+subject+'/anat' # raw image
    datapathAlign = data_dir+'/'+subject+'/align'# rigid transformed
    if os.path.exists(datapath) and os.listdir(datapath):
        print('doing FS recon-all and hippocampal subfields...')
        reconall = freesurfer.ReconAll()
        reconall.inputs.subject_id = subject # Subject folder
        reconall.inputs.directive = 'all'
        FS_Subjects_DIR = "/usr/users/nmri/projects/tummala/FS_Subjects_DIR"
        if not os.path.exists(FS_Subjects_DIR):
            os.mkdir(FS_Subjects_DIR)
        if tag == 'align':
            if os.path.exists(datapathAlign) and os.listdir(datapathAlign):
                print('doing recon-all on aligned images\n...')
                FS_Subjects_Align = FS_Subjects_DIR+'/'+'FS_Subjects_Align'
                if not os.path.exists(FS_Subjects_Align):
                    os.mkdir(FS_Subjects_Align)    
                reconall.inputs.subjects_dir =  FS_Subjects_Align # Path to freesurfer subjects directory
                alignimages = os.listdir(datapathAlign)
                isT2, isFLAIR = doCheckforT2andFLAIR(datapathAlign)
                for alignimage in alignimages:
                    if alignimage.endswith('reoriented.align.nii') and 'hrT1' in alignimage:
                        reconall.inputs.T1_files = datapathAlign+'/'+alignimage
                    elif alignimage.endswith('reoriented.align.nii') and (isT2 and not isFLAIR):
                        reconall.inputs.T2_file = datapathAlign+'/'+alignimage
                        reconall.inputs.use_T2 = True
                    elif alignimage.endswith('reoriented.align.nii') and isFLAIR:
                        reconall.inputs.FLAIR_file = datapathAlign+'/'+alignimage
                        reconall.inputs.use_FLAIR = True
            else:
                print(f'no align directory/no files in the align directory for {subject}\n')
        elif tag == 'anat':
            print('doing recon-all on anat images\n...')
            FS_Subjects_Anat = FS_Subjects_DIR+'/'+'FS_Subjects_Anat'
            if not os.path.exists(FS_Subjects_Anat):
                os.mkdir(FS_Subjects_Anat)    
            reconall.inputs.subjects_dir =  FS_Subjects_Anat # Path to freesurfer subjects directory
            anatimages = os.listdir(datapath)
            isT2, isFLAIR = doCheckforT2andFLAIR(datapath)
            for anatimage in anatimages:
                if anatimage.endswith('reoriented.nii') and 'hrT1' in anatimage:
                    reconall.inputs.T1_files = datapath+'/'+anatimage
                elif anatimage.endswith('reoriented.nii') and (isT2 and not isFLAIR):
                    reconall.inputs.T2_file = datapath+'/'+anatimage
                    reconall.inputs.use_T2 = True
                elif anatimage.endswith('reoriented.nii') and isFLAIR:
                    reconall.inputs.FLAIR_file = datapath+'/'+anatimage
                    reconall.inputs.use_FLAIR = True
    
        reconall.inputs.hippocampal_subfields_T1 = True
        reconall.run() # running the recon-all
    else:
        print(f'no anat directory/no files in anat directory for {subject}\n')