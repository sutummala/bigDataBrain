# Code created by Sudhakar on March 2020
# Processing for structural data (T1, T2 and FLAIR) aquired at 3T

import sys
import nipype_preprocessing_main as npm
import nipype_spm_segmentation as nss
import nipype_freesurfer_processing as nfs

data_dir = sys.argv[1] # Path to the subjects data directory
subject = sys.argv[2] # Subject ID

print(data_dir, subject)
 # pre-processing (cropping, bias-correction followed by rigid, affine transformation to MNI space)
npm.preprocessing_main(data_dir, subject)

 # gray matter, white matter and CSF segmentation using SPM
nss.do_spm_segmentation(data_dir, subject, image_type = 'anat', multi_channel = True)
            
 # Freesurfer processing (cortical segmentation, hippocampal subfields)
#nfs.fsProcessing(data_dir, subject, 'anat') # doing it on anat (raw) images
#nfs.fsProcessing(data_dir, subject, 'align') # doing it on aligned images
        

      
