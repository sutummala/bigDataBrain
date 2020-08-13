The code is brain structural MRI processing (such as T1-weighted, T2-weighted, FLAIR) pipeline using nipype. It also have quality check at different stages of pre and post processing such as initial image quality check, registration quality (rigid, affine and non-linear) check and quality of segmentations (gray matter, white matter and csf).

Several processing interfaces from nipype such as FSL, FreeSurfer, SPM, ANTS and etc. are used for pre-processing. For quality check (QC), python code was developed for different categories of QC.

Code also involves, renaming different files for consistency across different repositories such IXI, HCP and others by following BIDS guidelines.
