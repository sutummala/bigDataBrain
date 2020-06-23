# Code created by Sudhakar on June 2020
# code to convert an affine matrix to rigid matrix

import os
import decompose_compose_trans_matrix as dctm
import numpy as np
import nipype_all_functions as naf

refpath = "/usr/users/nmri/tools/fsl/6.0.3/data/standard" # FSL template
ref = refpath+'/MNI152_T1_1mm.nii.gz' # Whole brain MNI 

sub_path = '/usr/users/tummala/HCP-YA-test/127226'
mat_path = sub_path+'/mat'
img_path = sub_path+'/anat'
align_path = sub_path+'/align'
mni_path = sub_path+'/mni'

def modify_rots(mat):
    mat_new = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i==j:
                mat_new[i,j] = mat[i,j]
    return mat_new

for matfile in os.listdir(mat_path):
    if matfile.endswith('t1-mni.mat'):
        mat_orig = np.loadtxt(mat_path+'/'+matfile)
        print(f'original matrix: {mat_orig}\n')
        scales, trans, rots = dctm.decompose(mat_orig, angles = False)
        print(f'scales: {scales}')
        print(f'trans: {trans}')
        print(f'rots: {rots}')
        scales_new = np.ones(3)
        trans_new = trans
        rots_new = modify_rots(rots)
        print(f'new scales: {scales_new}')
        print(f'new trans: {trans_new}')
        print(f'new rots: {rots_new}')
        mat_modified = dctm.compose(scales_new, trans_new, rots_new)
        matfile_new = matfile[0:-4]+'.test.mat'
        np.savetxt(mat_path+'/'+matfile_new, mat_modified, fmt = '%10.19f')
        print(f'modified matrix: {mat_modified}\n')
        for imgfile in os.listdir(img_path):
            if imgfile.endswith('reoriented.nii') and 'hrT1' in imgfile:
                outfile = imgfile[0:-4]+'.align_from_affine.nii'
                if 0:
                    naf.doApplyXFM(img_path+'/'+imgfile, mat_path+'/'+matfile_new, ref, align_path+'/'+outfile, 'spline', 'hrT1')
        
        