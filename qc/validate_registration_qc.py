# Code created by Sudhakar on Oct 2020
# Validation of developed ML models for checking the registration 

import os
import pickle
import numpy as np
import registration_cost_function as rcf

refpath = "/home/tummala/mri/tools/fsl/data/standard" # FSL template
data_path = "/home/tummala/data/ABIDE-failed-mni"

ml_models_path = '/home/tummala/mri/ml_classifier_models_checking_reg'

subjects = sorted(os.listdir(data_path))

costs = ['ncc', 'nmi', 'cor']

def validate_reg(data_path, subject, tag):
    
    if tag == 'align':
        rigid_path = os.path.join(data_path, subject, 'align')
        image_tag = 'align.nii'
        # load scale model for rigid
        scale = pickle.load(open(ml_models_path+'/scale_alignT1', 'rb'))
        
        # load classifier model for rigid
        c_rigid = pickle.load(open(ml_models_path+'/lda_alignT1', 'rb'))
    elif tag == 'mni':
        rigid_path = os.path.join(data_path, subject, 'mni')
        image_tag = 'mni.nii'
        # load scale model for affine
        scale = pickle.load(open(ml_models_path+'/scale_mniT1', 'rb'))
        
        # load classifier model for affine
        c_rigid = pickle.load(open(ml_models_path+'/lda_mniT1', 'rb'))
        
    if os.path.exists(rigid_path):
        for image in os.listdir(rigid_path):
            cost_values = []
            if image.endswith(image_tag):
                print(f'validating {image}')
                for cost in costs:
                    _,local_cost = rcf.do_check_registration(refpath, rigid_path+'/'+image, cost, voi_size = 3, step_size = 3, masking = True, measure_global = False, measure_local = True)
                    cost_values.append(local_cost)
                    
                cost_values_scaled = scale.transform(np.reshape(np.array(cost_values), (1,3)))
                if c_rigid.predict(cost_values_scaled):
                    print(f'problem with {tag} registration of {image} for {subject}')
                else:
                    print(f'{tag} registration of {image} for {subject} is fine')


for subject in subjects:
    
    print(f'checking for {subject}\n')
    
    # validating rigid registrations
    validate_reg(data_path, subject, 'align')
    
    # validating affine registrations
    validate_reg(data_path, subject, 'mni')
    
                    
               

        
                    
