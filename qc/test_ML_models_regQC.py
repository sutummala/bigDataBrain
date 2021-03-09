# Code created by Sudhakar on Feb 2021 to test different ML models for QC of rigid and affine registrations (for both T1 and T2)

import os
import all_plots as ap
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as kNN
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

plt.rcParams.update({'font.size': 22})

subpath1 = '/media/tummala/TUMMALA/Work/Data/IXI-Re'
saved_models = '/home/tummala/mri/ml_classifier_models_checking_reg'

voi_size = 9
step_size = 9 # stride
# subpath3 = '/usr/users/tummala/bigdata'
# subpath4 = '/usr/users/tummala/HCP-YA-test'

def remove_nan(data):
    '''
    Parameters
    ----------
    data : list
        input list, could have nan values.

    Returns
    -------
    type: list
        all nan values from the input will be removed.
    '''
    data = np.array(data)
    return data[~np.isnan(data)]

def get_coreg_cost_vectors(cost_func, subpath, tag):
    '''
    Parameters
    ----------
    cost_func : str
        cost name, ncc:normalized correlation coefficient, nmi: normalized mutual information.
    subpath : str
        path containing all subjects.
    tag : str
        returns a tag to identify the image type.

    Returns
    -------
    type: str
        returns global and local cost vectors for all subjects under study.
    '''
    subjects = os.listdir(subpath)
    global_cost_vector = []
    local_cost_vector = []
    for index, subject in enumerate(subjects, start=1):
        cost_folder = subpath+'/'+subject+'/cost'+str(voi_size)+str(step_size)
        data_files = os.listdir(cost_folder)
        for data_file in data_files:
            if 'alignedToT1' in data_file and (tag in data_file and cost_func in data_file):
                cost_data = np.loadtxt(cost_folder+'/'+data_file)
                global_cost_vector.append(cost_data[0])
                local_cost_vector.append(cost_data[1])
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def get_cost_vectors(cost_func, reg_type, subpath, tag):
    '''
    Parameters
    ----------
    cost_func : str
        cost name, ncc:normalized correlation coefficient, nmi: normalized mutual information.
    reg_type: str
        registration type, align for rigid and mni for affine
    subpath : str
        path containing all subjects.
    tag : str
        returns a tag to identify the image type.

    Returns
    -------
    type: str
        returns global and local cost vectors for all subjects under study.
    '''
    subjects = os.listdir(subpath)
    global_cost_vector = []
    local_cost_vector = []
    for index, subject in enumerate(subjects, start=1):
        cost_folder = subpath+'/'+subject+'/cost'+str(voi_size)+str(step_size)
        #print('{}-{}, {}-{}'.format(index, subject, reg_type, cost_func))
        data_files = os.listdir(cost_folder)
        for data_file in data_files:
            if reg_type in data_file and (tag in data_file and cost_func in data_file):
                if not 'alignedToT1' in data_file:
                    cost_data = np.loadtxt(cost_folder+'/'+data_file)
                    global_cost_vector.append(cost_data[0])
                    local_cost_vector.append(cost_data[1])
    return global_cost_vector, local_cost_vector

def test_reg_quality(models_path, cost_matrix, reg):
    '''
    Parameters
    ----------
    models_path : str
        path to the saved ML models.
    cost_matrix : float
        matrix of cost values, number of rows should be number of samples (images) and number of column should be number of features. if not consider transpose
        reg : str
        registration type, align for rigid and mni for affine

    Returns
    -------
    model predicted probabilities for registration quality (0 is good and 1 is bad).

    '''
    if reg == 'align':
        model = pickle.load(open(os.path.join(models_path, 'ada_boost_alignT1'), 'rb'))
        scale = pickle.load(open(os.path.join(models_path, 'scale_alignT1'), 'rb'))
    else:
        model = pickle.load(open(os.path.join(models_path, 'ada_boost_mniT1'), 'rb'))
        scale = pickle.load(open(os.path.join(models_path, 'scale_mniT1'), 'rb'))
    
    print('saved models were loaded')
    
    data = np.transpose(cost_matrix) # transforming to get features as columns and samples as rows     
        
    data_scaled = scale.transform(data) # scaling features to match with the scaling during training 
    
    p = model.predict_proba(data_scaled)[:, 1]
    
    #print(len(p), p)
    
    for i, subject in enumerate(os.listdir(subpath1)):
        anat_path = os.path.join(subpath1, subject, 'anat')
        for image in os.listdir(anat_path):
            if image.endswith('hrT1.reoriented.nii'):
                if p[i] > 0.5:
                    print(f'Something is wrong with the {reg} registration for {subject} and Quality of registration is {(1-p[i])*100}')
                # else:
                #     print(f'The quality of {reg} registration for {subject} is good and it is {(1-p[i])*100}')
            
    
if __name__ == '__main__':
    
    costs = ['ncc', 'nmi', 'cor']
    reg_types = ['align', 'mni']
    
    local = True

    for reg_type in reg_types:
        combine_cost_vector_T1 = []
        combine_test_cost_vector_T1 = []
        
        combine_cost_vector_T2 = []
        combine_test_cost_vector_T2 = []
        
        combine_cost_vector_FLAIR = []
        combine_test_cost_vector_FLAIR = []
        
        for cost in costs:
            # getting normal values for hrT1, hrT2 and hrFLAIR for bigdata 
            global_cost_vector_bigdata_T1, local_cost_vector_bigdata_T1 = get_cost_vectors(cost, reg_type, subpath1, 'hrT1') # T1 to MNI
            global_cost_vector_bigdata_T2, local_cost_vector_bigdata_T2 = get_cost_vectors(cost, reg_type, subpath1, 'hrT2') # FLAIR to MNI
            
            if local:
                combine_cost_vector_T1.append(local_cost_vector_bigdata_T1) # T1 data
                combine_cost_vector_T2.append(local_cost_vector_bigdata_T2) # T2 data
            else:
                combine_cost_vector_T1.append(global_cost_vector_bigdata_T1) # T1 data
                combine_cost_vector_T2.append(global_cost_vector_bigdata_T2) # T2 data
        
        test_reg_quality(saved_models, combine_cost_vector_T1, reg_type)
        #test_reg_quality(saved_models, combine_cost_vector_T2, reg_type)
            
    
            
            
        
        
        
        
        
