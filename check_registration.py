# code created by Sudhakar on May 2020
# generate images and check registration


import os
import all_plots as ap
import numpy as np
from sklearn import metrics

subpath1 = '/usr/users/tummala/bigdata1'
subpath2 = '/usr/users/tummala/HCP-YA'

subpath3 = '/usr/users/tummala/bigdata'
subpath4 = '/usr/users/tummala/HCP-YA-test'

def remove_nan(data): 
    # removes nan values if any
    data = np.array(data)
    return data[~np.isnan(data)]

def get_coreg_cost_vectors(cost_func, subpath, tag):
    subjects = os.listdir(subpath)
    global_cost_vector = []
    local_cost_vector = []
    for index, subject in enumerate(subjects, start=1):
        cost_folder = subpath+'/'+subject+'/cost'
        #print('{}-{}, {}-{}'.format(index, subject, reg_type, cost_func))
        data_files = os.listdir(cost_folder)
        for data_file in data_files:
            if 'alignedToT1' in data_file and (tag in data_file and cost_func in data_file):
                cost_data = np.loadtxt(cost_folder+'/'+data_file)
                global_cost_vector.append(cost_data[0])
                local_cost_vector.append(cost_data[1])
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def get_cost_vectors(cost_func, reg_type, subpath, tag):
    subjects = os.listdir(subpath)
    global_cost_vector = []
    local_cost_vector = []
    for index, subject in enumerate(subjects, start=1):
        cost_folder = subpath+'/'+subject+'/cost'
        #print('{}-{}, {}-{}'.format(index, subject, reg_type, cost_func))
        data_files = os.listdir(cost_folder)
        for data_file in data_files:
            if reg_type in data_file and (tag in data_file and cost_func in data_file):
                if not 'alignedToT1' in data_file:
                    cost_data = np.loadtxt(cost_folder+'/'+data_file)
                    global_cost_vector.append(cost_data[0])
                    local_cost_vector.append(cost_data[1])
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def get_coreg_test_cost_vectors(cost_func, subpath, tag):
    subjects = os.listdir(subpath)
    global_cost_vector = []
    local_cost_vector = []
    for index, subject in enumerate(subjects, start=1):
        if tag == 'hrT2':
            cost_folder = subpath+'/'+subject+'/test_cost_T2_T1'
        elif tag == 'hrFLAIR':
            cost_folder = subpath+'/'+subject+'/test_cost_FLAIR_T1'

        if os.path.exists(cost_folder) and os.listdir(cost_folder):
            data_files = os.listdir(cost_folder)
            for data_file in data_files:
                if ('hrT1' in data_file and cost_func in data_file):
                    #print(reg_type, tag, cost_func)
                    cost_data = np.loadtxt(cost_folder+'/'+data_file)
                    global_cost_vector.append(cost_data[0])
                    local_cost_vector.append(cost_data[1])
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def get_test_cost_vectors(cost_func, reg_type, subpath, tag):
    subjects = os.listdir(subpath)
    global_cost_vector = []
    local_cost_vector = []
    for index, subject in enumerate(subjects, start=1):
        if tag == 'hrT1':
            if reg_type == 'align':
                cost_folder = subpath+'/'+subject+'/test_cost_T1_align'
            elif reg_type == 'mni':
                cost_folder = subpath+'/'+subject+'/test_cost_T1_mni'
        elif tag == 'hrT2':
            if reg_type == 'align':
                cost_folder = subpath+'/'+subject+'/test_cost_T2_align'
            elif reg_type == 'mni':
                cost_folder = subpath+'/'+subject+'/test_cost_T2_mni'
        elif tag == 'hrFLAIR':
            if reg_type == 'align':
                cost_folder = subpath+'/'+subject+'/test_cost_FLAIR_align'
            elif reg_type == 'mni':
                cost_folder = subpath+'/'+subject+'/test_cost_FLAIR_mni'
        
        if os.path.exists(cost_folder) and os.listdir(cost_folder):
            data_files = os.listdir(cost_folder)
            for data_file in data_files:
                if (tag in data_file and cost_func in data_file):
                    #print(reg_type, tag, cost_func)
                    cost_data = np.loadtxt(cost_folder+'/'+data_file)
                    global_cost_vector.append(cost_data[0])
                    local_cost_vector.append(cost_data[1])
    return remove_nan(global_cost_vector), remove_nan(local_cost_vector)

def compute_cutoff_auc(data1, data2, *tags):
    ''' computes cut-off point and AUC for given cost and reg type from data1 (normal vlaues), data2 (test values)'''
       
    labels = np.concatenate([np.ones(len(data1)), np.zeros(len(data2))])
    print(f'{len(data1)}, {len(data2)}')
    fpr, tpr, thresholds = metrics.roc_curve(labels, np.concatenate([data1, data2]), pos_label = 1)
    
    print(f'Threshold for {tags[2]}-{tags[1]}-{tags[3]}-{tags[0]} is: {thresholds[np.argmax(tpr-fpr)]}, AUC is: {metrics.auc(fpr, tpr)}\n')
    

if __name__ == '__main__':
    
    costs = ['ncc', 'nmi']
    reg_types = ['align', 'mni']

    for reg_type in reg_types:
        for cost in costs:
            # getting normal values for hrT1, hrT2 and hrFLAIR for bigdata 
            global_cost_vector_bigdata_T1, local_cost_vector_bigdata_T1 = get_cost_vectors(cost, reg_type, subpath1, 'hrT1') # T1 to MNI
            global_cost_vector_bigdata_FLAIR, local_cost_vector_bigdata_FLAIR = get_cost_vectors(cost, reg_type, subpath1, 'hrFLAIR') # FLAIR to MNI
            global_cost_vector_bigdata_FLAIRtoT1, local_cost_vector_bigdata_FLAIRtoT1 = get_coreg_cost_vectors(cost, subpath1, 'hrFLAIR') # FLAIR brain to T1 brain (only align)
            
            # HCP-YA
            global_cost_vector_hcpya_T1, local_cost_vector_hcpya_T1 = get_cost_vectors(cost, reg_type, subpath2, 'hrT1') # T1 to MNI
            global_cost_vector_hcpya_T2, local_cost_vector_hcpya_T2 = get_cost_vectors(cost, reg_type, subpath2, 'hrT2') # T2 to MNI
            global_cost_vector_hcpya_T2toT1, local_cost_vector_hcpya_T2toT1 = get_coreg_cost_vectors(cost, subpath2, 'hrT2') # T2 to T1 (only align)
    
            if False:
                # plotting normal values for T1, T2 and FLAIR
                ap.plot_cost([global_cost_vector_bigdata_T1, global_cost_vector_hcpya_T1, global_cost_vector_hcpya_T2, global_cost_vector_bigdata_FLAIR], cost,
                          ['T1', 'T1(hcp)', 'T2(hcp)', 'FLAIR'], f'global-{reg_type}') # plotting global cost
            if False:
                ap.plot_cost([local_cost_vector_bigdata_T1, local_cost_vector_hcpya_T1, local_cost_vector_hcpya_T2, local_cost_vector_bigdata_FLAIR], cost,
                          ['T1', 'T1(hcp)', 'T2(hcp)', 'FLAIR'], f'local-{reg_type}') # plotting local cost
            
            # getting test values for hrT1, hrT2 and hrFLAIR for bigdata 
            global_test_cost_vector_bigdata_T1, local_test_cost_vector_bigdata_T1 = get_test_cost_vectors(cost, reg_type, subpath1, 'hrT1') # T1 to MNI
            global_test_cost_vector_bigdata_FLAIR, local_test_cost_vector_bigdata_FLAIR = get_test_cost_vectors(cost, reg_type, subpath1, 'hrFLAIR') # FLAIR to MNI
            global_test_cost_vector_bigdata_FLAIRtoT1, local_test_cost_vector_bigdata_FLAIRtoT1 = get_coreg_test_cost_vectors(cost, subpath3, 'hrFLAIR') # FLAIR brain to T1 brain (only align)
            
            # HCPYA
            global_test_cost_vector_hcpya_T1, local_test_cost_vector_hcpya_T1 = get_test_cost_vectors(cost, reg_type, subpath2, 'hrT1') # T1 to MNI
            global_test_cost_vector_hcpya_T2, local_test_cost_vector_hcpya_T2 = get_test_cost_vectors(cost, reg_type, subpath2, 'hrT2') # T2 to MNI
            global_test_cost_vector_hcpya_T2toT1, local_test_cost_vector_hcpya_T2toT1 = get_coreg_test_cost_vectors(cost, subpath4, 'hrT2') # T2 to T1 (only align)
            
            if True:
                ap.plot_cost([local_cost_vector_bigdata_T1, local_test_cost_vector_bigdata_T1], cost,
                          ['T1', 'T1-test', 'T1(local)', 'T1-test(local)'], f'Big-Data {reg_type}') # plotting local cost for bigdata T1
            if True:
                ap.plot_cost([local_cost_vector_bigdata_FLAIR, local_test_cost_vector_bigdata_FLAIR], cost,
                          ['FLAIR', 'FLAIR-test', 'FLAIR(local)', 'FLAIR-test(local)'], f'Big-Data {reg_type}') # plotting local cost for bigdata FLAIR
            if False and reg_type == 'align':
                ap.plot_cost([local_cost_vector_bigdata_FLAIRtoT1, local_test_cost_vector_bigdata_FLAIRtoT1], cost,
                          ['FLAIR-T1', 'FLAIR-T1-test'], f'Big-Data-Align') # plotting local cost for bigdata FLAIR
                
            # Plottinf for HCP-YA 
            if True:    
                ap.plot_cost([local_cost_vector_hcpya_T1, local_test_cost_vector_hcpya_T1], cost,
                          ['T1', 'T1-test', 'T1(local)', 'T1-test(local)'], f'HCP-YA {reg_type}') # plotting local cost for HCPYA T1
            if True:
                ap.plot_cost([local_cost_vector_hcpya_T2, local_test_cost_vector_hcpya_T2], cost,
                          ['T2', 'T2-test', 'T2(local)', 'T2-test(local)'], f'HCP-YA {reg_type}') # plotting local cost for HCPYA T2
            if False and reg_type == 'align':
                ap.plot_cost([local_cost_vector_hcpya_T2toT1, local_test_cost_vector_hcpya_T2toT1], cost,
                          ['T2-T1', 'T2-T1-test'], f'HCP-YA-Align') # plotting local cost for HCPYA T2
            
            # Compute cut-off point and AUC for normal and test
            print('doing for Big Data\n')
            
            #compute_cutoff_auc(global_cost_vector_bigdata_T1, global_test_cost_vector_bigdata_T1, cost, reg_type, 'hrT1', 'global') # T1 to MNI
            compute_cutoff_auc(local_cost_vector_bigdata_T1, local_test_cost_vector_bigdata_T1, cost, reg_type, 'hrT1', 'local') # T1 to MNI
            print('----------------------------------------------------------------------------------------------------')
            
            #compute_cutoff_auc(global_cost_vector_bigdata_FLAIR, global_test_cost_vector_bigdata_FLAIR, cost, reg_type, 'hrFLAIR', 'global') # FLAIR to MNI
            compute_cutoff_auc(local_cost_vector_bigdata_FLAIR, local_test_cost_vector_bigdata_FLAIR, cost, reg_type, 'hrFLAIR', 'local') # FLAIR to MNI
            print('----------------------------------------------------------------------------------------------------')
            
            if reg_type == 'align':
                #compute_cutoff_auc(global_cost_vector_bigdata_FLAIRtoT1, global_test_cost_vector_bigdata_FLAIRtoT1, cost, 'T1', 'hrFLAIR', 'global') # FLAIR brain to T1 brain
                compute_cutoff_auc(local_cost_vector_bigdata_FLAIRtoT1, local_test_cost_vector_bigdata_FLAIRtoT1, cost, 'T1', 'hrFLAIR', 'local') # FLAIR brain to T1 brain
                print('----------------------------------------------------------------------------------------------------')
            
            print('doing for HCP-YA\n')
            
            #compute_cutoff_auc(global_cost_vector_hcpya_T1, global_test_cost_vector_hcpya_T1, cost, reg_type, 'hrT1', 'global') # T1 to MNI
            compute_cutoff_auc(local_cost_vector_hcpya_T1, local_test_cost_vector_hcpya_T1, cost, reg_type, 'hrT1', 'local') # T1 to MNI
            print('----------------------------------------------------------------------------------------------------')
            
            #compute_cutoff_auc(global_cost_vector_hcpya_T2, global_test_cost_vector_hcpya_T2, cost, reg_type, 'hrT2', 'global') # T2 to MNI
            compute_cutoff_auc(local_cost_vector_hcpya_T2, local_test_cost_vector_hcpya_T2, cost, reg_type, 'hrT2', 'local') # T2 to MNI
            print('----------------------------------------------------------------------------------------------------')
            
            if reg_type == 'align':
                #compute_cutoff_auc(global_cost_vector_hcpya_T2toT1, global_test_cost_vector_hcpya_T2toT1, cost, 'T1', 'hrT2', 'global') # T2 brain to T1 brain
                compute_cutoff_auc(local_cost_vector_hcpya_T2toT1, local_test_cost_vector_hcpya_T2toT1, cost, 'T1', 'hrT2', 'local') # T2 brain to T1 brain
        
